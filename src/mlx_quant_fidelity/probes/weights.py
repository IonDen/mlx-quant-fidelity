"""Teacher-forced, streaming weight-quant fidelity probe (quantized repo vs reference repo)."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import mlx.core as mx

from mlx_quant_fidelity._memory_caps import install_memory_caps
from mlx_quant_fidelity.errors import (
    CorpusError,
    ExactZeroError,  # noqa: F401 — re-exported for callers
    InsufficientMemoryError,
    ModelMismatchError,
)
from mlx_quant_fidelity.policy import _WEIGHT_TIERS_v0_2_0, verdict_for
from mlx_quant_fidelity.probes._paired import _aggregate_chunks, _check_exact_zero, _reduce_pair
from mlx_quant_fidelity.report import WeightFidelityReport

if TYPE_CHECKING:
    from mlx_quant_fidelity.corpora.provenance import Corpus

TOKENIZER_ASSUMPTION_WARNING = (
    "Assumes both repos share a tokenizer; only vocab_size (and bos/eos ids) were checked — "
    "a divergent token map would be measured as quant drift."
)


@dataclass(frozen=True, slots=True)
class QuantMeta:
    """Declared quantization parameters read from a model config."""

    bits: int | None
    group_size: int | None
    mode: str | None
    per_layer: bool


def _int_or_none(v: object) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _str_or_none(v: object) -> str | None:
    return v if isinstance(v, str) else None


def _quant_meta_from_native(block: dict[str, object]) -> QuantMeta:
    reserved = {"group_size", "bits", "mode"}
    per_layer = any(isinstance(v, dict) for k, v in block.items() if k not in reserved)
    return QuantMeta(
        bits=_int_or_none(block.get("bits")),
        group_size=_int_or_none(block.get("group_size")),
        mode=_str_or_none(block.get("mode")) if "mode" in block else "affine",
        per_layer=per_layer,
    )


def extract_quant_meta(config: dict[str, object]) -> QuantMeta | None:
    """Read quantization metadata across the real mlx-lm/HF config shapes, or None if not quantized.

    Checks, in order: native ``config["quantization"]`` (the post-`load` canonical form),
    HF ``config["quantization_config"]`` (legacy ``quant_method`` like awq/gptq/mxfp4),
    and the same keys nested under ``config["text_config"]`` (multimodal repos).
    """
    quant = config.get("quantization")
    if isinstance(quant, dict):
        return _quant_meta_from_native(cast("dict[str, object]", quant))
    hf = config.get("quantization_config")
    if isinstance(hf, dict):
        hf_block = cast("dict[str, object]", hf)
        return QuantMeta(
            bits=_int_or_none(hf_block.get("bits")),
            group_size=_int_or_none(hf_block.get("group_size")),
            mode=_str_or_none(hf_block.get("quant_method")),
            per_layer=False,
        )
    text = config.get("text_config")
    if isinstance(text, dict):
        return extract_quant_meta(cast("dict[str, object]", text))
    return None


def _top_or_text(config: dict[str, object], key: str) -> object:
    """Read a config key from the top level, falling back to a nested text_config."""
    if key in config:
        return config[key]
    text = config.get("text_config")
    return text.get(key) if isinstance(text, dict) else None


def _gate_configs(
    *, quant_config: dict[str, object], reference_config: dict[str, object]
) -> tuple[QuantMeta, int | None]:
    """Validate the quant/reference pair is comparable. Returns (quant_meta, reference_bits).

    Raises ModelMismatchError if:
    - the quant repo isn't quantized or carries no 'bits' (bits=None produces a misleading report),
    - model_type is absent from either config or differs between them,
    - vocab_size is absent from either config or differs between them.
    """
    quant_meta = extract_quant_meta(quant_config)
    if quant_meta is None:
        raise ModelMismatchError("quant repo declares no quantization — nothing to measure.")
    if quant_meta.bits is None:
        raise ModelMismatchError(
            "quant repo's quantization block carries no 'bits' — cannot determine bit-width."
        )
    qt, rt = _top_or_text(quant_config, "model_type"), _top_or_text(reference_config, "model_type")
    if qt is None or rt is None or qt != rt:
        raise ModelMismatchError(f"model_type mismatch: quant={qt!r} vs reference={rt!r}.")
    qv, rv = _top_or_text(quant_config, "vocab_size"), _top_or_text(reference_config, "vocab_size")
    if qv is None or rv is None or qv != rv:
        raise ModelMismatchError(f"vocab_size mismatch: quant={qv} vs reference={rv}.")
    reference_meta = extract_quant_meta(reference_config)
    reference_bits = reference_meta.bits if reference_meta is not None else None
    return quant_meta, reference_bits


def _tokenizer_warnings(ref_tok: object, quant_tok: object) -> list[str]:
    """The standing tokenizer-assumption warning, plus a bos/eos-mismatch warning when they differ."""
    mismatches = [
        f"tokenizer {attr} differs between the repos — the comparison may be invalid."
        for attr in ("bos_token_id", "eos_token_id")
        if getattr(ref_tok, attr, None) != getattr(quant_tok, attr, None)
    ]
    return [TOKENIZER_ASSUMPTION_WARNING, *mismatches]


_PREFLIGHT_HEADROOM_BYTES = 2 * 1024**3


def _resolve_weight_bytes(repo: str, revision: str | None) -> int | None:
    """Sum the repo's `model*.safetensors` bytes (local dir or already-cached snapshot), or None.

    Best-effort: returns None for an uncached HF repo (local_files_only) or any error, so a
    measurement is never aborted by a missing byte count. Mirrors mlx-lm's `model*.safetensors`
    glob (utils.py) — not all `*.safetensors`.
    """
    try:
        path = Path(repo)
        if not (path.exists() and path.is_dir()):
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(repo, revision=revision, local_files_only=True))
        total = sum(f.stat().st_size for f in path.glob("model*.safetensors"))
        return total or None
    except Exception:
        return None


def _preflight_memory(*, quant_bytes: int | None, reference_bytes: int | None) -> None:
    """Raise InsufficientMemoryError if both models + headroom exceed the device working set.

    Best-effort: skips silently when either size is unknown or the device reports no working set.
    The wired cap does NOT bound two live, un-evictable models, so this byte check is the real guard.
    """
    if quant_bytes is None or reference_bytes is None:
        return
    try:
        max_ws = int(mx.device_info().get("max_recommended_working_set_size", 0))
    except Exception:
        return
    if max_ws <= 0:
        return
    if quant_bytes + reference_bytes + _PREFLIGHT_HEADROOM_BYTES > max_ws:
        raise InsufficientMemoryError(
            f"reference ({reference_bytes / 1e9:.1f} GB) + quant ({quant_bytes / 1e9:.1f} GB) + "
            f"headroom exceed the device working set ({max_ws / 1e9:.1f} GB); "
            "use smaller models or a larger machine."
        )


def _score_weight_chunk(
    ref_model: object, quant_model: object, ids: mx.array
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """One teacher-forced chunk: forward BOTH models on identical tokens (no cache), reduce.

    Returns (kl, flips, ref_nll, quant_nll). The `[0]` strips the batch-of-1 axis: an mlx-lm
    model returns a single [1, L-1, vocab] array (NOT a tuple, and NOT eval_ppl's batched shape).
    """
    inp = ids[None, :-1]
    targets = ids[1:]
    ref_logits = ref_model(inp)[0].astype(mx.float32)  # type: ignore[operator]
    quant_logits = quant_model(inp)[0].astype(mx.float32)  # type: ignore[operator]
    return _reduce_pair(ref_logits, quant_logits, targets)


def measure_weight_fidelity(
    quant_model_id: str,
    reference_model_id: str,
    *,
    corpus: Corpus | None = None,
    max_chunks: int | None = None,
    quant_revision: str | None = None,
    reference_revision: str | None = None,
) -> WeightFidelityReport:
    """Measure how much weight quantization costs: a quantized repo vs a reference repo.

    Both models are scored teacher-forced on identical corpus tokens (no KV cache). The drift
    bundles quantized-matmul kernel numerics (the deployed model's real cost), with no
    quantized-attention confound. Raises ModelMismatchError (incomparable pair),
    InsufficientMemoryError (pair too large for the device), ExactZeroError (identical repos).

    Args:
        quant_model_id: Local path or HF repo id of the quantized model.
        reference_model_id: Local path or HF repo id of the full-precision (or lower-quant) reference.
        corpus: Pre-built corpus to score; if None, WikiText-2 is loaded automatically.
        max_chunks: Score at most this many corpus chunks (applies to both the auto-loaded
            and a caller-provided corpus).
        quant_revision: Optional git revision for the quantized repo.
        reference_revision: Optional git revision for the reference repo.
    """
    from mlx_lm import load

    if max_chunks is not None and max_chunks < 1:
        raise CorpusError(f"max_chunks must be >= 1 (got {max_chunks}).")
    if corpus is not None and len(corpus.chunks) == 0:
        raise CorpusError("the provided corpus has no chunks; at least one is required.")
    if corpus is not None and any(c.shape[0] < 2 for c in corpus.chunks):
        raise CorpusError(
            "every corpus chunk must have at least 2 tokens (one teacher-forced position)."
        )
    install_memory_caps()  # before any model load

    reference_bytes = _resolve_weight_bytes(reference_model_id, reference_revision)
    quant_bytes = _resolve_weight_bytes(quant_model_id, quant_revision)
    _preflight_memory(quant_bytes=quant_bytes, reference_bytes=reference_bytes)

    ref_model, tokenizer, reference_config = load(  # type: ignore[misc]
        reference_model_id, revision=reference_revision, return_config=True
    )
    quant_model, quant_tok, quant_config = load(  # type: ignore[misc]
        quant_model_id, revision=quant_revision, return_config=True
    )
    quant_meta, reference_bits = _gate_configs(
        quant_config=quant_config, reference_config=reference_config
    )
    warnings = [*_tokenizer_warnings(tokenizer, quant_tok)]
    if reference_bits is not None:
        warnings.append(
            f"reference is itself {reference_bits}-bit, not full precision; drift is relative to it."
        )
    # re-resolve now that snapshots are cached, for the report (pre-flight may have seen None)
    quant_bytes = quant_bytes or _resolve_weight_bytes(quant_model_id, quant_revision)
    reference_bytes = reference_bytes or _resolve_weight_bytes(
        reference_model_id, reference_revision
    )

    if corpus is None:  # pragma: no cover - real-corpus/network path, covered by --run-slow
        from mlx_quant_fidelity.corpora.wikitext import load_wikitext2

        corpus = load_wikitext2(tokenizer, max_chunks=max_chunks, tokenizer_id=reference_model_id)
        if len(corpus.chunks) == 0:
            raise CorpusError("the evaluation corpus yielded no chunks; at least one is required.")

    # Cap the scored chunks — applies to both an auto-loaded and a caller-provided corpus.
    chunks = corpus.chunks[:max_chunks] if max_chunks is not None else corpus.chunks
    kls: list[mx.array] = []
    flips: list[mx.array] = []
    ref_nlls: list[mx.array] = []
    quant_nlls: list[mx.array] = []
    for ids in chunks:
        kl, flip, ref_nll, quant_nll = _score_weight_chunk(ref_model, quant_model, ids)
        mx.eval(kl, flip, ref_nll, quant_nll)
        kls.append(kl)
        flips.append(flip)
        ref_nlls.append(ref_nll)
        quant_nlls.append(quant_nll)
        mx.clear_cache()

    agg = _aggregate_chunks(kls, flips, ref_nlls, quant_nlls)
    _check_exact_zero(
        kl_mean=agg.kl.mean,
        flip_rate=agg.flip_rate,
        context=(
            "identical distributions — the two repos are numerically identical "
            "(reference == quant, or the same artifact twice); nothing to measure"
        ),
    )
    return WeightFidelityReport(
        quant_model_id=quant_model_id,
        quant_revision=quant_revision,
        reference_model_id=reference_model_id,
        reference_revision=reference_revision,
        quant_bits=quant_meta.bits,
        quant_group_size=quant_meta.group_size,
        quant_mode=quant_meta.mode,
        per_layer=quant_meta.per_layer,
        reference_bits=reference_bits,
        kl=agg.kl,
        flip_rate=agg.flip_rate,
        perplexity_ref=agg.perplexity_ref,
        perplexity_quant=agg.perplexity_quant,
        perplexity_delta=agg.perplexity_quant - agg.perplexity_ref,
        n_positions=agg.n_positions,
        n_chunks=len(chunks),
        corpus=corpus.provenance,
        mlx_version=importlib.metadata.version("mlx"),
        mlx_lm_version=importlib.metadata.version("mlx-lm"),
        peak_memory_bytes=int(mx.get_peak_memory()),
        quant_model_bytes=quant_bytes,
        reference_model_bytes=reference_bytes,
        verdict=verdict_for(
            agg.kl.mean, agg.kl.p99, agg.flip_rate, thresholds=_WEIGHT_TIERS_v0_2_0
        ),
        warnings=tuple(warnings),
    )
