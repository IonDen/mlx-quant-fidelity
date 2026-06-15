"""Teacher-forced, streaming KV-quant fidelity probe."""

from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING

import mlx.core as mx

from mlx_quant_fidelity._memory_caps import install_memory_caps
from mlx_quant_fidelity.errors import (
    CacheNotQuantizableError,
    CorpusError,
    QuantFidelityError,
)
from mlx_quant_fidelity.policy import verdict_for
from mlx_quant_fidelity.probes._paired import _aggregate_chunks, _check_exact_zero, _reduce_pair
from mlx_quant_fidelity.report import FidelityReport

if TYPE_CHECKING:
    from mlx_quant_fidelity.corpora.provenance import Corpus


def _kv_head_dim(model: object) -> int | None:
    """Best-effort per-head KV dim for the group-size gate. None if not derivable.

    ``ModelArgs.head_dim`` is frequently None (llama populates the real value as an
    Attention-layer local, not on args), so the hidden//heads fallback is the primary
    path. Field names are llama-family-specific; unusual archs return None.
    """
    args = getattr(model, "args", None)
    head_dim = getattr(args, "head_dim", None)
    if head_dim:
        return int(head_dim)
    hidden = getattr(args, "hidden_size", None)
    heads = getattr(args, "num_attention_heads", None)
    if hidden and heads:
        return int(hidden) // int(heads)
    return None


def _head_dim_gate(*, head_dim: int | None, kv_group_size: int, model_type: str) -> str | None:
    """Validate kv_group_size divides the per-head KV dim. Raise if it doesn't; warn if unknown.

    The quantity is the per-head head_dim (QuantizedKVCache quantizes the per-head last axis;
    mlx_lm/models/cache.py). Returns a warning string when head_dim can't be derived (do not
    silently pass on odd architectures), else None.
    """
    if head_dim is None:
        return (
            f"head_dim/kv_group_size compatibility unverified for '{model_type}'; "
            "relying on MLX to surface a mismatch at first use."
        )
    if head_dim % kv_group_size != 0:
        raise CacheNotQuantizableError(
            f"kv_group_size={kv_group_size} does not divide the model's KV head_dim={head_dim}; "
            f"choose a group size that divides {head_dim} (e.g. 32 or 64)."
        )
    return None


def _cache_is_quantizable(cache: list[object], *, group_size: int, bits: int) -> bool:
    """Return True if every layer cache has a NON-RAISING to_quantized; else raise, naming the type.

    Validates ``bits`` up front (MLX only supports 2/3/4/6/8). An attribute check alone is
    insufficient: RotatingKVCache / BatchRotatingKVCache (sliding-window) HAVE to_quantized but
    raise NotImplementedError. So probe by actually calling it on the (empty, cheap) cache.
    """
    if bits not in (2, 3, 4, 6, 8):
        raise CacheNotQuantizableError(f"unsupported kv_bits={bits}; MLX supports 2/3/4/6/8.")
    for layer in cache:
        to_q = getattr(layer, "to_quantized", None)
        if to_q is None:
            raise CacheNotQuantizableError(
                f"cache layer {type(layer).__name__} has no to_quantized; "
                "this model's KV cache cannot be quantized (e.g. sliding-window / MLA)."
            )
        try:
            to_q(group_size=group_size, bits=bits)
        except NotImplementedError as exc:
            raise CacheNotQuantizableError(
                f"cache layer {type(layer).__name__} declares to_quantized but it is NYI: {exc}"
            ) from exc
        except (ValueError, RuntimeError) as exc:
            raise CacheNotQuantizableError(
                f"cache layer {type(layer).__name__} cannot quantize at "
                f"group_size={group_size}, bits={bits}: {exc}"
            ) from exc
    return True


def _score_chunk(
    model: object,
    ids: mx.array,
    ref_cache: list[object],
    quant_cache: list[object],
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """One teacher-forced chunk: forward twice on identical tokens, reduce to per-position scalars.

    Returns (kl[positions-1], flips[positions-1], ref_nll[positions-1], quant_nll[positions-1]).
    The caller must mx.eval the returns and let the vocab-wide logits leave scope before the
    next chunk; this function holds no logits beyond its own frame.
    """
    inp = ids[None, :-1]
    targets = ids[1:]
    ref_logits = model(inp, cache=ref_cache)[0].astype(mx.float32)  # type: ignore[operator]
    quant_logits = model(inp, cache=quant_cache)[0].astype(mx.float32)  # type: ignore[operator]
    return _reduce_pair(ref_logits, quant_logits, targets)


def measure_kv_fidelity(
    model_id: str,
    *,
    kv_bits: int = 4,
    kv_group_size: int = 64,
    quantize_start: int = 0,
    corpus: Corpus | None = None,
    max_chunks: int | None = None,
    model_revision: str | None = None,
) -> FidelityReport:
    """Measure how much KV-cache quantization costs, via teacher-forced paired scoring.

    Args:
        model_id: HuggingFace model ID (e.g. ``mlx-community/Llama-3.2-1B-Instruct-4bit``).
        kv_bits: KV-cache quantization bits (default 4).
        kv_group_size: KV-cache quantization group size (default 64).
        quantize_start: Token position where quantization begins. Only 0 (stress mode) is
            supported in 0.1.0; any other value raises ``QuantFidelityError``.
        corpus: Pre-built corpus to score. If None, WikiText-2 test split is fetched (requires
            network access and the ``--run-network`` marker in tests).
        max_chunks: Maximum number of corpus chunks to score.
        model_revision: HuggingFace model revision (commit SHA or tag).

    Returns:
        A :class:`~mlx_quant_fidelity.report.FidelityReport` with all metrics and provenance.

    Raises:
        QuantFidelityError: If ``quantize_start != 0`` (deployment mode, not in 0.1.0).
        CacheNotQuantizableError: If the model's KV cache does not support quantization.
        ExactZeroError: If KLD and flip rate are exactly 0 (quantization did not engage).
    """
    from mlx_lm import load
    from mlx_lm.models.cache import QuantizedKVCache, make_prompt_cache

    if quantize_start != 0:
        raise QuantFidelityError(
            "deployment mode (quantize_start > 0) is not implemented in 0.1.0; "
            "0.1.0 measures stress mode only (quantize_start=0)."
        )
    if max_chunks is not None and max_chunks < 1:
        raise CorpusError(f"max_chunks must be >= 1 (got {max_chunks}).")
    if corpus is not None and len(corpus.chunks) == 0:
        raise CorpusError("the provided corpus has no chunks; at least one is required.")
    install_memory_caps()  # must precede model load
    _loaded = load(model_id, revision=model_revision)  # pragma: no cover
    model, tokenizer = _loaded[0], _loaded[1]  # pragma: no cover
    probe_warnings: list[str] = []
    model_type = str(getattr(getattr(model, "args", None), "model_type", "unknown"))
    head_dim_warning = _head_dim_gate(
        head_dim=_kv_head_dim(model), kv_group_size=kv_group_size, model_type=model_type
    )
    if head_dim_warning is not None:  # pragma: no cover
        probe_warnings.append(head_dim_warning)

    if corpus is None:  # pragma: no cover
        from mlx_quant_fidelity.corpora.wikitext import load_wikitext2

        corpus = load_wikitext2(tokenizer, max_chunks=max_chunks, tokenizer_id=model_id)
        if len(corpus.chunks) == 0:
            raise CorpusError("the evaluation corpus yielded no chunks; at least one is required.")

    probe_cache = make_prompt_cache(model)  # pragma: no cover
    n_layers = len(probe_cache)  # pragma: no cover
    _cache_is_quantizable(probe_cache, group_size=kv_group_size, bits=kv_bits)  # pragma: no cover
    del probe_cache  # pragma: no cover

    kls: list[mx.array] = []  # pragma: no cover
    flips: list[mx.array] = []  # pragma: no cover
    ref_nlls: list[mx.array] = []  # pragma: no cover
    quant_nlls: list[mx.array] = []  # pragma: no cover
    for ids in corpus.chunks:  # pragma: no cover
        ref_cache = make_prompt_cache(model)
        # stress mode (quantize_start=0): QuantizedKVCache from token 0 — validated by the spike.
        quant_cache: list[object] = [
            QuantizedKVCache(group_size=kv_group_size, bits=kv_bits) for _ in range(n_layers)
        ]
        kl, flip, ref_nll, quant_nll = _score_chunk(model, ids, ref_cache, quant_cache)
        mx.eval(kl, flip, ref_nll, quant_nll)
        kls.append(kl)
        flips.append(flip)
        ref_nlls.append(ref_nll)
        quant_nlls.append(quant_nll)
        del ref_cache, quant_cache  # drop refs before next chunk
        mx.clear_cache()

    agg = _aggregate_chunks(kls, flips, ref_nlls, quant_nlls)  # pragma: no cover
    _check_exact_zero(  # pragma: no cover
        kl_mean=agg.kl.mean,
        flip_rate=agg.flip_rate,
        context=(
            f"quantization did not engage (quantize_start={quantize_start}; chunk may be "
            "shorter than the keep-first-N boundary, or the quantized cache was bypassed)"
        ),
    )

    return FidelityReport(  # pragma: no cover
        model_id=model_id,
        model_revision=model_revision,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantize_start=quantize_start,
        quantize_mode="stress" if quantize_start == 0 else "deployment",
        kl=agg.kl,
        flip_rate=agg.flip_rate,
        perplexity_ref=agg.perplexity_ref,
        perplexity_quant=agg.perplexity_quant,
        perplexity_delta=agg.perplexity_quant - agg.perplexity_ref,
        n_positions=agg.n_positions,
        n_chunks=len(corpus.chunks),
        corpus=corpus.provenance,
        mlx_version=importlib.metadata.version("mlx"),
        mlx_lm_version=importlib.metadata.version("mlx-lm"),
        peak_memory_bytes=int(mx.get_peak_memory()),
        cache_supported=True,
        verdict=verdict_for(agg.kl.mean, agg.kl.p99, agg.flip_rate),
        warnings=tuple(probe_warnings),
    )
