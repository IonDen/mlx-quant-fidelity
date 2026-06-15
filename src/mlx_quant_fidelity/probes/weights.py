"""Teacher-forced, streaming weight-quant fidelity probe (quantized repo vs reference repo)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import mlx.core as mx

from mlx_quant_fidelity.errors import ModelMismatchError
from mlx_quant_fidelity.probes._paired import _reduce_pair

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

    Raises ModelMismatchError if the quant repo isn't quantized, or model_type / vocab_size differ.
    """
    quant_meta = extract_quant_meta(quant_config)
    if quant_meta is None:
        raise ModelMismatchError("quant repo declares no quantization — nothing to measure.")
    qt, rt = _top_or_text(quant_config, "model_type"), _top_or_text(reference_config, "model_type")
    if qt != rt:
        raise ModelMismatchError(f"model_type mismatch: quant={qt!r} vs reference={rt!r}.")
    qv, rv = _top_or_text(quant_config, "vocab_size"), _top_or_text(reference_config, "vocab_size")
    if qv != rv:
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
