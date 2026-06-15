"""Teacher-forced, streaming weight-quant fidelity probe (quantized repo vs reference repo)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import mlx.core as mx

from mlx_quant_fidelity.probes._paired import _reduce_pair


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
