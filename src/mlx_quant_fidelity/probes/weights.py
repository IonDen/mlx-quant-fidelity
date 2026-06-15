"""Teacher-forced, streaming weight-quant fidelity probe (quantized repo vs reference repo)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True, slots=True)
class QuantMeta:
    """Declared quantization parameters read from a model config."""

    bits: int | None
    group_size: int | None
    mode: str | None
    per_layer: bool


def _int_or_none(v: object) -> int | None:
    return v if isinstance(v, int) else None


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
