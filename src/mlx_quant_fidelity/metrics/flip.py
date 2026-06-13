"""Top-token agreement: per-position top-1 disagreement on fp32 logits."""

import mlx.core as mx


def top_token_flips(ref_logits: mx.array, quant_logits: mx.array) -> mx.array:
    """Per-position bool: argmax(ref) != argmax(quant), on fp32 logits. Output [positions]."""
    ref_top = mx.argmax(ref_logits.astype(mx.float32), axis=-1)
    quant_top = mx.argmax(quant_logits.astype(mx.float32), axis=-1)
    return mx.array(ref_top != quant_top)
