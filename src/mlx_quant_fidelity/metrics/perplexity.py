"""Teacher-forced perplexity primitives."""

import math

import mlx.core as mx
import mlx.nn as nn


def token_nll(logits: mx.array, target_ids: mx.array) -> mx.array:
    """Per-position negative log-likelihood -log softmax(logits)[target]. fp32. Output [positions]."""
    log_probs = nn.log_softmax(logits.astype(mx.float32), axis=-1)
    chosen = mx.take_along_axis(log_probs, target_ids[:, None], axis=-1).squeeze(-1)
    return -chosen


def perplexity(nlls: mx.array) -> float:
    """exp(mean(nlls))."""
    return math.exp(float(nlls.mean()))


def perplexity_delta(ref_nlls: mx.array, quant_nlls: mx.array) -> float:
    """perplexity(quant) - perplexity(ref). Positive when the quant is more uncertain."""
    return perplexity(quant_nlls) - perplexity(ref_nlls)
