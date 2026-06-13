"""Full-vocab KL divergence, KL(softmax(ref) || softmax(quant)), fp32, per position."""

import mlx.core as mx
import mlx.nn as nn


def kl_divergence(ref_logits: mx.array, quant_logits: mx.array) -> mx.array:
    """Per-position KL(P_ref || Q_quant) in nats. Inputs [positions, vocab], output [positions].

    Computed in fp32. Where P=0 the term is 0 (0*log0 := 0); where P>0 and Q->0 the term is
    +inf -- the honest "certain under ref, impossible under quant" cost, never eps-smoothed.
    softmax of real (finite) logits is strictly positive, so real runs are always finite.
    """
    log_p = nn.log_softmax(ref_logits.astype(mx.float32), axis=-1)  # type: ignore[attr-defined]
    log_q = nn.log_softmax(quant_logits.astype(mx.float32), axis=-1)  # type: ignore[attr-defined]
    p = mx.exp(log_p)
    terms = mx.where(p > 0, p * (log_p - log_q), mx.array(0.0, dtype=mx.float32))
    return terms.sum(axis=-1)
