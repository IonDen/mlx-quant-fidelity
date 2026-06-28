import math

import mlx.core as mx

from mlx_quant_fidelity.metrics import kl_divergence


def _logits(probs):
    return mx.log(mx.array(probs, dtype=mx.float32))


def test_identity_is_exactly_zero():
    lg = _logits([[0.2, 0.3, 0.5]])
    assert float(kl_divergence(lg, lg)[0]) == 0.0


def test_magnitude_sharp_vs_spread():
    # KL(P||Q), P=[.99,.005,.005], Q=[.005,.99,.005] -> 5.20895 nats (hand-computed)
    kl = kl_divergence(_logits([[0.99, 0.005, 0.005]]), _logits([[0.005, 0.99, 0.005]]))
    assert math.isclose(float(kl[0]), 5.20895, abs_tol=1e-3)


def test_direction_is_p_ref_to_q_quant():
    # NON-mirror pair: KL(P||Q)=0.39606, KL(Q||P)=0.36527 (hand-computed) -> distinct
    p, q = [[0.6, 0.3, 0.1]], [[0.2, 0.5, 0.3]]
    fwd = float(kl_divergence(_logits(p), _logits(q))[0])
    rev = float(kl_divergence(_logits(q), _logits(p))[0])
    assert math.isclose(fwd, 0.39606, abs_tol=1e-3)
    assert not math.isclose(fwd, rev, abs_tol=1e-2)


def test_zero_prob_is_inf_not_smoothed():
    # P certain where Q impossible -> +inf (honest), not an eps-smoothed near-zero.
    kl = kl_divergence(_logits([[1.0, 0.0, 0.0]]), _logits([[0.0, 1.0, 0.0]]))
    # Positive infinity specifically: math.isinf is also True for -inf, which a reversed-direction
    # SUT (log_q - log_p) would return here, so pin the sign.
    assert float(kl[0]) == math.inf


def test_per_position_shape():
    out = kl_divergence(_logits([[0.5, 0.5], [0.9, 0.1]]), _logits([[0.4, 0.6], [0.1, 0.9]]))
    assert out.shape == (2,)
