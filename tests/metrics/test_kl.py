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


def _np_kl(ref, quant):
    """First-principles KL(softmax(ref) || softmax(quant)) in float64, independent of the code."""
    import numpy as np

    p = np.exp(np.array(ref, dtype=np.float64))
    p /= p.sum()
    q = np.exp(np.array(quant, dtype=np.float64))
    q /= q.sum()
    return float((p * np.log(p / q)).sum())


def test_kl_is_shift_invariant():
    """Reds if the logits are not normalized (log_softmax dropped): +7 on one side is then a
    constant 7-nat error instead of nothing. Softmax is invariant to adding a constant."""
    ref = mx.array([[2.0, 1.0, 0.5]], dtype=mx.float32)
    assert abs(float(kl_divergence(ref, ref + 7.0)[0])) < 1e-6
    assert abs(float(kl_divergence(ref + 7.0, ref)[0])) < 1e-6
    # and a shifted DIFFERENT pair still reports the true divergence
    other = mx.array([[1.5, 1.0, 1.0]], dtype=mx.float32)
    want = _np_kl([2.0, 1.0, 0.5], [1.5, 1.0, 1.0])
    assert math.isclose(float(kl_divergence(ref + 7.0, other - 3.0)[0]), want, abs_tol=1e-6)


def test_kl_bf16_input_matches_fp32():
    """Reds if the fp32 upcast is dropped: bf16 has 8 significant bits (relative eps 2^-8), so
    log_softmax run in bf16 is off by ~2^-8 * |logit| ~ 4e-3..8e-3 nats on these logits.

    The logits below are exactly representable in bf16, so the fp32 upcast makes the bf16 call
    the same fp32 computation as the fp32 call; agreement must be at fp32 precision (1e-6), far
    tighter than the bf16 error a missing upcast would leave.
    """
    ref = [[2.0, 1.0, 0.5]]
    quant = [[1.5, 1.0, 1.0]]
    got = kl_divergence(mx.array(ref, dtype=mx.bfloat16), mx.array(quant, dtype=mx.bfloat16))
    assert got.dtype == mx.float32
    assert math.isclose(float(got[0]), _np_kl(ref[0], quant[0]), abs_tol=1e-6)
