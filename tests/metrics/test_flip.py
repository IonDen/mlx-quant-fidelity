import mlx.core as mx

from mlx_quant_fidelity.metrics import top_token_flips


def test_flip_rate_matches_n_diffs_over_n_tokens():
    # 4 positions; argmax differs at exactly 1 -> rate 0.25
    ref = mx.array([[9.0, 0.0], [9.0, 0.0], [9.0, 0.0], [0.0, 9.0]])
    quant = mx.array([[9.0, 0.0], [9.0, 0.0], [9.0, 0.0], [9.0, 0.0]])
    flips = top_token_flips(ref, quant)
    assert flips.shape == (4,)
    assert float(flips.astype(mx.float32).mean()) == 0.25


def test_identity_no_flips():
    lg = mx.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    assert int(top_token_flips(lg, lg).astype(mx.int32).sum()) == 0
