from mlx_quant_fidelity.policy import verdict_for


def test_verdicts_by_threshold():
    assert verdict_for(kl_mean=0.001, kl_p99=0.01, flip_rate=0.001) == "good"
    assert verdict_for(kl_mean=0.05, kl_p99=0.4, flip_rate=0.03) == "marginal"
    assert verdict_for(kl_mean=0.5, kl_p99=3.0, flip_rate=0.2) == "bad"


def test_good_requires_all_three_axes():
    # only the tail (p99) exceeds the good ceiling -> NOT good (conjunction, not just mean)
    assert verdict_for(kl_mean=0.001, kl_p99=0.5, flip_rate=0.001) == "marginal"


def test_good_boundary_is_inclusive():
    # exactly at each good ceiling stays good (<=)
    assert verdict_for(kl_mean=0.01, kl_p99=0.10, flip_rate=0.01) == "good"
