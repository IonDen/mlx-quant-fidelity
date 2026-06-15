from mlx_quant_fidelity.policy import _WEIGHT_TIERS_v0_2_0, verdict_for


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


def test_weight_verdicts_by_threshold():
    good = verdict_for(kl_mean=0.005, kl_p99=0.05, flip_rate=0.003, thresholds=_WEIGHT_TIERS_v0_2_0)
    marginal = verdict_for(
        kl_mean=0.08, kl_p99=0.4, flip_rate=0.03, thresholds=_WEIGHT_TIERS_v0_2_0
    )
    bad = verdict_for(kl_mean=0.8, kl_p99=4.0, flip_rate=0.3, thresholds=_WEIGHT_TIERS_v0_2_0)
    assert (good, marginal, bad) == ("good", "marginal", "bad")


def test_weight_good_requires_all_three_axes():
    # only the tail exceeds the good ceiling -> not good
    assert (
        verdict_for(kl_mean=0.005, kl_p99=2.0, flip_rate=0.003, thresholds=_WEIGHT_TIERS_v0_2_0)
        == "marginal"
    )
