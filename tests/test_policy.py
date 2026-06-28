from mlx_quant_fidelity.policy import _WEIGHT_TIERS_v0_2_0, qualifies, tier_rank, verdict_for


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


def test_kv_marginal_boundary_is_pinned():
    # Literal thresholds (not _KV_TIERS.marginal) so the boundary is anchored: at the marginal
    # ceiling -> marginal; one ulp over the tail -> bad. Catches a relaxed marginal ceiling
    # (e.g. kl_p99 1.00 -> 1.50 would silently keep a worse measurement at "marginal").
    assert verdict_for(kl_mean=0.10, kl_p99=1.00, flip_rate=0.05) == "marginal"
    assert verdict_for(kl_mean=0.10, kl_p99=1.001, flip_rate=0.05) == "bad"


def test_weight_verdicts_by_threshold():
    # independent synthetic values bracketing the calibrated weight tiers (not the sample points)
    good = verdict_for(kl_mean=0.005, kl_p99=0.05, flip_rate=0.02, thresholds=_WEIGHT_TIERS_v0_2_0)
    marginal = verdict_for(
        kl_mean=0.10, kl_p99=0.8, flip_rate=0.15, thresholds=_WEIGHT_TIERS_v0_2_0
    )
    bad = verdict_for(kl_mean=0.5, kl_p99=3.0, flip_rate=0.4, thresholds=_WEIGHT_TIERS_v0_2_0)
    assert (good, marginal, bad) == ("good", "marginal", "bad")


def test_weight_good_requires_all_three_axes():
    # only the tail exceeds the good ceiling -> not good (still within the marginal tail ceiling)
    assert (
        verdict_for(kl_mean=0.005, kl_p99=0.5, flip_rate=0.02, thresholds=_WEIGHT_TIERS_v0_2_0)
        == "marginal"
    )


def test_weight_good_requires_kl_mean_and_flip_axes():
    # kl_mean alone over the good ceiling -> not good
    assert (
        verdict_for(kl_mean=0.10, kl_p99=0.05, flip_rate=0.02, thresholds=_WEIGHT_TIERS_v0_2_0)
        == "marginal"
    )
    # flip_rate alone over the good ceiling -> not good
    assert (
        verdict_for(kl_mean=0.005, kl_p99=0.05, flip_rate=0.15, thresholds=_WEIGHT_TIERS_v0_2_0)
        == "marginal"
    )


def test_weight_good_boundary_relaxation_side_is_pinned():
    # Literal thresholds: at the weight good ceiling -> good; one ulp over the tail -> marginal.
    # The existing axis tests sit far from the ceiling, so a relaxed good ceiling (kl_p99
    # 0.10 -> 0.20) would go undetected without this.
    assert (
        verdict_for(kl_mean=0.01, kl_p99=0.10, flip_rate=0.05, thresholds=_WEIGHT_TIERS_v0_2_0)
        == "good"
    )
    assert (
        verdict_for(kl_mean=0.01, kl_p99=0.101, flip_rate=0.05, thresholds=_WEIGHT_TIERS_v0_2_0)
        == "marginal"
    )


def test_weight_marginal_boundary_is_inclusive():
    # exactly at each marginal ceiling stays marginal (<=)
    assert (
        verdict_for(kl_mean=0.20, kl_p99=1.50, flip_rate=0.25, thresholds=_WEIGHT_TIERS_v0_2_0)
        == "marginal"
    )


def test_tier_rank_orders_verdicts():
    assert tier_rank("bad") < tier_rank("marginal") < tier_rank("good")


def test_qualifies_max_kld_only():
    assert qualifies(kl_mean=0.05, verdict="bad", max_kld=0.1, min_tier=None) is True
    assert qualifies(kl_mean=0.20, verdict="good", max_kld=0.1, min_tier=None) is False


def test_qualifies_min_tier_uses_full_verdict():
    # audit #2: a good mean but a 'marginal' verdict (bad tail/flip) must FAIL --min-tier good
    assert qualifies(kl_mean=0.001, verdict="marginal", max_kld=None, min_tier="good") is False
    assert qualifies(kl_mean=0.001, verdict="good", max_kld=None, min_tier="good") is True


def test_qualifies_anded_and_default_true():
    assert qualifies(kl_mean=0.05, verdict="good", max_kld=0.1, min_tier="good") is True
    assert qualifies(kl_mean=0.05, verdict="marginal", max_kld=0.1, min_tier="good") is False
    assert qualifies(kl_mean=9.9, verdict="bad", max_kld=None, min_tier=None) is True  # no budget
    assert qualifies(kl_mean=0.5, verdict="good", max_kld=0.1, min_tier="good") is False


def test_qualifies_min_tier_marginal_boundary():
    assert qualifies(kl_mean=0.05, verdict="marginal", max_kld=None, min_tier="marginal") is True
    assert qualifies(kl_mean=0.05, verdict="bad", max_kld=None, min_tier="marginal") is False


def test_tier_rank_rejects_unknown_verdict():
    import pytest

    from mlx_quant_fidelity.policy import tier_rank

    with pytest.raises(ValueError, match="unknown verdict"):
        tier_rank("nonsense")
