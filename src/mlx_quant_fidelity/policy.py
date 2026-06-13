"""Pure threshold -> verdict layer (the badge's brain). Separate from rendering.

Thresholds are conservative for 0.1.0 and will be tuned in the north-star release.
A result is only "good" if the mean drift, the tail (p99), AND the flip rate are all low.
"""

# nats / fraction thresholds (worst-passing value for each tier)
_GOOD = {"kl_mean": 0.01, "kl_p99": 0.10, "flip_rate": 0.01}
_MARGINAL = {"kl_mean": 0.10, "kl_p99": 1.00, "flip_rate": 0.05}


def verdict_for(kl_mean: float, kl_p99: float, flip_rate: float) -> str:
    """Return 'good' | 'marginal' | 'bad' from KLD mean+tail and flip rate."""
    if (
        kl_mean <= _GOOD["kl_mean"]
        and kl_p99 <= _GOOD["kl_p99"]
        and flip_rate <= _GOOD["flip_rate"]
    ):
        return "good"
    if (
        kl_mean <= _MARGINAL["kl_mean"]
        and kl_p99 <= _MARGINAL["kl_p99"]
        and flip_rate <= _MARGINAL["flip_rate"]
    ):
        return "marginal"
    return "bad"
