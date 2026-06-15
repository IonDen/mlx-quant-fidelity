"""Pure threshold -> verdict layer (the badge's brain). Separate from rendering.

A result is only "good" if the mean drift, the tail (p99), AND the flip rate are all low.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Tiers:
    """Worst-passing values for the 'good' and 'marginal' tiers (nats / fraction)."""

    good: dict[str, float]
    marginal: dict[str, float]


# KV tiers (unchanged from 0.1.0; calibrated for KV-cache quantization).
_KV_TIERS = Tiers(
    good={"kl_mean": 0.01, "kl_p99": 0.10, "flip_rate": 0.01},
    marginal={"kl_mean": 0.10, "kl_p99": 1.00, "flip_rate": 0.05},
)

# Weight tiers: PROVISIONAL (WikiText-2 short prose, not validated against downstream accuracy).
# Calibrated from the 0.2.0 sample reports (Llama-3.2-1B/3B 4bit+8bit vs bf16, Qwen2.5-7B q4-vs-q8):
# q8 lands "good" (kl_mean ~1e-3, ~2% flips), q4 "marginal" (kl_mean ~0.08-0.15, 15-21% flips),
# lower bit-widths "bad". Weight quant drifts more than KV, so the ceilings are looser.
_WEIGHT_TIERS_v0_2_0 = Tiers(
    good={"kl_mean": 0.01, "kl_p99": 0.10, "flip_rate": 0.05},
    marginal={"kl_mean": 0.20, "kl_p99": 1.50, "flip_rate": 0.25},
)


def verdict_for(
    kl_mean: float, kl_p99: float, flip_rate: float, *, thresholds: Tiers = _KV_TIERS
) -> str:
    """Return 'good' | 'marginal' | 'bad' from KLD mean+tail and flip rate, against `thresholds`."""
    g, m = thresholds.good, thresholds.marginal
    if kl_mean <= g["kl_mean"] and kl_p99 <= g["kl_p99"] and flip_rate <= g["flip_rate"]:
        return "good"
    if kl_mean <= m["kl_mean"] and kl_p99 <= m["kl_p99"] and flip_rate <= m["flip_rate"]:
        return "marginal"
    return "bad"
