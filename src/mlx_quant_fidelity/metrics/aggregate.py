"""Host-side scalar aggregation of per-position metric arrays (small; runs on numpy)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ScalarSummary:
    """Mean / median / p99 / max of a per-position metric. The tail (p99/max) is load-bearing."""

    mean: float
    median: float
    p99: float
    max: float


def _p99_linear(values: np.ndarray) -> float:
    """Linear-interpolation 99th percentile that stays exact when +inf is present.

    numpy computes ``lo + (hi - lo) * t`` and gets NaN from ``inf - inf`` even when ``t == 0``
    (a lone +inf above the percentile index must not poison it). Here a zero interpolation
    weight returns the lower neighbour, and a positive weight toward +inf returns +inf.
    """
    if np.isnan(values).any():
        return float("nan")
    if np.isfinite(values).all():
        # Finite data: numpy's own percentile, bit-identical to 0.9.0 reports.
        return float(np.percentile(values, 99))
    ordered = np.sort(values.astype(np.float64, copy=False))
    idx = (ordered.size - 1) * 0.99
    lo_i = int(np.floor(idx))
    hi_i = int(np.ceil(idx))
    lo, hi = float(ordered[lo_i]), float(ordered[hi_i])
    t = idx - lo_i
    if t == 0 or lo == hi:
        return lo
    if np.isposinf(hi):
        return float("inf")
    return lo + (hi - lo) * t


def summarize(values: np.ndarray) -> ScalarSummary:
    """Reduce a 1-D per-position array to a ScalarSummary (linear-interpolated p99)."""
    values = np.asarray(values)  # tolerate mx.array / array-like callers (host-side reduction)
    return ScalarSummary(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p99=_p99_linear(values),
        max=float(np.max(values)),
    )
