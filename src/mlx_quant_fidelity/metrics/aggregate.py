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


def summarize(values: np.ndarray) -> ScalarSummary:
    """Reduce a 1-D per-position array to a ScalarSummary (numpy linear percentile)."""
    values = np.asarray(values)  # tolerate mx.array / array-like callers (host-side reduction)
    return ScalarSummary(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p99=float(np.percentile(values, 99, method="linear")),
        max=float(np.max(values)),
    )
