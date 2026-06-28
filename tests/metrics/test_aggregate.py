import math

import numpy as np

from mlx_quant_fidelity.metrics import ScalarSummary, summarize


def test_summarize_tail_distribution():
    # 90 positions at 0.01 + 10 at 2.0
    arr = np.array([0.01] * 90 + [2.0] * 10, dtype=np.float64)
    s = summarize(arr)
    assert isinstance(s, ScalarSummary)
    assert math.isclose(s.mean, 0.209, abs_tol=1e-6)
    assert math.isclose(s.median, 0.01, abs_tol=1e-9)
    assert math.isclose(s.p99, 2.0, abs_tol=1e-9)
    assert s.max == 2.0


def test_summarize_pins_p99_distinct_from_max():
    # 99 positions at 1.0 + one spike at 100.0: p99 (linear interp) = 1.99, max = 100.0.
    # The 90/10 fixture above has p99 == max == 2.0, so a `p99 = max(values)` shortcut passes
    # there; this fixture pins the tail (p99) as a separate quantity from the extreme (max).
    arr = np.array([1.0] * 99 + [100.0], dtype=np.float64)
    s = summarize(arr)
    assert s.max == 100.0
    assert math.isclose(s.p99, 1.99, abs_tol=1e-6)
    assert s.p99 < s.max
