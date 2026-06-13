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
