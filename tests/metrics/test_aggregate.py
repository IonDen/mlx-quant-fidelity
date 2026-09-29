import math

import numpy as np
import pytest

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


def test_summarize_legal_inf_kl_gives_inf_tail_without_warning():
    """Bug: numpy's linear percentile computes inf - inf * 0 on a small array with a legal +inf
    KL (the documented zero-probability policy), returning p99=NaN and warning."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = summarize(np.array([0.1, np.inf]))
    assert s.p99 == float("inf")
    assert s.max == float("inf")
    assert s.mean == float("inf")


def test_summarize_finite_input_is_unchanged_by_inf_handling():
    s = summarize(np.array([1.0, 2.0, 3.0, 4.0]))
    assert math.isclose(
        s.p99, 3.97, abs_tol=1e-9
    )  # linear interpolation, 0.99 * 3 = 2.97 -> 3 + 0.97*1


def test_summarize_p99_ignores_a_single_inf_that_sits_above_the_p99_index():
    """Bug: a lone +inf in 101 values reports p99=inf although the 99th percentile index lands
    exactly on the finite value 99.0 (interpolation weight 0 on the inf neighbour)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = summarize(np.append(np.arange(100.0), np.inf))
    assert s.p99 == 99.0
    assert s.max == float("inf")


def test_summarize_p99_is_inf_when_the_inf_neighbour_carries_weight():
    """n=100 with the last value inf: index 98.01 interpolates toward inf, so p99 is inf."""
    values = np.append(np.arange(99.0), np.inf)
    assert summarize(values).p99 == float("inf")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_p99_of_finite_values_is_bit_identical_to_numpy(dtype):
    """Bug: the hand-rolled interpolation drifts from np.percentile in the last bit on finite
    data, silently changing every 0.9.0 report's p99."""
    rng = np.random.default_rng(0)
    for n in (2, 3, 7, 100, 101, 997, 4095):
        values = rng.exponential(0.05, size=n).astype(dtype)
        assert summarize(values).p99 == float(np.percentile(values, 99))
