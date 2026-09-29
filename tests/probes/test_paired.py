"""Non-finite guards in the shared paired-run aggregation."""

import mlx.core as mx
import pytest

from mlx_quant_fidelity.errors import NonFiniteMetricError, QuantFidelityError
from mlx_quant_fidelity.probes._paired import _aggregate_chunks

_NO_FLIP = [mx.array([False, False])]
_ONES = [mx.array([1.0, 1.0])]


def test_aggregate_rejects_nan_kl():
    """Bug: a NaN per-position KL slips past the exact-zero guard and ranks as a real number."""
    with pytest.raises(NonFiniteMetricError):
        _aggregate_chunks([mx.array([0.1, float("nan")])], _NO_FLIP, _ONES, _ONES)


def test_aggregate_allows_inf_kl():
    """Bug: refusing +inf would break the documented zero-probability KL policy."""
    # 200 finite positions so numpy's percentile interpolation never touches the inf
    n = 201
    kl = mx.array([0.1] * (n - 1) + [float("inf")])
    agg = _aggregate_chunks(
        [kl], [mx.zeros((n,), dtype=mx.bool_)], [mx.ones((n,))], [mx.ones((n,))]
    )
    assert agg.kl.max == float("inf")


def test_aggregate_rejects_nan_nll():
    """Bug: NaN NLL silently yields a NaN perplexity in the report."""
    with pytest.raises(NonFiniteMetricError):
        _aggregate_chunks([mx.array([0.1, 0.2])], _NO_FLIP, [mx.array([1.0, float("nan")])], _ONES)


def test_aggregate_rejects_inf_nll():
    with pytest.raises(NonFiniteMetricError):
        _aggregate_chunks([mx.array([0.1, 0.2])], _NO_FLIP, _ONES, [mx.array([1.0, float("inf")])])


def test_non_finite_error_is_package_rooted():
    assert issubclass(NonFiniteMetricError, QuantFidelityError)
