"""`_require_finite`: which non-finite values each metric may legally carry."""

import numpy as np
import pytest

from mlx_quant_fidelity.errors import NonFiniteMetricError
from mlx_quant_fidelity.probes._paired import _require_finite


def test_kl_may_carry_positive_infinity():
    """The documented zero-probability policy: +inf KL is legal."""
    _require_finite("kl", np.array([0.1, np.inf]), allow_inf=True)


@pytest.mark.parametrize("bad", [np.nan, -np.inf])
def test_allow_inf_still_rejects_nan_and_negative_infinity(bad):
    """Bug: allow_inf only checked NaN, so a -inf KL (impossible for a KL; a broken kernel)
    passed the guard and was ranked."""
    with pytest.raises(NonFiniteMetricError):
        _require_finite("kl", np.array([0.1, bad]), allow_inf=True)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_default_rejects_every_non_finite_value(bad):
    with pytest.raises(NonFiniteMetricError):
        _require_finite("nll", np.array([1.0, bad]))
