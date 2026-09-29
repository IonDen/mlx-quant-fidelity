"""One skip rule for every slow test that needs the pinned TurboQuant port."""

import pytest

from mlx_quant_fidelity.errors import MethodUnavailableError
from mlx_quant_fidelity.probes.kv_methods import TURBOQUANT_PINNED_COMMIT, _installed_commit


def port_or_skip(method: object) -> None:
    """Skip ONLY when the port is absent or not at the pinned commit.

    A ``MethodUnavailableError`` while the installed commit equals the pin is a real contract
    failure (the guard misreporting a correctly pinned port as unavailable) and must fail the
    lane loudly instead of reading as a routine skip. ``_installed_commit()`` returns
    ``"unknown"`` (never raises) when ``turboquant-mlx`` is not installed or its metadata is
    unreadable, so comparing it to the pin covers "absent" and "wrong commit" in one check.
    """
    try:
        method.probe_capability([])  # type: ignore[attr-defined]
    except MethodUnavailableError:
        installed = _installed_commit()
        if installed != TURBOQUANT_PINNED_COMMIT:
            pytest.skip(
                f"installed turboquant_mlx commit {installed!r} is not the pinned "
                f"{TURBOQUANT_PINNED_COMMIT}"
            )
        raise  # installed commit IS the pin: a real contract failure, not a routine skip
