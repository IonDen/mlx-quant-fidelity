"""The shared slow-lane port skip: skip when the port is missing or off-pin, fail when it is
the pinned commit and still unavailable."""

import pytest
from tests.probes import port_skip

from mlx_quant_fidelity.errors import MethodUnavailableError


class _Unavailable:
    def probe_capability(self, empty_cache):
        raise MethodUnavailableError("port not usable")


class _Available:
    def probe_capability(self, empty_cache):
        return None


def test_available_method_neither_skips_nor_raises():
    port_skip.port_or_skip(_Available())


def test_unavailable_off_pin_skips(monkeypatch):
    """Reds if a missing / wrong-commit port stops being a skip (the lane would fail spuriously)."""
    monkeypatch.setattr(port_skip, "_installed_commit", lambda: "unknown")
    with pytest.raises(pytest.skip.Exception):
        port_skip.port_or_skip(_Unavailable())


def test_unavailable_at_the_pinned_commit_raises(monkeypatch):
    """Reds if an unavailable port AT the pinned commit is skipped: a behavioural regression
    in the guard would then read as a routine skip."""
    monkeypatch.setattr(port_skip, "_installed_commit", lambda: port_skip.TURBOQUANT_PINNED_COMMIT)
    try:
        port_skip.port_or_skip(_Unavailable())
    except pytest.skip.Exception:
        pytest.fail("skipped at the pinned commit; must raise")  # a skip would hide the failure
    except MethodUnavailableError:
        return
    pytest.fail("neither skipped nor raised")
