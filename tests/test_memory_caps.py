import pytest

from mlx_quant_fidelity import _memory_caps
from mlx_quant_fidelity._memory_caps import device_string


def test_device_string_formats_name_and_memory(monkeypatch):
    monkeypatch.setattr(
        _memory_caps.mx,
        "device_info",
        lambda: {"device_name": "Apple M1 Max", "memory_size": 34359738368},
    )
    assert device_string() == "Apple M1 Max, 32 GB"


def test_device_string_none_when_unreported(monkeypatch):
    monkeypatch.setattr(_memory_caps.mx, "device_info", dict)
    assert device_string() is None


def test_device_string_none_when_device_info_raises(monkeypatch):
    def _boom():
        raise RuntimeError("no metal device")

    monkeypatch.setattr(_memory_caps.mx, "device_info", _boom)
    assert device_string() is None


def test_device_string_name_only_when_memory_size_absent(monkeypatch):
    monkeypatch.setattr(_memory_caps.mx, "device_info", lambda: {"device_name": "Apple M1 Max"})
    assert device_string() == "Apple M1 Max"


def test_device_string_name_only_when_memory_size_non_positive(monkeypatch):
    monkeypatch.setattr(
        _memory_caps.mx,
        "device_info",
        lambda: {"device_name": "Apple M1 Max", "memory_size": 0},
    )
    assert device_string() == "Apple M1 Max"


def test_clamp_uses_desired_on_large_device():
    # 25 GB recommended → desired (20, 22) fits unchanged
    assert _memory_caps._clamp_caps_gb(25) == (20, 22)


@pytest.mark.parametrize("max_recommended_gb", range(1, 41))
def test_clamp_never_exceeds_the_device_working_set(max_recommended_gb):
    """Spec rule: memory <= max_recommended and wired strictly below it (the wired cap is the
    panic guard) whenever caps are installed at all; a device too small for that gets (0, 0)."""
    wired, memory = _memory_caps._clamp_caps_gb(max_recommended_gb)
    if (wired, memory) == (0, 0):
        return
    assert 0 < wired < max_recommended_gb
    assert wired < memory <= max_recommended_gb


@pytest.mark.parametrize("max_recommended_gb", [1, 2])
def test_clamp_refuses_a_device_too_small_for_any_valid_cap(max_recommended_gb):
    """Bug: a 1 GB working set got wired=1 (a cap AT the working set, which is invalid)."""
    assert _memory_caps._clamp_caps_gb(max_recommended_gb) == (0, 0)


def test_caps_warning_fires_on_a_device_too_small_for_caps(monkeypatch):
    """A 1 GiB working set installs nothing; the report must say the run is unbounded."""
    monkeypatch.setattr(
        _memory_caps.mx, "device_info", lambda: {"max_recommended_working_set_size": 1024**3}
    )
    installed = _memory_caps.install_memory_caps()
    assert installed == (0, 0)
    assert _memory_caps.caps_warning(installed) is not None


def test_clamp_shrinks_wired_on_small_device():
    # 10 GB recommended -> wired = min(20, 10 - 2 headroom) = 8
    assert _memory_caps._clamp_caps_gb(10) == (8, 10)


def test_zero_recommended_is_noop_signal():
    assert _memory_caps._clamp_caps_gb(0) == (0, 0)


def test_compute_safe_caps_handles_device_info_failure(monkeypatch):
    def _boom():
        raise RuntimeError("no metal device")

    monkeypatch.setattr(_memory_caps.mx, "device_info", _boom)
    assert _memory_caps.compute_safe_caps_gb() == (0, 0)


def test_compute_safe_caps_zero_working_set(monkeypatch):
    monkeypatch.setattr(
        _memory_caps.mx, "device_info", lambda: {"max_recommended_working_set_size": 0}
    )
    assert _memory_caps.compute_safe_caps_gb() == (0, 0)


def test_install_memory_caps_noop_when_no_working_set(monkeypatch):
    monkeypatch.setattr(
        _memory_caps.mx, "device_info", lambda: {"max_recommended_working_set_size": 0}
    )
    assert _memory_caps.install_memory_caps() == (0, 0)


def test_install_memory_caps_swallows_set_limit_failure(monkeypatch):
    # The whole point of the guard: a Metal-less device (e.g. CI) returns (0, 0), never crashes.
    monkeypatch.setattr(
        _memory_caps.mx, "device_info", lambda: {"max_recommended_working_set_size": 25 * 1024**3}
    )

    def _boom(_limit):
        raise RuntimeError("metal unavailable")

    monkeypatch.setattr(_memory_caps.mx, "set_wired_limit", _boom)
    assert _memory_caps.install_memory_caps() == (0, 0)


def test_install_memory_caps_pushes_strict_byte_caps_on_healthy_device(monkeypatch):
    # The actual safety action, never exercised by the (0, 0) degraded-path tests above:
    # on a healthy device the GB caps are converted to BYTES and pushed STRICTLY BELOW the
    # device working-set max. A regression dropping the `* 1024**3` (installing 20 *bytes*
    # instead of 20 GB) silently disables the kernel-panic guard while still returning (20, 22).
    max_bytes = 25 * 1024**3
    monkeypatch.setattr(
        _memory_caps.mx, "device_info", lambda: {"max_recommended_working_set_size": max_bytes}
    )
    seen: dict[str, int] = {}
    monkeypatch.setattr(_memory_caps.mx, "set_wired_limit", lambda b: seen.__setitem__("wired", b))
    monkeypatch.setattr(
        _memory_caps.mx, "set_memory_limit", lambda b: seen.__setitem__("memory", b)
    )

    assert _memory_caps.install_memory_caps() == (20, 22)
    assert seen["wired"] == 20 * 1024**3  # bytes, not 20
    assert seen["memory"] == 22 * 1024**3
    assert seen["wired"] < max_bytes  # the wired cap is strictly below the device max (the guard)
    assert seen["memory"] < max_bytes


def test_install_memory_caps_sets_cache_limit(monkeypatch):
    """Reds if install_memory_caps never bounds MLX's retained cache pool (default: ~device size)."""
    monkeypatch.setattr(
        _memory_caps.mx, "device_info", lambda: {"max_recommended_working_set_size": 25 * 1024**3}
    )
    monkeypatch.setattr(_memory_caps.mx, "set_wired_limit", lambda b: None)
    monkeypatch.setattr(_memory_caps.mx, "set_memory_limit", lambda b: None)
    seen: list[int] = []
    monkeypatch.setattr(_memory_caps.mx, "set_cache_limit", seen.append)

    assert _memory_caps.install_memory_caps() == (20, 22)
    assert seen == [4 * 1024**3]


def _reporting_device(monkeypatch, *, wired_raises):
    monkeypatch.setattr(
        _memory_caps.mx, "device_info", lambda: {"max_recommended_working_set_size": 25 * 1024**3}
    )

    def _boom(_b):
        raise RuntimeError("metal unavailable")

    monkeypatch.setattr(
        _memory_caps.mx, "set_wired_limit", _boom if wired_raises else lambda b: None
    )
    monkeypatch.setattr(_memory_caps.mx, "set_memory_limit", lambda b: None)
    monkeypatch.setattr(_memory_caps.mx, "set_cache_limit", lambda b: None)


def test_caps_warning_when_install_failed_on_a_reporting_device(monkeypatch):
    _reporting_device(monkeypatch, wired_raises=True)
    assert _memory_caps.install_memory_caps() == (0, 0)
    warning = _memory_caps.caps_warning((0, 0))
    assert warning is not None
    assert "memory caps could not be installed on this device" in warning
    assert "not bounded by the wired cap" in warning


def test_caps_warning_none_after_a_successful_install(monkeypatch):
    _reporting_device(monkeypatch, wired_raises=False)
    assert _memory_caps.caps_warning(_memory_caps.install_memory_caps()) is None


def test_caps_warning_none_when_the_device_reports_no_working_set(monkeypatch):
    monkeypatch.setattr(_memory_caps.mx, "device_info", dict)
    assert _memory_caps.caps_warning(_memory_caps.install_memory_caps()) is None


def test_cache_limit_failure_does_not_report_the_caps_as_uninstalled(monkeypatch):
    """Bug: set_cache_limit raising lands in the same handler as the wired/memory caps, so the
    run reports 'caps not installed' although the panic-guard caps are in place."""
    monkeypatch.setattr(
        _memory_caps.mx, "device_info", lambda: {"max_recommended_working_set_size": 25 * 1024**3}
    )
    monkeypatch.setattr(_memory_caps.mx, "set_wired_limit", lambda b: None)
    monkeypatch.setattr(_memory_caps.mx, "set_memory_limit", lambda b: None)

    def _boom(_b):
        raise RuntimeError("cache limit unsupported")

    monkeypatch.setattr(_memory_caps.mx, "set_cache_limit", _boom)
    assert _memory_caps.install_memory_caps() == (20, 22)


def test_memory_limit_failure_keeps_the_installed_wired_cap_reported(monkeypatch):
    """Bug: set_memory_limit (advisory) raising after the wired cap succeeded reports (0, 0),
    so a run that IS bounded by the wired cap warns 'not bounded by the wired cap'."""
    monkeypatch.setattr(
        _memory_caps.mx, "device_info", lambda: {"max_recommended_working_set_size": 25 * 1024**3}
    )
    monkeypatch.setattr(_memory_caps.mx, "set_wired_limit", lambda b: None)
    monkeypatch.setattr(_memory_caps.mx, "set_cache_limit", lambda b: None)

    def _boom(_b):
        raise RuntimeError("memory limit unsupported")

    monkeypatch.setattr(_memory_caps.mx, "set_memory_limit", _boom)
    installed = _memory_caps.install_memory_caps()
    assert installed == (20, 0)
    assert _memory_caps.caps_warning(installed) is None
