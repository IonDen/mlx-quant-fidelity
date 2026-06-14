from mlx_quant_fidelity import _memory_caps


def test_clamp_uses_desired_on_large_device():
    # 25 GB recommended → desired (20, 22) fits unchanged
    assert _memory_caps._clamp_caps_gb(25) == (20, 22)


def test_clamp_shrinks_on_small_device():
    # 10 GB recommended → wired = min(20, 10-2) = 8; memory = min(22, max(9, 10)) = 10
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
