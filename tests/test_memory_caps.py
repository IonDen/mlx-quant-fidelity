from mlx_quant_fidelity import _memory_caps


def test_clamp_uses_desired_on_large_device():
    # 25 GB recommended → desired (20, 22) fits unchanged
    assert _memory_caps._clamp_caps_gb(25) == (20, 22)


def test_clamp_shrinks_on_small_device():
    # 10 GB recommended → wired = min(20, 10-2) = 8; memory = min(22, max(9, 10)) = 10
    assert _memory_caps._clamp_caps_gb(10) == (8, 10)


def test_zero_recommended_is_noop_signal():
    assert _memory_caps._clamp_caps_gb(0) == (0, 0)
