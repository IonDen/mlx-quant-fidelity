"""The `--emulate-small-device` lane: the suite must pass on a 4.7 GiB-working-set runner."""

import mlx.core as mx
from tests._small_device import (
    SMALL_MAX_RECOMMENDED_BYTES,
    SMALL_MEMORY_SIZE,
    apply_small_device,
    small_device_info,
)


def test_small_device_info_overrides_only_the_two_sizes():
    """Reds if either size is left at the host's value (the gates would see the 32 GiB Mac) or
    an unrelated field such as the device name is dropped."""
    real = {
        "device_name": "Apple M1 Max",
        "memory_size": 34359738368,
        "max_recommended_working_set_size": 26800603136,
    }
    info = small_device_info(lambda: real)
    assert info["max_recommended_working_set_size"] == 5046586572  # int(4.7 * 1024**3)
    assert info["memory_size"] == 7516192768  # 7 GiB
    assert info["device_name"] == "Apple M1 Max"
    assert real["memory_size"] == 34359738368  # the input mapping is not mutated


def test_constants_match_the_github_runner_figures():
    assert SMALL_MAX_RECOMMENDED_BYTES == 5046586572
    assert SMALL_MEMORY_SIZE == 7 * 1024**3


def test_apply_small_device_off_is_a_noop(monkeypatch):
    """Reds if the disabled flag still patches device_info."""
    sentinel = object()
    monkeypatch.setattr(mx, "device_info", sentinel)
    assert apply_small_device(enabled=False) is False
    assert mx.device_info is sentinel


def test_apply_small_device_on_patches_device_info(monkeypatch):
    """Reds if enabling leaves the real device_info in place."""
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "x"})
    assert apply_small_device(enabled=True) is True
    info = mx.device_info()
    assert info["max_recommended_working_set_size"] == 5046586572
    assert info["memory_size"] == 7 * 1024**3
