"""Hardware-aware MLX memory caps (kernel-watchdog panic guard).

Derives wired + memory caps from the device's reported working-set size and
clamps strictly below it. Returns (0, 0) as a no-op signal on devices/CI images
that report no working-set size. Mirrors the proven mlx-taef pattern.
"""

import mlx.core as mx

DESIRED_WIRED_GB = 20
DESIRED_MEMORY_GB = 22
HEADROOM_GB = 2


def _clamp_caps_gb(max_recommended_gb: int) -> tuple[int, int]:
    """Clamp the desired caps to fit a device with `max_recommended_gb` working set."""
    if max_recommended_gb <= 0:
        return (0, 0)
    wired_gb = min(DESIRED_WIRED_GB, max(1, max_recommended_gb - HEADROOM_GB))
    memory_gb = min(DESIRED_MEMORY_GB, max(wired_gb + 1, max_recommended_gb))
    return (wired_gb, memory_gb)


def compute_safe_caps_gb() -> tuple[int, int]:
    """Return (wired_gb, memory_gb) that fit the current device, or (0, 0)."""
    try:
        info = mx.device_info()
        max_gb = int(info.get("max_recommended_working_set_size", 0)) // (1024**3)
    except Exception:
        return (0, 0)
    return _clamp_caps_gb(max_gb)


def install_memory_caps() -> tuple[int, int]:
    """Apply wired + memory caps for the current device. Idempotent; never raises.

    Returns the (wired_gb, memory_gb) actually installed, or (0, 0) on a device
    with no reported working-set size or where caps could not be applied.
    """
    wired_gb, memory_gb = compute_safe_caps_gb()
    if wired_gb == 0:
        return (0, 0)
    try:
        mx.set_wired_limit(wired_gb * 1024**3)
        mx.set_memory_limit(memory_gb * 1024**3)
    except Exception:
        return (0, 0)
    return (wired_gb, memory_gb)


__all__ = ["compute_safe_caps_gb", "install_memory_caps"]
