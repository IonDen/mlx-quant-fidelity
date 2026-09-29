"""Hardware-aware MLX memory caps (kernel-watchdog panic guard).

Derives wired + memory caps from the device's reported working-set size and
clamps strictly below it. Returns (0, 0) as a no-op signal on devices/CI images
that report no working-set size. Mirrors the proven mlx-taef pattern.
"""

import contextlib

import mlx.core as mx

DESIRED_WIRED_GB = 20
DESIRED_MEMORY_GB = 22
HEADROOM_GB = 2
MIN_WORKING_SET_GB = 3  # smallest working set that admits wired < memory <= working set
CACHE_LIMIT_GB = 4  # bounds MLX's retained buffer pool (counted by the watchdog)


def _clamp_caps_gb(max_recommended_gb: int) -> tuple[int, int]:
    """Clamp the desired caps to fit a device with `max_recommended_gb` working set."""
    if max_recommended_gb < MIN_WORKING_SET_GB:
        return (0, 0)  # no cap strictly below the working set fits; caps_warning reports it
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


def device_string() -> str | None:
    """Human-readable chip + unified-memory size (e.g. 'Apple M1 Max, 32 GB'), or None.

    Report provenance only — never used for gating. Returns None when MLX does not
    report a device name (CI containers, future backends).
    """
    try:
        info = mx.device_info()
        name = info.get("device_name")
        if not isinstance(name, str) or not name:
            return None
        mem = info.get("memory_size")
        if isinstance(mem, int) and mem > 0:
            return f"{name}, {round(mem / 1024**3)} GB"
        return name
    except Exception:
        return None


def install_memory_caps() -> tuple[int, int]:
    """Apply wired + memory caps for the current device. Idempotent; never raises.

    Returns the (wired_gb, memory_gb) actually installed (memory_gb is 0 when only the
    advisory memory limit failed), or (0, 0) on a device with no reported working-set size
    or where the wired cap could not be applied.
    """
    wired_gb, memory_gb = compute_safe_caps_gb()
    if wired_gb == 0:
        return (0, 0)
    try:
        mx.set_wired_limit(wired_gb * 1024**3)
    except Exception:
        return (0, 0)
    # The wired cap is the panic guard; the memory limit is advisory, so its failure must not
    # make a wired-bounded run report itself as unbounded.
    try:
        mx.set_memory_limit(memory_gb * 1024**3)
    except Exception:
        memory_gb = 0
    # Bounds the retained pool only; the panic-guard caps above are already in place.
    with contextlib.suppress(Exception):
        mx.set_cache_limit(CACHE_LIMIT_GB * 1024**3)
    return (wired_gb, memory_gb)


def caps_warning(installed: tuple[int, int]) -> str | None:
    """A report warning when caps are missing on a device that does have a working-set limit.

    ``installed`` is what :func:`install_memory_caps` returned. None when the caps were
    installed, or when the device reports no working-set size (the intended no-op on CI images).
    """
    if installed != (0, 0):
        return None
    try:
        reported = int(mx.device_info().get("max_recommended_working_set_size", 0) or 0)
    except Exception:
        return None
    if reported <= 0:
        return None
    return (
        "memory caps could not be installed on this device; the run was not bounded by the "
        "wired cap."
    )


__all__ = ["caps_warning", "compute_safe_caps_gb", "device_string", "install_memory_caps"]
