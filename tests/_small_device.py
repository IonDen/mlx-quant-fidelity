"""Emulate the GitHub macOS runner's small device so the suite catches size-dependent gates."""

from collections.abc import Callable, Mapping

import mlx.core as mx

# The hosted runner reports a 4.7 GiB recommended working set on 7 GiB of unified memory.
SMALL_MAX_RECOMMENDED_BYTES = int(4.7 * 1024**3)
SMALL_MEMORY_SIZE = 7 * 1024**3


def small_device_info(real: Callable[[], Mapping[str, object]]) -> dict[str, object]:
    """`real()` with the two size fields replaced; every other field passes through."""
    info = dict(real())
    info["max_recommended_working_set_size"] = SMALL_MAX_RECOMMENDED_BYTES
    info["memory_size"] = SMALL_MEMORY_SIZE
    return info


def apply_small_device(*, enabled: bool) -> bool:
    """Patch `mlx.core.device_info` for the rest of the process when `enabled`."""
    if not enabled:
        return False
    real = mx.device_info

    def _small() -> dict[str, object]:
        return small_device_info(real)

    mx.device_info = _small  # type: ignore[assignment]
    return True
