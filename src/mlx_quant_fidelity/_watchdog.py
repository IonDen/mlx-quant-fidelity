"""Active-memory ceiling watchdog for heavy MLX workers.

MLX's wired cap prevents only the wired-exhaustion panic; pageable GPU over-allocation pages
instead of failing, and a sustained paging storm can take the machine down. The watchdog polls
``active + cache`` memory (dropped buffers sit in MLX's retained cache pool, which the active
figure does not count) and hard-exits with a clear message once it crosses a ceiling.

Started only by the two console entry points (the CLI and the weight worker), never by the
library API: ``os._exit`` inside a caller's notebook kernel would be hostile.
"""

import os
import sys
import threading
from collections.abc import Callable
from types import TracebackType
from typing import Self

import mlx.core as mx

EXIT_MEMORY_CEILING = 3
CEILING_MARGIN_BYTES = 4 * 1024**3
POLL_SECONDS = 0.05
_THREAD_NAME = "mqf-memory-watchdog"


def over_ceiling(active: int, cache: int, ceiling: int) -> bool:
    """True when resident MLX memory (active + cache) exceeds ``ceiling``; never for ``<= 0``."""
    return ceiling > 0 and active + cache > ceiling


def default_ceiling() -> int:
    """``memory_size`` minus a 4 GiB margin, or 0 when the device reports no size."""
    try:
        size = int(mx.device_info().get("memory_size", 0))
    except Exception:
        return 0
    return size - CEILING_MARGIN_BYTES if size > CEILING_MARGIN_BYTES else 0


def _default_on_breach(active: int, cache: int, ceiling: int) -> None:
    gib = 1024**3
    print(
        f"memory watchdog: active {active / gib:.2f} GiB + cache {cache / gib:.2f} GiB "
        f"exceeds the {ceiling / gib:.2f} GiB ceiling; aborting (exit {EXIT_MEMORY_CEILING})",
        file=sys.stderr,
    )
    sys.stderr.flush()
    os._exit(EXIT_MEMORY_CEILING)


class MemoryWatchdog:
    """Daemon polling thread that calls ``on_breach`` when the ceiling is crossed."""

    def __init__(
        self,
        *,
        ceiling: int | None = None,
        poll_seconds: float = POLL_SECONDS,
        on_breach: Callable[[int, int, int], None] | None = None,
    ) -> None:
        self._ceiling = default_ceiling() if ceiling is None else ceiling
        self._poll = poll_seconds
        self._on_breach = _default_on_breach if on_breach is None else on_breach
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            active, cache = int(mx.get_active_memory()), int(mx.get_cache_memory())
            if over_ceiling(active, cache, self._ceiling):
                self._on_breach(active, cache, self._ceiling)
                return
            self._stop.wait(self._poll)

    def start(self) -> Self:
        if self._ceiling > 0 and self._thread is None:
            self._thread = threading.Thread(target=self._run, name=_THREAD_NAME, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
