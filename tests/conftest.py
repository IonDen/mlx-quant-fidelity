"""Pytest gates and MLX memory-safety guard."""

from __future__ import annotations

import atexit
import os
import sys

import pytest

from mlx_quant_fidelity._memory_caps import install_memory_caps

GATED_MARKERS: tuple[tuple[str, str, str], ...] = (
    ("slow", "--run-slow", "real-model probe integration"),
    ("network", "--run-network", "real network I/O"),
)

# Install caps at import, before collection imports any MLX-heavy worker module.
INSTALLED_CAPS_GB = install_memory_caps()


def _markers_to_skip(enabled_flags: set[str]) -> list[tuple[str, str]]:
    return [
        (marker, f"requires {flag} ({description})")
        for marker, flag, description in GATED_MARKERS
        if flag not in enabled_flags
    ]


def pytest_addoption(parser: pytest.Parser) -> None:
    for marker, flag, description in GATED_MARKERS:
        parser.addoption(
            flag,
            action="store_true",
            default=False,
            help=f"run `{marker}` tests ({description}); skipped by default",
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    enabled = {flag for _marker, flag, _description in GATED_MARKERS if config.getoption(flag)}
    for marker, reason in _markers_to_skip(enabled):
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


_FINAL_EXIT_CODE = 0


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    # Record pytest's final exit code (already reflects test failures and the
    # --cov-fail-under gate) for the atexit hard-exit below.
    global _FINAL_EXIT_CODE
    _FINAL_EXIT_CODE = int(session.exitstatus)


@atexit.register
def _hard_exit_past_metal_teardown() -> None:  # pragma: no cover - runs at interpreter shutdown
    """Skip MLX's Metal backend C++ destructor, which segfaults at interpreter shutdown
    on Apple Silicon ("Python quit unexpectedly").

    This runs AFTER pytest has printed its summary + coverage report and recorded the
    final exit code, so output is preserved and the coverage gate's exit code is honored
    (a real failure still exits non-zero). We just avoid the buggy teardown. CI, which
    does not crash on teardown, gets the same clean exit.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_FINAL_EXIT_CODE)
