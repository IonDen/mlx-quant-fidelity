"""Pytest gates and MLX memory-safety guard."""

from __future__ import annotations

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


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    """Hard-exit past MLX's Metal teardown segfault.

    Once GPU compute has run, MLX's Metal backend segfaults in its C++ destructor at
    interpreter shutdown on Apple Silicon ("Python quit unexpectedly"). As a wrapper hook
    we ``yield`` first, so every other sessionfinish hook — coverage data, the terminal
    summary, and ``--cov-fail-under`` gating — completes and ``session.exitstatus`` is
    final. Then we flush and ``os._exit`` with that exact status, skipping the buggy
    teardown. CI (no Metal device) is unaffected: it just gets a clean exit with the
    same code.
    """
    try:
        return (yield)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(int(session.exitstatus))
