"""Pytest gates and MLX memory-safety guard."""

from __future__ import annotations

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
