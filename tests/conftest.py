"""Pytest gates and MLX memory-safety guard."""

from __future__ import annotations

import atexit
import os
import sys
from typing import TYPE_CHECKING

import pytest
from hypothesis import settings as hypothesis_settings
from tests._hide_port import apply_hide_port
from tests._small_device import apply_small_device

from mlx_quant_fidelity._memory_caps import install_memory_caps

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

GATED_MARKERS: tuple[tuple[str, str, str], ...] = (
    ("slow", "--run-slow", "real-model probe integration"),
    ("network", "--run-network", "real network I/O"),
)

# Install caps at import, before collection imports any MLX-heavy worker module.
INSTALLED_CAPS_GB = install_memory_caps()

_FINAL_EXIT_CODE = 0

# Property tests draw the same examples on every run, so a red property test is reproducible
# and the default lane cannot flake on a rare draw. Exploratory runs can still opt out with
# `--hypothesis-profile=default`.
hypothesis_settings.register_profile("derandomized", derandomize=True)
hypothesis_settings.load_profile("derandomized")


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
    parser.addoption(
        "--hide-port",
        action="store_true",
        default=False,
        help="run the default suite with turboquant_mlx masked (import + distribution metadata), "
        "mirroring CI; default-lane only, not combinable with --run-slow",
    )
    parser.addoption(
        "--emulate-small-device",
        action="store_true",
        default=False,
        help="report a 4.7 GiB working set / 7 GiB memory from mx.device_info() for the whole "
        "session, mirroring the hosted CI runner; the memory caps are re-installed against it",
    )


def pytest_configure(config: pytest.Config) -> None:
    def _set_exit_code(code: int) -> None:
        global _FINAL_EXIT_CODE
        _FINAL_EXIT_CODE = code

    # The session caps above were installed at import (the option is unreadable before
    # configure), against the real device. Emulation then re-installs them against the small
    # device; lowering a cap is always safe, and no MLX allocation happens in between.
    if apply_small_device(enabled=bool(config.getoption("--emulate-small-device"))):
        install_memory_caps()

    apply_hide_port(
        hide_port=bool(config.getoption("--hide-port")),
        run_slow=bool(config.getoption("--run-slow")),
        modules=sys.modules,
        set_exit_code=_set_exit_code,
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    enabled = {flag for _marker, flag, _description in GATED_MARKERS if config.getoption(flag)}
    for marker, reason in _markers_to_skip(enabled):
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


class _RealHubCall(BaseException):
    """A test reached a real Hub download.

    A BaseException, not an Exception: the code under test reads an unreadable Hub file as
    "defer to the load" with ``except Exception``, which would swallow an AssertionError.
    """


# Every name a real download goes through, including the copies other modules bound at import
# time. Each module is imported before patching, so the cover does not depend on import order.
_HUB_ENTRY_POINTS: tuple[tuple[str, str], ...] = (
    ("huggingface_hub", "hf_hub_download"),
    ("huggingface_hub", "snapshot_download"),
    ("huggingface_hub.file_download", "hf_hub_download"),
    ("huggingface_hub._snapshot_download", "hf_hub_download"),
    ("huggingface_hub._snapshot_download", "snapshot_download"),
    ("huggingface_hub.repocard", "hf_hub_download"),
    ("mlx_lm.utils", "snapshot_download"),
)
# Names another transformers release may not bind; a missing one is skipped, not an error.
_OPTIONAL_HUB_ENTRY_POINTS: tuple[tuple[str, str], ...] = (
    ("transformers.utils.hub", "hf_hub_download"),
    ("transformers.utils.hub", "snapshot_download"),
)
# Hub metadata calls, patched on the class so every HfApi instance refuses them.
_HUB_API_METHODS: tuple[str, ...] = ("model_info",)


def _hub_guard_exempt(marker_names: Iterable[str]) -> bool:
    """True for a test in an opt-in lane that may reach the Hub (the gated markers)."""
    return not {marker for marker, _flag, _description in GATED_MARKERS}.isdisjoint(marker_names)


@pytest.fixture(autouse=True)
def _no_real_hub(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[list[str]]:
    """Refuse every real Hub call outside the network and slow lanes.

    Covered: ``hf_hub_download`` and ``snapshot_download`` as huggingface_hub exports them and
    as ``file_download``, ``_snapshot_download``, ``repocard``, ``mlx_lm.utils`` and (when that
    release binds them) ``transformers.utils.hub`` hold them, plus ``HfApi.model_info``. Other
    HfApi methods and direct HTTP calls are not covered.

    A refused call raises _RealHubCall and is recorded; the teardown fails the test on any
    recorded call, so one swallowed by ``except BaseException`` or lost in a thread still
    shows. A test's own fake installed with monkeypatch replaces the guard for that test.
    """
    calls: list[str] = []
    if _hub_guard_exempt(marker.name for marker in request.node.iter_markers()):
        yield calls
        return
    import huggingface_hub
    import huggingface_hub._snapshot_download
    import huggingface_hub.file_download
    import huggingface_hub.repocard
    import mlx_lm.utils  # noqa: F401
    import transformers.utils.hub  # noqa: F401

    def refuse(name: str) -> Callable[..., object]:
        def _refuse(*args: object, **kwargs: object) -> object:
            call = f"{name}{args!r}"
            calls.append(call)
            raise _RealHubCall(f"real Hub call in a test: {call}; stub it or mark it `network`")

        return _refuse

    for module_name, attr in _HUB_ENTRY_POINTS:
        monkeypatch.setattr(sys.modules[module_name], attr, refuse(f"{module_name}.{attr}"))
    for module_name, attr in _OPTIONAL_HUB_ENTRY_POINTS:
        if hasattr(sys.modules[module_name], attr):
            monkeypatch.setattr(sys.modules[module_name], attr, refuse(f"{module_name}.{attr}"))
    for method in _HUB_API_METHODS:
        monkeypatch.setattr(
            huggingface_hub.HfApi, method, refuse(f"huggingface_hub.HfApi.{method}")
        )
    yield calls
    if calls:
        pytest.fail("real Hub call in a test: " + "; ".join(calls), pytrace=False)


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
