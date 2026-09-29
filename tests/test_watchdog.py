import threading

import pytest

from mlx_quant_fidelity import _watchdog
from mlx_quant_fidelity._watchdog import MemoryWatchdog, over_ceiling

GIB = 1024**3


def test_over_ceiling_counts_cache_memory():
    """Reds if the check looks at active memory only (cache pool is resident footprint too)."""
    assert over_ceiling(10, 10, 15) is True
    assert over_ceiling(10, 4, 15) is False
    assert over_ceiling(10, 5, 15) is False  # exactly at the ceiling is not a breach
    assert over_ceiling(10**12, 0, 0) is False  # unknown ceiling never fires


def test_default_ceiling_is_memory_size_minus_four_gib(monkeypatch):
    monkeypatch.setattr(_watchdog.mx, "device_info", lambda: {"memory_size": 32 * GIB})
    assert _watchdog.default_ceiling() == 28 * GIB


def test_default_ceiling_zero_when_unknown(monkeypatch):
    monkeypatch.setattr(_watchdog.mx, "device_info", dict)
    assert _watchdog.default_ceiling() == 0

    def _boom():
        raise RuntimeError("no device")

    monkeypatch.setattr(_watchdog.mx, "device_info", _boom)
    assert _watchdog.default_ceiling() == 0


def test_watchdog_breach_calls_handler(monkeypatch):
    monkeypatch.setattr(_watchdog.mx, "get_active_memory", lambda: 10)
    monkeypatch.setattr(_watchdog.mx, "get_cache_memory", lambda: 10)
    fired = threading.Event()
    seen: list[tuple[int, int, int]] = []

    def _handler(active: int, cache: int, ceiling: int) -> None:
        seen.append((active, cache, ceiling))
        fired.set()

    dog = MemoryWatchdog(ceiling=15, poll_seconds=0.001, on_breach=_handler)
    try:
        dog.start()
        assert fired.wait(timeout=1.0)
    finally:
        dog.stop()
    assert seen[0] == (10, 10, 15)


def test_watchdog_quiet_below_ceiling(monkeypatch):
    monkeypatch.setattr(_watchdog.mx, "get_active_memory", lambda: 10)
    monkeypatch.setattr(_watchdog.mx, "get_cache_memory", lambda: 4)
    fired = threading.Event()
    dog = MemoryWatchdog(ceiling=15, poll_seconds=0.001, on_breach=lambda *_: fired.set())
    try:
        dog.start()
        assert not fired.wait(timeout=0.1)
    finally:
        dog.stop()


def test_watchdog_noop_without_ceiling():
    dog = MemoryWatchdog(ceiling=0, poll_seconds=0.001, on_breach=lambda *_: None)
    try:
        dog.start()
        assert not [t for t in threading.enumerate() if t.name == "mqf-memory-watchdog"]
    finally:
        dog.stop()


def test_watchdog_stop_ends_thread(monkeypatch):
    monkeypatch.setattr(_watchdog.mx, "get_active_memory", lambda: 0)
    monkeypatch.setattr(_watchdog.mx, "get_cache_memory", lambda: 0)
    with MemoryWatchdog(ceiling=15, poll_seconds=0.001, on_breach=lambda *_: None):
        assert [t for t in threading.enumerate() if t.name == "mqf-memory-watchdog"]
    assert not [t for t in threading.enumerate() if t.name == "mqf-memory-watchdog"]


def test_default_breach_writes_message_and_exits_3(monkeypatch, capsys):
    exits: list[int] = []
    monkeypatch.setattr(_watchdog.os, "_exit", exits.append)
    _watchdog._default_on_breach(10 * GIB, 5 * GIB, 12 * GIB)
    assert exits == [3]
    assert "memory watchdog" in capsys.readouterr().err


def test_exit_code_constant():
    assert _watchdog.EXIT_MEMORY_CEILING == 3


def test_measure_kv_fidelity_does_not_start_watchdog(monkeypatch):
    """Reds if main()/the library API starts the watchdog (os._exit in a notebook kernel)."""
    from mlx_quant_fidelity import cli

    def _boom(self):
        raise AssertionError("watchdog started outside a console entry")

    monkeypatch.setattr(MemoryWatchdog, "start", _boom)
    monkeypatch.setattr(cli, "install_memory_caps", lambda: (0, 0))
    monkeypatch.setattr(
        cli, "measure_kv_fidelity", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    )
    # error path is enough: main() must not touch the watchdog on any path
    with pytest.raises(RuntimeError):
        cli.main(["kv", "m"])
