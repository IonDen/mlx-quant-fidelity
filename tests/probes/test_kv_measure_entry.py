"""measure_kv_fidelity's model-free front door: refusals that must precede any load."""

import mlx_lm
import pytest
from tests.factories import make_corpus, make_fid_report

from mlx_quant_fidelity import _memory_caps
from mlx_quant_fidelity.errors import CompareConfigError
from mlx_quant_fidelity.probes import kv as kv_mod
from mlx_quant_fidelity.probes._preload import PreloadResult
from mlx_quant_fidelity.probes.kv_methods import parse_method_spec


def _no_load(*_a, **_k):
    raise AssertionError("the model must not be loaded")


def test_control_on_method_without_control_lane_refuses_before_load(monkeypatch):
    # bug caught: the --control refusal firing inside score_kv_config, i.e. after the download
    monkeypatch.setattr(mlx_lm, "load", _no_load)
    monkeypatch.setattr(kv_mod, "preload_check", _no_load)
    with pytest.raises(CompareConfigError, match="--control"):
        kv_mod.measure_kv_fidelity(
            "org/m", method=parse_method_spec("affine:8:4"), control=True, corpus=make_corpus()
        )


def test_control_refusal_names_a_fake_method_without_control_method(monkeypatch):
    # bug caught: the pre-load check keyed on a method name instead of the capability
    monkeypatch.setattr(mlx_lm, "load", _no_load)
    monkeypatch.setattr(kv_mod, "preload_check", _no_load)

    class Bare:
        name = "bare"
        label = "bare"

        def validate_before_load(self):
            pass

    monkeypatch.setattr(kv_mod, "_resolve_method", lambda m, b, g: m)
    with pytest.raises(CompareConfigError, match="bare"):
        kv_mod.measure_kv_fidelity("org/m", method=Bare(), control=True, corpus=make_corpus())


def _stub_pipeline(monkeypatch, *, report):
    monkeypatch.setattr(mlx_lm, "load", lambda *a, **k: (object(), object()))
    monkeypatch.setattr(kv_mod, "preload_check", lambda *a, **k: PreloadResult({}, None))
    monkeypatch.setattr(kv_mod, "score_kv_config", lambda *a, **k: report)


def _device_reports_working_set(monkeypatch, *, set_wired_raises):
    monkeypatch.setattr(
        _memory_caps.mx, "device_info", lambda: {"max_recommended_working_set_size": 25 * 1024**3}
    )

    def _boom(_b):
        raise RuntimeError("metal unavailable")

    monkeypatch.setattr(
        _memory_caps.mx, "set_wired_limit", _boom if set_wired_raises else lambda b: None
    )
    monkeypatch.setattr(_memory_caps.mx, "set_memory_limit", lambda b: None)
    monkeypatch.setattr(_memory_caps.mx, "set_cache_limit", lambda b: None)
    monkeypatch.setattr(kv_mod, "install_memory_caps", _memory_caps.install_memory_caps)


def test_uninstalled_caps_on_a_reporting_device_add_a_report_warning(monkeypatch):
    # bug caught: caps silently not installed, so the run is unbounded yet the report is silent
    _stub_pipeline(monkeypatch, report=make_fid_report())
    _device_reports_working_set(monkeypatch, set_wired_raises=True)
    report = kv_mod.measure_kv_fidelity("org/m", corpus=make_corpus())
    assert any("memory caps could not be installed" in w for w in report.warnings)


def test_installed_caps_add_no_warning(monkeypatch):
    # bug caught: the warning firing on every run
    _stub_pipeline(monkeypatch, report=make_fid_report())
    _device_reports_working_set(monkeypatch, set_wired_raises=False)
    report = kv_mod.measure_kv_fidelity("org/m", corpus=make_corpus())
    assert not any("memory caps" in w for w in report.warnings)


def test_no_working_set_reported_is_not_a_warning(monkeypatch):
    # bug caught: warning on CI images that report no working set (the (0, 0) no-op is intended)
    _stub_pipeline(monkeypatch, report=make_fid_report())
    monkeypatch.setattr(_memory_caps.mx, "device_info", dict)
    monkeypatch.setattr(kv_mod, "install_memory_caps", _memory_caps.install_memory_caps)
    report = kv_mod.measure_kv_fidelity("org/m", corpus=make_corpus())
    assert not any("memory caps" in w for w in report.warnings)


def test_measure_kv_fidelity_never_starts_the_watchdog(monkeypatch):
    """Bug: the library entry point arms the watchdog, so os._exit could fire inside a caller's
    notebook kernel. Drives the REAL measure_kv_fidelity with the pipeline stubbed."""
    from mlx_quant_fidelity._watchdog import MemoryWatchdog

    def _boom(self):
        raise AssertionError("watchdog started by the library API")

    monkeypatch.setattr(MemoryWatchdog, "start", _boom)
    _stub_pipeline(monkeypatch, report=make_fid_report())
    monkeypatch.setattr(kv_mod, "install_memory_caps", lambda: (0, 0))
    report = kv_mod.measure_kv_fidelity("org/m", corpus=make_corpus())
    assert report is not None


def test_measure_kv_fidelity_loads_the_model_lazily(monkeypatch):
    # bug caught: an eager load materialises every weight before the "weights alone exceed"
    # gate can refuse, so the refusal fires after the memory it was meant to protect is used
    seen: dict[str, object] = {}

    def _spy(*a, **k):
        seen.update(k)
        return (object(), object())

    _stub_pipeline(monkeypatch, report=make_fid_report())
    monkeypatch.setattr(mlx_lm, "load", _spy)
    kv_mod.measure_kv_fidelity("org/m", corpus=make_corpus())
    assert seen.get("lazy") is True


def test_compare_kv_load_model_is_lazy(monkeypatch):
    # bug caught: the KV compare path loads eagerly, defeating the resident-size gate
    from mlx_quant_fidelity.runners import compare as cmp

    seen: dict[str, object] = {}

    def _spy(*a, **k):
        seen.update(k)
        return (object(), object())

    monkeypatch.setattr(mlx_lm, "load", _spy)
    cmp._load_model("org/m", "rev")
    assert seen.get("lazy") is True
