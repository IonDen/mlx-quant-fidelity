"""The pre-PR lane that mirrors CI: no turboquant_mlx import, no distribution metadata."""

import importlib
import importlib.metadata
import importlib.util
import sys

import conftest  # pytest prepend mode adds tests/ to sys.path; this is the loaded module
import pytest
from tests._hide_port import (
    HIDDEN_DISTRIBUTIONS,
    apply_hide_port,
    flag_conflict,
    hiding_distribution,
    mask_port,
)


def test_mask_port_sets_none_and_evicts_only_the_port_tree():
    """Reds if submodules survive (a cached `turboquant_mlx.cache` would import fine — CPython
    returns a cached submodule without consulting the parent) or an unrelated key sharing the
    prefix is evicted. The `None` sentinel itself is CPython's documented ModuleNotFoundError
    trigger."""
    modules = {
        "turboquant_mlx": object(),
        "turboquant_mlx.cache": object(),
        "turboquant_mlxy": 1,
        "mlx": 2,
    }
    mask_port(modules)
    assert modules["turboquant_mlx"] is None
    assert "turboquant_mlx.cache" not in modules
    assert modules["turboquant_mlxy"] == 1
    assert modules["mlx"] == 2


def test_what_mask_port_parks_makes_the_real_import_raise(monkeypatch):
    """Pins the whole chain, not just CPython's contract: whatever `mask_port` parks in
    sys.modules must halt a real `import turboquant_mlx`. A future `mask_port` that parked a
    stub module instead would leave every other test green while the import quietly succeeded."""
    modules: dict[str, object] = {"turboquant_mlx": object()}
    mask_port(modules)
    monkeypatch.setitem(sys.modules, "turboquant_mlx", modules["turboquant_mlx"])
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("turboquant_mlx")


def test_hiding_distribution_hides_both_spellings_and_passes_others_through():
    """Reds if only one spelling is hidden or the wrapper swallows other packages."""
    calls = []

    def real(name):
        calls.append(name)
        return f"dist:{name}"

    wrapped = hiding_distribution(real, HIDDEN_DISTRIBUTIONS)
    for hidden in ("turboquant-mlx", "turboquant_mlx", "Turboquant-MLX"):
        with pytest.raises(importlib.metadata.PackageNotFoundError):
            wrapped(hidden)
    assert wrapped("mlx-lm") == "dist:mlx-lm"
    assert calls == ["mlx-lm"]


def test_hiding_distribution_accepts_the_stdlib_keyword():
    """Reds if the wrapper renames importlib.metadata.distribution's `distribution_name`
    parameter: any caller using the documented keyword form would TypeError under the lane."""
    wrapped = hiding_distribution(lambda name: f"dist:{name}", HIDDEN_DISTRIBUTIONS)
    assert wrapped(distribution_name="mlx-lm") == "dist:mlx-lm"
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        wrapped(distribution_name="turboquant-mlx")


def test_flag_conflict_names_run_slow():
    """Reds if the lane silently accepts --run-slow (the slow lane needs the real port)."""
    assert "--run-slow" in (flag_conflict(hide_port=True, run_slow=True) or "")
    assert flag_conflict(hide_port=True, run_slow=False) is None
    assert flag_conflict(hide_port=False, run_slow=True) is None


def test_apply_hide_port_records_usage_error_exit_code_before_raising():
    """Reds if the exit code is set after the raise (or not at all): the atexit hard-exit
    would then report 0 for a usage error."""
    events: list[object] = []
    with pytest.raises(pytest.UsageError):
        apply_hide_port(
            hide_port=True, run_slow=True, modules={}, set_exit_code=lambda c: events.append(c)
        )
    assert events == [int(pytest.ExitCode.USAGE_ERROR)]


def test_apply_hide_port_masks_the_module_tree_and_the_distribution_lookup(monkeypatch):
    """Reds if the enabled path drops the mask or the metadata wrap — a mutant deleting
    `mask_port(modules)` (or the `distribution` reassignment) survives every other test here.
    The monkeypatch goes first so teardown restores `distribution` even though the helper
    reassigns the attribute itself."""
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: f"dist:{name}")
    modules: dict[str, object] = {
        "turboquant_mlx": object(),
        "turboquant_mlx.cache": object(),
        "mlx": 1,
    }
    events: list[object] = []
    assert (
        apply_hide_port(
            hide_port=True, run_slow=False, modules=modules, set_exit_code=events.append
        )
        is True
    )
    assert modules["turboquant_mlx"] is None
    assert "turboquant_mlx.cache" not in modules
    assert modules["mlx"] == 1
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        importlib.metadata.distribution("turboquant-mlx")
    assert importlib.metadata.distribution("mlx-lm") == "dist:mlx-lm"
    assert events == []


def test_apply_hide_port_is_a_no_op_without_the_flag():
    """Reds if the mask fires (or the exit code is touched) when --hide-port is absent."""
    modules: dict[str, object] = {"turboquant_mlx": object()}
    events: list[object] = []
    assert (
        apply_hide_port(
            hide_port=False, run_slow=True, modules=modules, set_exit_code=events.append
        )
        is False
    )
    assert modules["turboquant_mlx"] is not None
    assert events == []


def test_pytest_configure_records_the_usage_error_in_the_module_global(monkeypatch):
    """Reds if the closure stops writing the module global — the atexit hard-exit would then
    report 0 for a usage error. The helper-level test above only proves the callback is called;
    this proves conftest's callback is the one that moves `_FINAL_EXIT_CODE`."""
    monkeypatch.setattr(conftest, "_FINAL_EXIT_CODE", 0)

    class _StubConfig:
        def getoption(self, name):
            return name in ("--hide-port", "--run-slow")

    # The conflict branch raises before masking, so sys.modules is never touched here.
    with pytest.raises(pytest.UsageError):
        conftest.pytest_configure(_StubConfig())
    recorded = conftest._FINAL_EXIT_CODE
    assert recorded == int(pytest.ExitCode.USAGE_ERROR)


def _lane_only(request):
    """Skip only when the port is installed AND the mask is off — the canaries are meaningful
    wherever `turboquant_mlx` is genuinely absent, which is how CI runs."""
    if (
        not request.config.getoption("--hide-port")
        and importlib.util.find_spec("turboquant_mlx") is not None
    ):
        pytest.skip("port installed and --hide-port not set")


def test_lane_reproduces_the_0_7_0_offender(request):
    """The exact shape of the v0.7.0 CI red (commit 9958da2): a provenance() call that reaches
    the real port passed locally because the pin is installed. Under the lane it must fail."""
    _lane_only(request)
    from mlx_quant_fidelity.errors import MethodUnavailableError
    from mlx_quant_fidelity.probes.kv_methods import TurboQuantVOnlyKVMethod

    with pytest.raises(MethodUnavailableError, match="not installed"):
        TurboQuantVOnlyKVMethod(v_bits=3, seed=41).provenance()


def test_lane_hides_the_distribution_metadata_too(request, monkeypatch):
    """Reds if only the import is masked: with a fake port installed, the provenance would
    still read the locally-installed distribution version instead of the "unknown" fallback."""
    _lane_only(request)
    from tests.probes.fake_turboquant import install_fake_port

    from mlx_quant_fidelity.probes.kv_methods import TurboQuantKVMethod

    install_fake_port(monkeypatch)
    assert TurboQuantKVMethod(bits=4).provenance()["dist_version"] == "unknown"
