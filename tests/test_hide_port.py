"""The pre-PR lane that mirrors CI: no turboquant_mlx import, no distribution metadata."""

import importlib.metadata

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


def _lane_only(request):
    if not request.config.getoption("--hide-port"):
        pytest.skip("meaningful only under --hide-port")


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
