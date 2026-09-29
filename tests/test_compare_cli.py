import json
from unittest.mock import patch

import pytest

from mlx_quant_fidelity import cli
from mlx_quant_fidelity.cli import main
from mlx_quant_fidelity.probes.kv_methods import StockKVMethod
from mlx_quant_fidelity.report import ComparisonReport


def _fake_comparison(mode):
    return ComparisonReport(
        mode=mode,
        reference="ref" if mode == "weight" else None,
        model=None if mode == "weight" else "m",
        corpus=None,
        quantize_start=None,
        quantize_mode=None,
        budget=None,
        results=(),
        frontier=(),
        dominated=(),
        budget_pick=None,
        mlx_version="0.21",
        mlx_lm_version="0.31.3",
    )


def test_compare_weights_dispatches(monkeypatch, capsys):
    captured = {}

    def fake(quant_ids, reference, **kw):
        captured["args"] = (quant_ids, reference, kw)
        return _fake_comparison("weight")

    monkeypatch.setattr(cli, "compare_weight_fidelity", fake)
    rc = cli.main(
        ["compare", "weights", "q4", "q6", "q8", "--reference", "ref", "--format", "json"]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "weight"
    assert captured["args"][0] == ["q4", "q6", "q8"]
    assert captured["args"][1] == "ref"


def test_compare_kv_parses_configs(monkeypatch, capsys):
    captured = {}

    def fake(model, configs, **kw):
        captured["args"] = (model, configs, kw)
        return _fake_comparison("kv")

    monkeypatch.setattr(cli, "compare_kv_fidelity", fake)
    rc = cli.main(["compare", "kv", "m", "--configs", "4:32,4:64,8:64", "--min-tier", "good"])
    assert rc == 0
    assert captured["args"][1] == [
        StockKVMethod(bits=4, group_size=32),
        StockKVMethod(bits=4, group_size=64),
        StockKVMethod(bits=8, group_size=64),
    ]
    assert captured["args"][2]["min_tier"] == "good"


def test_compare_kv_rejects_bad_config(monkeypatch, capsys):
    rc = cli.main(["compare", "kv", "m", "--configs", "oops"])
    assert rc == 2
    assert "configs" in capsys.readouterr().err.lower()


def test_compare_kv_rejects_non_digit_group_size(monkeypatch, capsys):
    rc = cli.main(["compare", "kv", "m", "--configs", "4:abc"])
    assert rc == 2
    assert "configs" in capsys.readouterr().err.lower()


def test_compare_kv_rejects_zero_group_size(monkeypatch, capsys):
    rc = cli.main(["compare", "kv", "m", "--configs", "4:0"])
    assert rc == 2
    assert "configs" in capsys.readouterr().err.lower()


def test_compare_weights_forwards_filter_kwargs(monkeypatch, capsys):
    captured = {}

    def fake(quant_ids, reference, **kw):
        captured["kw"] = kw
        return _fake_comparison("weight")

    monkeypatch.setattr(cli, "compare_weight_fidelity", fake)
    rc = cli.main(
        [
            "compare",
            "weights",
            "q4",
            "q6",
            "--reference",
            "ref",
            "--min-tier",
            "good",
            "--max-kld",
            "0.5",
        ]
    )
    assert rc == 0
    assert captured["kw"]["min_tier"] == "good"
    assert captured["kw"]["max_kld"] == 0.5


def test_compare_weights_quant_and_reference_revision_thread(monkeypatch, capsys):
    """RED if `compare weights` has no --quant-revision/--reference-revision flags, or they
    are not forwarded to compare_weight_fidelity.
    """
    captured = {}

    def fake(quant_ids, reference, **kw):
        captured["kw"] = kw
        return _fake_comparison("weight")

    monkeypatch.setattr(cli, "compare_weight_fidelity", fake)
    rc = cli.main(
        [
            "compare",
            "weights",
            "q4",
            "q6",
            "--reference",
            "ref",
            "--quant-revision",
            "rev-q",
            "--reference-revision",
            "rev-r",
        ]
    )
    assert rc == 0
    assert captured["kw"]["quant_revision"] == "rev-q"
    assert captured["kw"]["reference_revision"] == "rev-r"


def test_compare_kv_forwards_filter_kwargs(monkeypatch, capsys):
    captured = {}

    def fake(model, configs, **kw):
        captured["kw"] = kw
        return _fake_comparison("kv")

    monkeypatch.setattr(cli, "compare_kv_fidelity", fake)
    rc = cli.main(
        [
            "compare",
            "kv",
            "m",
            "--configs",
            "4:32,8:64",
            "--min-tier",
            "good",
            "--max-kld",
            "0.3",
            "--quantize-start",
            "0",
        ]
    )
    assert rc == 0
    assert captured["kw"]["min_tier"] == "good"
    assert captured["kw"]["max_kld"] == 0.3
    assert captured["kw"]["quantize_start"] == 0


def test_compare_weights_invalid_args_exits_2(capsys):
    rc = main(["compare", "weights", "only/one", "--reference", "ref/repo"])
    assert rc == 2
    assert "at least 2 quant targets" in capsys.readouterr().err


def test_compare_does_not_swallow_unexpected_valueerror():
    with (
        patch(
            "mlx_quant_fidelity.cli.compare_weight_fidelity",
            side_effect=ValueError("unexpected boom"),
        ),
        pytest.raises(ValueError, match="unexpected boom"),
    ):
        main(["compare", "weights", "a/x-4bit", "b/y-8bit", "--reference", "ref/repo"])


def test_sweep_reads_local_config_json(monkeypatch, tmp_path, capsys):
    """Bug: --sweep on a local model directory asks the Hub for a repo named after the path."""
    import huggingface_hub

    (tmp_path / "config.json").write_text(json.dumps(_SWEEP_CONFIG_JSON))

    def boom(*a, **k):
        raise AssertionError("hub must not be called for a local directory")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    captured = {}

    def fake_compare(model, configs, **kw):
        captured["configs"] = configs
        return _fake_comparison("kv")

    monkeypatch.setattr(cli, "compare_kv_fidelity", fake_compare)
    assert main(["compare", "kv", str(tmp_path), "--sweep"]) == 0
    assert len(captured["configs"]) == 10


def test_sweep_config_fetch_passes_revision(monkeypatch):
    """Bug: --sweep reads the default-branch config while the run measures --model-revision."""
    seen = {}

    def fake_read(model, revision=None):
        seen["args"] = (model, revision)
        return _SWEEP_CONFIG_JSON

    monkeypatch.setattr(cli, "read_model_config", fake_read)
    monkeypatch.setattr(cli, "compare_kv_fidelity", lambda *a, **k: _fake_comparison("kv"))
    assert main(["compare", "kv", "org/m", "--sweep", "--model-revision", "abc"]) == 0
    assert seen["args"] == ("org/m", "abc")


def test_kv_refuses_custom_code_before_load(monkeypatch, capsys):
    """Bug: a repo naming a model_file reaches mlx_lm.load, which executes it."""
    import mlx_lm

    from mlx_quant_fidelity.probes import _preload

    monkeypatch.setattr(_preload, "_fetch_config", lambda m, r=None: ({"model_file": "m.py"}, None))

    def boom(*a, **k):
        raise AssertionError("mlx_lm.load must not run")

    monkeypatch.setattr(mlx_lm, "load", boom)
    rc = main(["kv", "evil/repo"])
    assert rc == 2
    assert "--allow-custom-code" in capsys.readouterr().err


def test_kv_allow_custom_code_flag_is_threaded(monkeypatch, capsys):
    captured = {}

    def fake_measure(model, **kw):
        captured.update(kw)
        raise cli.QuantFidelityError("stop")

    monkeypatch.setattr(cli, "measure_kv_fidelity", fake_measure)
    main(["kv", "m", "--allow-custom-code"])
    assert captured["allow_custom_code"] is True


# ── regression: compare kv --sweep + KV-byte budget filter ────────────────────

_SWEEP_CONFIG_JSON = {
    "num_hidden_layers": 16,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "hidden_size": 512,
}  # head_dim = 512 // 8 = 64


def test_cli_sweep_and_configs_mutually_exclusive(capsys):
    assert main(["compare", "kv", "m", "--configs", "4:64,8:64", "--sweep"]) == 2


def test_cli_kv_neither_configs_nor_sweep_exits_2(capsys):
    assert main(["compare", "kv", "m"]) == 2


def test_cli_max_kv_bytes_without_sweep_exits_2(capsys):
    rc = main(["compare", "kv", "m", "--configs", "4:64,8:64", "--max-kv-bytes-per-token", "5000"])
    assert rc == 2
    assert "sweep" in capsys.readouterr().err.lower()


def test_cli_sweep_dispatches_generated_grid(monkeypatch, capsys):
    captured = {}

    def fake_compare(model, configs, **kw):
        captured["args"] = (model, configs, kw)
        return _fake_comparison("kv")

    monkeypatch.setattr(cli, "compare_kv_fidelity", fake_compare)
    monkeypatch.setattr(
        cli, "read_model_config", lambda model_id, revision=None: _SWEEP_CONFIG_JSON
    )
    rc = main(["compare", "kv", "m", "--sweep"])
    assert rc == 0
    model, configs, kw = captured["args"]
    assert model == "m"
    assert StockKVMethod(bits=4, group_size=64) in configs
    assert len(configs) == 10  # 5 bits x {32, 64}; 128 doesn't divide head_dim=64
    assert kw["skipped_configs"] == []


def test_cli_sweep_head_dim_none_exits_2(monkeypatch, capsys):
    monkeypatch.setattr(cli, "read_model_config", lambda model_id, revision=None: {})
    rc = main(["compare", "kv", "m", "--sweep"])
    assert rc == 2
    assert "head_dim" in capsys.readouterr().err.lower()


def test_cli_sweep_with_budget_filters_and_names_skips(monkeypatch, capsys):
    captured = {}

    def fake_compare(model, configs, **kw):
        captured["args"] = (model, configs, kw)
        return _fake_comparison("kv")

    monkeypatch.setattr(cli, "compare_kv_fidelity", fake_compare)
    monkeypatch.setattr(
        cli, "read_model_config", lambda model_id, revision=None: _SWEEP_CONFIG_JSON
    )
    rc = main(["compare", "kv", "m", "--sweep", "--max-kv-bytes-per-token", "9000"])
    assert rc == 0
    _, configs, kw = captured["args"]
    assert 2 <= len(configs) < 10
    assert kw["skipped_configs"]  # some configs were pushed to skipped by the budget


def test_cli_sweep_budget_below_two_kept_exits_2(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "read_model_config", lambda model_id, revision=None: _SWEEP_CONFIG_JSON
    )
    rc = main(["compare", "kv", "m", "--sweep", "--max-kv-bytes-per-token", "1"])
    assert rc == 2
    assert "max-kv-bytes-per-token" in capsys.readouterr().err.lower()


def test_compare_kv_model_revision_threads(monkeypatch, capsys):
    """RED if `compare kv` has no --model-revision flag, or it is not forwarded to
    compare_kv_fidelity.
    """
    captured = {}

    def fake(model, configs, **kw):
        captured["kw"] = kw
        return _fake_comparison("kv")

    monkeypatch.setattr(cli, "compare_kv_fidelity", fake)
    rc = cli.main(["compare", "kv", "m", "--configs", "4:64,8:64", "--model-revision", "abc123"])
    assert rc == 0
    assert captured["kw"]["model_revision"] == "abc123"


def test_cli_sweep_budget_with_incomplete_geometry_exits_2(monkeypatch, capsys):
    """head_dim is resolvable (explicit key) but n_layers/n_kv_heads are not; the budget
    can't be costed, so this must exit 2 rather than crash inside filter_configs_by_kv_budget.
    """
    monkeypatch.setattr(cli, "read_model_config", lambda model_id, revision=None: {"head_dim": 64})
    rc = main(["compare", "kv", "m", "--sweep", "--max-kv-bytes-per-token", "9000"])
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "max-kv-bytes-per-token" in err
    assert "num_hidden_layers" in err or "num_key_value_heads" in err


@pytest.mark.parametrize(
    ("extra", "expected"),
    [([], 21600.0), (["--worker-timeout", "90"], 90.0), (["--worker-timeout", "0"], None)],
)
def test_compare_weights_worker_timeout_flag(monkeypatch, extra, expected):
    """Reds if --worker-timeout is ignored, or 0 fails to disable the limit."""
    captured = {}

    def fake(quant_ids, reference, **kw):
        captured.update(kw)
        return _fake_comparison("weight")

    monkeypatch.setattr(cli, "compare_weight_fidelity", fake)
    assert cli.main(["compare", "weights", "a", "b", "--reference", "r", *extra]) == 0
    assert captured["worker_timeout_s"] == expected


def _failed_row(label):
    from mlx_quant_fidelity.report import ComparisonTargetResult

    return ComparisonTargetResult(label, "failed", None, None, None, "WorkerError", "boom")


@pytest.mark.parametrize("mode", ["weights", "kv"])
def test_compare_all_failed_exits_nonzero(monkeypatch, capsys, mode):
    """Reds if a compare where every row failed still exits 0 (scripts read it as success)."""
    import dataclasses

    report = dataclasses.replace(
        _fake_comparison("weight" if mode == "weights" else "kv"),
        results=(_failed_row("a"), _failed_row("b")),
    )
    monkeypatch.setattr(cli, "compare_weight_fidelity", lambda *a, **k: report)
    monkeypatch.setattr(cli, "compare_kv_fidelity", lambda *a, **k: report)
    argv = (
        ["compare", "weights", "a", "b", "--reference", "r"]
        if mode == "weights"
        else ["compare", "kv", "m", "--configs", "4:64,8:64"]
    )
    assert cli.main(argv) == 1
    assert "boom" in capsys.readouterr().out  # the report is still printed to stdout


def test_compare_with_one_ok_row_exits_zero(monkeypatch):
    import dataclasses

    from mlx_quant_fidelity.report import ComparisonTargetResult

    ok = ComparisonTargetResult("a", "ok", None, None, None, None, None)
    report = dataclasses.replace(_fake_comparison("kv"), results=(ok, _failed_row("b")))
    monkeypatch.setattr(cli, "compare_kv_fidelity", lambda *a, **k: report)
    assert cli.main(["compare", "kv", "m", "--configs", "4:64,8:64"]) == 0


def test_compare_weights_reports_target_progress_on_stderr(monkeypatch, capsys):
    """Reds if progress lands on stdout (corrupts the report) or is never wired to stderr."""

    def fake(quant_ids, reference, *, progress=None, **kw):
        assert progress is not None
        progress("[1/2] a")
        return _fake_comparison("weight")

    monkeypatch.setattr(cli, "compare_weight_fidelity", fake)
    assert cli.main(["compare", "weights", "a", "b", "--reference", "r", "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert "[1/2] a" in captured.err
    assert "[1/2]" not in captured.out
    assert json.loads(captured.out)["mode"] == "weight"


def test_compare_kv_reports_progress_on_stderr(monkeypatch, capsys):
    def fake(model, configs, *, progress=None, **kw):
        assert progress is not None
        progress("[1/2] 4:64")
        return _fake_comparison("kv")

    monkeypatch.setattr(cli, "compare_kv_fidelity", fake)
    assert cli.main(["compare", "kv", "m", "--configs", "4:64,8:64", "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert "[1/2] 4:64" in captured.err
    assert json.loads(captured.out)["mode"] == "kv"


def test_compare_weights_negative_worker_timeout_is_a_usage_error(monkeypatch, capsys):
    """Bug: a negative --worker-timeout silently meant 'no limit' (the `> 0` else-branch)."""
    monkeypatch.setattr(cli, "compare_weight_fidelity", lambda *a, **k: pytest.fail("must not run"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["compare", "weights", "a", "b", "--reference", "r", "--worker-timeout", "-5"])
    assert exc.value.code == 2
    assert "worker-timeout" in capsys.readouterr().err


def test_compare_weights_infinite_worker_timeout_is_a_usage_error(monkeypatch, capsys):
    """Bug: `--worker-timeout inf` passed the >= 0 check and crashed subprocess.run with an
    OverflowError reported as an internal error (exit 1) instead of a usage error."""
    monkeypatch.setattr(cli, "compare_weight_fidelity", lambda *a, **k: pytest.fail("must not run"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["compare", "weights", "a", "b", "--reference", "r", "--worker-timeout", "inf"])
    assert exc.value.code == 2
    assert "worker-timeout" in capsys.readouterr().err


def test_compare_weights_unreadable_reference_exits_2(monkeypatch, capsys, tmp_path):
    """Bug: an unreadable shared reference exits 1 (a report of failed rows) instead of 2."""
    from mlx_quant_fidelity.errors import ModelNotAccessibleError
    from mlx_quant_fidelity.probes import _preload
    from mlx_quant_fidelity.runners import compare as cmp

    def _fetch(model, revision=None):
        if model == "ref":
            raise ModelNotAccessibleError("could not read config.json for 'ref'")
        return {}, None

    monkeypatch.setattr(_preload, "_fetch_config", _fetch)
    monkeypatch.setattr(
        cmp, "_run_weight_target", lambda *a, **k: pytest.fail("no worker may spawn")
    )
    monkeypatch.chdir(tmp_path)
    assert main(["compare", "weights", "q8", "q4", "--reference", "ref"]) == 2
    assert "ref" in capsys.readouterr().err


def test_compare_weights_oversized_worker_timeout_is_a_usage_error(monkeypatch, capsys):
    """Bug: a timeout above ~2,147,483 s passes the >= 0 check and overflows
    subprocess.run on macOS (an internal error, exit 1) instead of a usage error."""
    monkeypatch.setattr(cli, "compare_weight_fidelity", lambda *a, **k: pytest.fail("must not run"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["compare", "weights", "a", "b", "--reference", "r", "--worker-timeout", "3e9"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "worker-timeout" in err
    assert "use 0 to disable" in err
    assert "2,000,000" in err  # readable limit, not "2e+06"


def test_worker_timeout_at_the_bound_is_accepted():
    assert cli._non_negative_seconds("2000000") == 2_000_000.0
