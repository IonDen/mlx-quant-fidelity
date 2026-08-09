import json
from unittest.mock import patch

import pytest

from mlx_quant_fidelity import cli
from mlx_quant_fidelity.cli import main
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
    assert captured["args"][1] == [(4, 32), (4, 64), (8, 64)]
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


# ── Task 8 (0035): compare kv --sweep + KV-byte budget filter ─────────────────

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
    monkeypatch.setattr(cli, "_fetch_model_config", lambda model_id: _SWEEP_CONFIG_JSON)
    rc = main(["compare", "kv", "m", "--sweep"])
    assert rc == 0
    model, configs, kw = captured["args"]
    assert model == "m"
    assert (4, 64) in configs
    assert len(configs) == 10  # 5 bits x {32, 64}; 128 doesn't divide head_dim=64
    assert kw["skipped_configs"] == []


def test_cli_sweep_head_dim_none_exits_2(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_fetch_model_config", lambda model_id: {})
    rc = main(["compare", "kv", "m", "--sweep"])
    assert rc == 2
    assert "head_dim" in capsys.readouterr().err.lower()


def test_cli_sweep_with_budget_filters_and_names_skips(monkeypatch, capsys):
    captured = {}

    def fake_compare(model, configs, **kw):
        captured["args"] = (model, configs, kw)
        return _fake_comparison("kv")

    monkeypatch.setattr(cli, "compare_kv_fidelity", fake_compare)
    monkeypatch.setattr(cli, "_fetch_model_config", lambda model_id: _SWEEP_CONFIG_JSON)
    rc = main(["compare", "kv", "m", "--sweep", "--max-kv-bytes-per-token", "9000"])
    assert rc == 0
    _, configs, kw = captured["args"]
    assert 2 <= len(configs) < 10
    assert kw["skipped_configs"]  # some configs were pushed to skipped by the budget


def test_cli_sweep_budget_below_two_kept_exits_2(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_fetch_model_config", lambda model_id: _SWEEP_CONFIG_JSON)
    rc = main(["compare", "kv", "m", "--sweep", "--max-kv-bytes-per-token", "1"])
    assert rc == 2
    assert "max-kv-bytes-per-token" in capsys.readouterr().err.lower()


def test_cli_sweep_budget_with_incomplete_geometry_exits_2(monkeypatch, capsys):
    """head_dim is resolvable (explicit key) but n_layers/n_kv_heads are not; the budget
    can't be costed, so this must exit 2 rather than crash inside filter_configs_by_kv_budget.
    """
    monkeypatch.setattr(cli, "_fetch_model_config", lambda model_id: {"head_dim": 64})
    rc = main(["compare", "kv", "m", "--sweep", "--max-kv-bytes-per-token", "9000"])
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "max-kv-bytes-per-token" in err
    assert "num_hidden_layers" in err or "num_key_value_heads" in err
