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
