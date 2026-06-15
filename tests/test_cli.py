import json

import pytest

from mlx_quant_fidelity import cli
from mlx_quant_fidelity.corpora.provenance import CorpusProvenance
from mlx_quant_fidelity.errors import QuantFidelityError
from mlx_quant_fidelity.metrics import ScalarSummary
from mlx_quant_fidelity.report import FidelityReport, WeightFidelityReport


def _fake_report() -> FidelityReport:
    return FidelityReport(
        "m",
        None,
        4,
        64,
        0,
        "stress",
        ScalarSummary(0.02, 0.01, 0.2, 1.5),
        0.03,
        10.0,
        10.4,
        0.4,
        100,
        1,
        CorpusProvenance("wikitext-2-raw", "test", "tok", 512, 512, "none", "drop", "raw", 100),
        "0.21",
        "0.31.3",
        1,
        True,
        "marginal",
        (),
    )


def test_cli_kv_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "measure_kv_fidelity", lambda *a, **k: _fake_report())
    rc = cli.main(["kv", "mlx-community/x", "--kv-bits", "4", "--format", "json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "marginal"


def test_cli_installs_caps_before_measure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "install_memory_caps", lambda: calls.append("caps") or (20, 22))
    monkeypatch.setattr(
        cli, "measure_kv_fidelity", lambda *a, **k: calls.append("measure") or _fake_report()
    )
    cli.main(["kv", "mlx-community/x"])
    assert calls == ["caps", "measure"]  # caps FIRST


def test_cli_reports_domain_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _raise(*_a: object, **_k: object) -> FidelityReport:
        raise QuantFidelityError("deployment mode not supported")

    monkeypatch.setattr(cli, "measure_kv_fidelity", _raise)
    rc = cli.main(["kv", "m"])
    assert rc == 2
    assert "deployment mode not supported" in capsys.readouterr().err


def _weight_report() -> WeightFidelityReport:
    return WeightFidelityReport(
        quant_model_id="org/m-4bit",
        quant_revision=None,
        reference_model_id="org/m-bf16",
        reference_revision=None,
        quant_bits=4,
        quant_group_size=64,
        quant_mode="affine",
        per_layer=False,
        reference_bits=None,
        kl=ScalarSummary(0.06, 0.03, 0.4, 2.0),
        flip_rate=0.03,
        perplexity_ref=10.0,
        perplexity_quant=10.6,
        perplexity_delta=0.6,
        n_positions=1000,
        n_chunks=2,
        corpus=CorpusProvenance(
            "wikitext-2-raw", "test", "org/m-bf16", 512, 512, "none", "drop", "raw", 1024
        ),
        mlx_version="0.21",
        mlx_lm_version="0.31.3",
        peak_memory_bytes=18_000_000_000,
        quant_model_bytes=4_000_000_000,
        reference_model_bytes=14_000_000_000,
        verdict="marginal",
        warnings=(),
    )


def test_weights_subcommand_dispatches_and_renders_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_measure(quant: str, reference: str, **kw: object) -> WeightFidelityReport:
        captured["args"] = (quant, reference, kw)
        return _weight_report()

    monkeypatch.setattr(cli, "measure_weight_fidelity", fake_measure)
    rc = cli.main(["weights", "org/m-4bit", "--reference", "org/m-bf16", "--format", "json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["quant_model_id"] == "org/m-4bit"
    args = captured["args"]
    assert isinstance(args, tuple)
    assert args[0] == "org/m-4bit"
    assert args[1] == "org/m-bf16"
    assert captured["args"][2]["max_chunks"] is None


def test_weights_subcommand_reports_domain_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mlx_quant_fidelity.errors import ModelMismatchError

    def boom(*a: object, **k: object) -> WeightFidelityReport:
        raise ModelMismatchError("vocab_size mismatch")

    monkeypatch.setattr(cli, "measure_weight_fidelity", boom)
    rc = cli.main(["weights", "q", "--reference", "r"])
    assert rc == 2
    assert "vocab_size mismatch" in capsys.readouterr().err


def test_weights_subcommand_forwards_max_chunks(monkeypatch, capsys):
    captured = {}

    def fake_measure(quant, reference, **kw):
        captured["kw"] = kw
        return _weight_report()

    monkeypatch.setattr(cli, "measure_weight_fidelity", fake_measure)
    rc = cli.main(["weights", "q", "--reference", "r", "--max-chunks", "3"])
    assert rc == 0
    assert captured["kw"]["max_chunks"] == 3
