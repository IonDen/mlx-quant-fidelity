import json

import pytest

from mlx_quant_fidelity import cli
from mlx_quant_fidelity.corpora.provenance import CorpusProvenance
from mlx_quant_fidelity.metrics import ScalarSummary
from mlx_quant_fidelity.report import FidelityReport


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
