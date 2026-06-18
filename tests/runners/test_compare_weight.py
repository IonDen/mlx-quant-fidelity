import dataclasses
import json

import pytest
from tests.test_compare_report import _wreport

from mlx_quant_fidelity.runners import compare as cmp


def _ok_envelope(label, kl_mean, cost):
    rep = _wreport(label, kl_mean, cost)
    return {"status": "ok", "report": dataclasses.asdict(rep)}


def test_compare_weight_builds_frontier(monkeypatch, tmp_path):
    envelopes = {
        "q8": _ok_envelope("q8", 0.01, 8000),
        "q6": _ok_envelope("q6", 0.04, 6200),
        "q4": _ok_envelope("q4", 0.09, 4200),
    }
    monkeypatch.setattr(
        cmp,
        "_run_weight_target",
        lambda quant, reference, partial_path, max_chunks: envelopes[quant],
    )
    report = cmp.compare_weight_fidelity(["q8", "q6", "q4"], "ref", artifacts_dir=tmp_path)
    assert set(report.frontier) == {"q4", "q6", "q8"}
    assert report.mode == "weight"
    assert report.reference == "ref"


def test_compare_weight_unrankable_when_cost_none(monkeypatch, tmp_path):
    nocost = _ok_envelope("q4", 0.09, 4200)
    nocost["report"]["quant_model_bytes"] = None  # cost unavailable (audit #3)
    envelopes = {"q4": nocost, "q8": _ok_envelope("q8", 0.01, 8000)}
    monkeypatch.setattr(
        cmp,
        "_run_weight_target",
        lambda quant, reference, partial_path, max_chunks: envelopes[quant],
    )
    report = cmp.compare_weight_fidelity(["q4", "q8"], "ref", artifacts_dir=tmp_path)
    res = next(r for r in report.results if r.label == "q4")
    assert res.status == "ok"
    assert res.point is None
    assert res.excluded_reason == "cost unavailable"
    assert "q4" not in report.frontier
    assert "q8" in report.frontier


def test_compare_weight_failed_target_isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cmp,
        "_run_weight_target",
        lambda quant, **k: (
            {"status": "failed", "error_type": "ModelMismatchError", "message": "bad"}
            if quant == "q2"
            else _ok_envelope(quant, 0.01, 8000)
        ),
    )
    report = cmp.compare_weight_fidelity(["q8", "q2"], "ref", artifacts_dir=tmp_path)
    failed = next(r for r in report.results if r.label == "q2")
    assert failed.status == "failed"
    assert failed.error_type == "ModelMismatchError"
    assert "q2" not in report.frontier
    assert "q8" in report.frontier


def test_compare_weight_requires_two_targets(tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        cmp.compare_weight_fidelity(["q8"], "ref", artifacts_dir=tmp_path)


def test_compare_weight_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        cmp.compare_weight_fidelity(["q8", "q8"], "ref", artifacts_dir=tmp_path)


def test_compare_weight_resume_skips_existing_partial(monkeypatch, tmp_path):
    (tmp_path / "q8.json").write_text(json.dumps(_ok_envelope("q8", 0.01, 8000)))
    calls = []
    monkeypatch.setattr(
        cmp,
        "_run_weight_target",
        lambda quant, **k: calls.append(quant) or _ok_envelope(quant, 0.04, 6200),
    )
    cmp.compare_weight_fidelity(["q8", "q6"], "ref", artifacts_dir=tmp_path)
    assert calls == ["q6"]  # q8 resumed from its partial, not re-run
