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
    assert report.corpus is not None  # fix 10: corpus must be populated


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
    # fix 8: full-signature lambda instead of **k
    monkeypatch.setattr(
        cmp,
        "_run_weight_target",
        lambda quant, reference, partial_path, max_chunks: (
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
    # fix 6(a): pre-write partial; mock writes its own partial and records calls.
    # Partial must include a matching run_identity (quant='q8', reference='ref', max_chunks=None).
    (tmp_path / "q8.json").write_text(
        json.dumps(_weight_ok_envelope_with_identity("q8", 0.01, 8000))
    )
    calls = []

    def _fake_run(quant, reference, partial_path, max_chunks):
        calls.append(quant)
        env = _weight_ok_envelope_with_identity(quant, 0.04, 6200)
        partial_path.write_text(json.dumps(env))  # mirror real worker: write partial
        return env

    monkeypatch.setattr(cmp, "_run_weight_target", _fake_run)
    cmp.compare_weight_fidelity(["q8", "q6"], "ref", artifacts_dir=tmp_path)
    assert calls == ["q6"]  # q8 resumed from its partial, not re-run
    assert (tmp_path / "q6.json").exists()  # partial written for the un-cached target


# ── fix 6(b): corrupt partial must re-run ────────────────────────────────────


def test_compare_weight_corrupt_partial_reruns(monkeypatch, tmp_path):
    """A truncated/invalid partial must not crash — the target is re-run."""
    (tmp_path / "q8.json").write_text("{bad json")  # corrupt partial
    calls = []

    def _fake_run(quant, reference, partial_path, max_chunks):
        calls.append(quant)
        env = _ok_envelope(quant, 0.01, 8000)
        partial_path.write_text(json.dumps(env))
        return env

    monkeypatch.setattr(cmp, "_run_weight_target", _fake_run)
    report = cmp.compare_weight_fidelity(["q8", "q6"], "ref", artifacts_dir=tmp_path)
    assert "q8" in calls  # corrupt partial triggered a re-run
    assert len(report.results) == 2  # both targets produced results


# ── fix 7: filename-collision guard ──────────────────────────────────────────


def test_compare_weight_rejects_filename_collision(tmp_path):
    """'a/b' and 'a_b' map to the same partial filename — must raise before any spawn."""
    with pytest.raises(ValueError, match="collision"):
        cmp.compare_weight_fidelity(["a/b", "a_b"], "ref", artifacts_dir=tmp_path)


def test_compare_weight_rejects_nul_in_repo_id(tmp_path):
    """A NUL byte in a repo id must be rejected up front."""
    with pytest.raises(ValueError, match="NUL"):
        cmp.compare_weight_fidelity(["q8\x00", "q9"], "ref", artifacts_dir=tmp_path)


def test_compare_weight_rejects_oversized_filename(tmp_path):
    """A repo id whose partial filename exceeds 255 bytes must be rejected up front."""
    long_repo = "a" * 253  # + ".json" = 258 bytes, over the 255 limit
    with pytest.raises(ValueError, match="255"):
        cmp.compare_weight_fidelity([long_repo, "q9"], "ref", artifacts_dir=tmp_path)


# ── fix 9: missing envelope keys yield None, not "None" ──────────────────────


def test_compare_weight_failed_missing_envelope_keys_are_none(monkeypatch, tmp_path):
    """A 'failed' envelope with absent error_type/message keys → None, not the string 'None'."""
    monkeypatch.setattr(
        cmp,
        "_run_weight_target",
        lambda quant, reference, partial_path, max_chunks: {"status": "failed"},
    )
    report = cmp.compare_weight_fidelity(["q8", "q9"], "ref", artifacts_dir=tmp_path)
    failed_q8 = next(r for r in report.results if r.label == "q8")
    assert failed_q8.status == "failed"
    assert failed_q8.error_type is None
    assert failed_q8.message is None


# ── Line 202: failed partial recomputes on resume ─────────────────────────────


def test_compare_weight_failed_partial_is_recomputed(monkeypatch, tmp_path):
    """A partial with status='failed' carries no stored identity and must be recomputed.

    Line 202 (elif env is not None: env = None) is the branch for non-ok, non-corrupt
    partials — they are discarded and the target is re-run.
    """
    from tests.test_compare_report import _wreport

    # Pre-seed a failed partial for q8 (no reference identity stored)
    failed_envelope = {"status": "failed", "error_type": "RuntimeError", "message": "boom"}
    (tmp_path / "q8.json").write_text(json.dumps(failed_envelope))

    calls = []

    def _fake_run(quant, reference, partial_path, max_chunks):
        calls.append(quant)
        env = {"status": "ok", "report": dataclasses.asdict(_wreport(quant, 0.01, 8000))}
        partial_path.write_text(json.dumps(env))
        return env

    monkeypatch.setattr(cmp, "_run_weight_target", _fake_run)

    cmp.compare_weight_fidelity(["q8", "q9"], "ref", artifacts_dir=tmp_path)

    # q8's failed partial must have been discarded — _run_weight_target called for it
    assert "q8" in calls


# ── Fix B (weight): stale-reference resume recomputes instead of resuming ─────


# ── Run-identity validation (finding 1) ───────────────────────────────────────


def _weight_ok_envelope_with_identity(
    label: str,
    kl_mean: float,
    cost: int,
    *,
    quant: str | None = None,
    reference: str = "ref",
    max_chunks: int | None = None,
    schema_version: int | None = None,
) -> dict[str, object]:
    """Build a weight partial envelope with a run_identity block.

    quant defaults to label (the common case).
    schema_version defaults to _PARTIAL_SCHEMA_VERSION from the live module.
    """
    import mlx_quant_fidelity.runners.compare as compare_mod

    sv = schema_version if schema_version is not None else compare_mod._PARTIAL_SCHEMA_VERSION
    identity: dict[str, object] = {
        "mode": "weight",
        "quant": quant if quant is not None else label,
        "reference": reference,
        "max_chunks": max_chunks,
        "schema_version": sv,
    }
    env = _ok_envelope(label, kl_mean, cost)
    env["run_identity"] = identity
    return env


def test_compare_weight_stale_max_chunks_partial_is_recomputed(monkeypatch, tmp_path):
    """A partial scored at max_chunks=2 must NOT be resumed for a max_chunks=100 run.

    Pre-seed q8.json with a run_identity recording max_chunks=2. Call with max_chunks=100.
    The identity mismatch must cause the orchestrator to discard the partial and call
    _run_weight_target for q8.
    """
    stale_env = _weight_ok_envelope_with_identity("q8", 0.01, 8000, max_chunks=2)
    (tmp_path / "q8.json").write_text(json.dumps(stale_env))

    calls = []

    def _fake_run(quant, reference, partial_path, max_chunks):
        calls.append(quant)
        env = _weight_ok_envelope_with_identity(quant, 0.01, 8000, max_chunks=max_chunks)
        partial_path.write_text(json.dumps(env))
        return env

    monkeypatch.setattr(cmp, "_run_weight_target", _fake_run)

    cmp.compare_weight_fidelity(["q8", "q9"], "ref", max_chunks=100, artifacts_dir=tmp_path)

    assert "q8" in calls, "q8 must be re-run; stale max_chunks=2 partial must not resume"


def test_compare_weight_sanitized_filename_collision_causes_recompute(monkeypatch, tmp_path):
    """A partial whose stored run_identity.quant differs from the requested repo must be recomputed.

    Scenario: a previous run for repo 'org/m' wrote a partial (sanitized to 'org_m.json' with
    run_identity.quant='org/m'). A later run for repo 'org_m' (different repo, same filename)
    must NOT resume that partial — the quant field mismatch triggers recompute.
    """
    # Partial was written for 'org/m' (different repo, same sanitized filename 'org_m.json')
    stale_env = _weight_ok_envelope_with_identity("org/m", 0.01, 8000, quant="org/m")
    (tmp_path / "org_m.json").write_text(json.dumps(stale_env))

    calls = []

    def _fake_run(quant, reference, partial_path, max_chunks):
        calls.append(quant)
        env = _weight_ok_envelope_with_identity(quant, 0.01, 8000, quant=quant)
        partial_path.write_text(json.dumps(env))
        return env

    monkeypatch.setattr(cmp, "_run_weight_target", _fake_run)

    cmp.compare_weight_fidelity(["org_m", "q9"], "ref", artifacts_dir=tmp_path)

    assert "org_m" in calls, "org_m must be re-run; stale partial for 'org/m' must not resume"


def test_compare_weight_matching_identity_resumes(monkeypatch, tmp_path):
    """A partial with a fully matching run_identity is correctly resumed (not re-run).

    When all identity fields (quant, reference, max_chunks, schema_version) match,
    _run_weight_target must NOT be called for that target.
    """
    matching_env = _weight_ok_envelope_with_identity("q8", 0.01, 8000, max_chunks=5)
    (tmp_path / "q8.json").write_text(json.dumps(matching_env))

    calls = []

    def _fake_run(quant, reference, partial_path, max_chunks):
        calls.append(quant)
        env = _weight_ok_envelope_with_identity(quant, 0.04, 6200, max_chunks=max_chunks)
        partial_path.write_text(json.dumps(env))
        return env

    monkeypatch.setattr(cmp, "_run_weight_target", _fake_run)

    cmp.compare_weight_fidelity(["q8", "q9"], "ref", max_chunks=5, artifacts_dir=tmp_path)

    assert "q8" not in calls, "q8 must be resumed from partial; run_identity matches"
    assert "q9" in calls, "q9 must be run (no pre-existing partial)"


def test_compare_weight_stale_reference_partial_is_recomputed(monkeypatch, tmp_path):
    """A partial measured against reference 'ref-A' is not resumed when compare is called
    with reference_model_id='ref-B'.  The stale partial must be treated as absent and
    _run_weight_target must be called for that target.

    The existing resume test (test_compare_weight_resume_skips_existing_partial) uses
    a MATCHING reference and must still pass.
    """
    from tests.test_compare_report import _wreport

    # Build a valid envelope for q8 that was measured against ref-A
    rep_a = _wreport("q8", 0.01, 8000)
    # Patch the reference_model_id in the serialised report to simulate ref-A measurement
    rep_dict = dataclasses.asdict(rep_a)
    rep_dict["reference_model_id"] = "ref-A"
    stale_envelope = {"status": "ok", "report": rep_dict}
    (tmp_path / "q8.json").write_text(json.dumps(stale_envelope))

    calls = []

    def _fake_run(quant, reference, partial_path, max_chunks):
        calls.append((quant, reference))
        env = {"status": "ok", "report": dataclasses.asdict(_wreport(quant, 0.01, 8000))}
        partial_path.write_text(json.dumps(env))
        return env

    monkeypatch.setattr(cmp, "_run_weight_target", _fake_run)

    # Call with ref-B — the stale q8 partial (measured against ref-A) must be recomputed
    cmp.compare_weight_fidelity(["q8", "q9"], "ref-B", artifacts_dir=tmp_path)

    # q8 must have been re-run against ref-B
    assert any(quant == "q8" and ref == "ref-B" for quant, ref in calls)
