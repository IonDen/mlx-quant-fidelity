import dataclasses
import json

import pytest
from tests.test_compare_report import _wreport

from mlx_quant_fidelity.errors import CompareConfigError
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
        lambda quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None: (
            envelopes[quant]
        ),
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
        lambda quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None: (
            envelopes[quant]
        ),
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
        lambda quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None: (
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
    with pytest.raises(CompareConfigError, match="at least 2"):
        cmp.compare_weight_fidelity(["q8"], "ref", artifacts_dir=tmp_path)


def test_compare_weight_rejects_duplicate_ids(tmp_path):
    with pytest.raises(CompareConfigError, match="duplicate"):
        cmp.compare_weight_fidelity(["q8", "q8"], "ref", artifacts_dir=tmp_path)


def test_compare_weight_resume_skips_existing_partial(monkeypatch, tmp_path):
    # fix 6(a): pre-write partial; mock writes its own partial and records calls.
    # Partial must include a matching run_identity (quant='q8', reference='ref', max_chunks=None).
    (tmp_path / "q8.json").write_text(
        json.dumps(_weight_ok_envelope_with_identity("q8", 0.01, 8000))
    )
    calls = []

    def _fake_run(
        quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None
    ):
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

    def _fake_run(
        quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None
    ):
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
    with pytest.raises(CompareConfigError, match="collision"):
        cmp.compare_weight_fidelity(["a/b", "a_b"], "ref", artifacts_dir=tmp_path)


def test_compare_weight_rejects_nul_in_repo_id(tmp_path):
    """A NUL byte in a repo id must be rejected up front."""
    with pytest.raises(CompareConfigError, match="NUL"):
        cmp.compare_weight_fidelity(["q8\x00", "q9"], "ref", artifacts_dir=tmp_path)


def test_compare_weight_rejects_oversized_filename(tmp_path):
    """A repo id whose partial filename exceeds 255 bytes must be rejected up front."""
    long_repo = "a" * 253  # + ".json" = 258 bytes, over the 255 limit
    with pytest.raises(CompareConfigError, match="255"):
        cmp.compare_weight_fidelity([long_repo, "q9"], "ref", artifacts_dir=tmp_path)


# ── fix 9: missing envelope keys yield None, not "None" ──────────────────────


def test_compare_weight_failed_missing_envelope_keys_are_none(monkeypatch, tmp_path):
    """A 'failed' envelope with absent error_type/message keys → None, not the string 'None'."""
    monkeypatch.setattr(
        cmp,
        "_run_weight_target",
        lambda quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None: {
            "status": "failed"
        },
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

    def _fake_run(
        quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None
    ):
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
    quant_revision: str | None = None,
    reference_revision: str | None = None,
) -> dict[str, object]:
    """Build a weight partial envelope with a run_identity block.

    quant defaults to label (the common case).
    schema_version defaults to _WEIGHT_PARTIAL_SCHEMA_VERSION from the live module.
    """
    import mlx_quant_fidelity.runners.compare as compare_mod

    sv = (
        schema_version if schema_version is not None else compare_mod._WEIGHT_PARTIAL_SCHEMA_VERSION
    )
    identity: dict[str, object] = {
        "mode": "weight",
        "quant": quant if quant is not None else label,
        "reference": reference,
        "max_chunks": max_chunks,
        "schema_version": sv,
        "quant_revision": quant_revision,
        "reference_revision": reference_revision,
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

    def _fake_run(
        quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None
    ):
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

    def _fake_run(
        quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None
    ):
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

    def _fake_run(
        quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None
    ):
        calls.append(quant)
        env = _weight_ok_envelope_with_identity(quant, 0.04, 6200, max_chunks=max_chunks)
        partial_path.write_text(json.dumps(env))
        return env

    monkeypatch.setattr(cmp, "_run_weight_target", _fake_run)

    cmp.compare_weight_fidelity(["q8", "q9"], "ref", max_chunks=5, artifacts_dir=tmp_path)

    assert "q8" not in calls, "q8 must be resumed from partial; run_identity matches"
    assert "q9" in calls, "q9 must be run (no pre-existing partial)"


def test_compare_weight_stale_reference_partial_is_recomputed(monkeypatch, tmp_path):
    """A partial whose run_identity records reference='ref-A' must NOT resume for reference 'ref-B'.

    Uses a full, same-shape run_identity differing ONLY in the reference field, so the recompute
    is driven by the reference field mismatch specifically — a mutation that excluded reference
    from the identity comparison would resume the stale partial and go red. (The earlier form wrote
    no run_identity at all, so it only proved "absent identity recomputes".)
    """
    stale_env = _weight_ok_envelope_with_identity("q8", 0.01, 8000, reference="ref-A")
    (tmp_path / "q8.json").write_text(json.dumps(stale_env))

    calls = []

    def _fake_run(
        quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None
    ):
        calls.append((quant, reference))
        env = _weight_ok_envelope_with_identity(quant, 0.01, 8000, reference=reference)
        partial_path.write_text(json.dumps(env))
        return env

    monkeypatch.setattr(cmp, "_run_weight_target", _fake_run)

    # Call with ref-B — the stale q8 partial (run_identity.reference='ref-A') must be recomputed
    cmp.compare_weight_fidelity(["q8", "q9"], "ref-B", artifacts_dir=tmp_path)

    assert any(quant == "q8" and ref == "ref-B" for quant, ref in calls)


def test_validate_compare_weights_args_rejects_single_target():
    from mlx_quant_fidelity.runners.compare import _validate_compare_weights_args

    with pytest.raises(CompareConfigError, match="at least 2 quant targets"):
        _validate_compare_weights_args(["only/one"])


def test_weight_envelope_missing_report_body_is_corrupt_partial():
    from mlx_quant_fidelity.runners.compare import _envelope_to_result

    env = _weight_ok_envelope_with_identity("q4", 0.09, 4200)
    del env["report"]  # Layer 1: envelope has no "report" key
    result = _envelope_to_result("q4", env)
    assert result.status == "failed"
    assert result.error_type == "CorruptPartial"


def test_weight_envelope_malformed_report_body_is_corrupt_partial():
    from mlx_quant_fidelity.runners.compare import _envelope_to_result

    env = _weight_ok_envelope_with_identity("q4", 0.09, 4200)
    del env["report"]["kl"]  # Layer 2: "report" present but body missing "kl"
    result = _envelope_to_result("q4", env)
    assert result.status == "failed"
    assert result.error_type == "CorruptPartial"


def test_weight_collect_isolates_malformed_cached_partial(monkeypatch, tmp_path):
    """All-cached: a pre-seeded partial with a malformed report body is isolated; run completes."""
    import json

    good = _weight_ok_envelope_with_identity("q8", 0.01, 8000)
    bad = _weight_ok_envelope_with_identity("q6", 0.04, 6200)
    del bad["report"]["kl"]  # malformed body; run_identity still valid → no re-run
    (tmp_path / "q8.json").write_text(json.dumps(good))
    (tmp_path / "q6.json").write_text(json.dumps(bad))

    def _boom(*a, **k):
        raise AssertionError("worker must not run for cached partials")

    monkeypatch.setattr(cmp, "_run_weight_target", _boom)
    report = cmp.compare_weight_fidelity(["q8", "q6"], "ref", artifacts_dir=tmp_path)
    q6 = next(r for r in report.results if r.label == "q6")
    assert q6.status == "failed"
    assert q6.error_type == "CorruptPartial"
    assert "q8" in report.frontier  # the good target still ranks; no abort


def test_weight_envelope_with_invalid_verdict_is_corrupt_partial():
    from mlx_quant_fidelity.runners.compare import _envelope_to_result

    env = _weight_ok_envelope_with_identity("q4", 0.09, 4200)
    # Precondition: the converter must return status=="ok" on the valid (un-tampered) envelope.
    # If this fails, the test below would be vacuously true (converter always fails).
    precondition_result = _envelope_to_result("q4", env)
    assert precondition_result.status == "ok", (
        f"Precondition failed: expected 'ok' on valid envelope, got {precondition_result.status!r}"
    )
    env["report"]["verdict"] = "nonsense"
    result = _envelope_to_result("q4", env)
    assert result.status == "failed"
    assert result.error_type == "CorruptPartial"


# ── regression: non-dict top-level partial isolation ───────────────────────────


def test_weight_envelope_non_dict_is_corrupt_partial():
    """`_envelope_to_result` on a non-dict envelope is a CorruptPartial, not an
    AttributeError crash (previously `env.get("status")` raised on an int/list).
    """
    from mlx_quant_fidelity.runners.compare import _envelope_to_result

    result = _envelope_to_result("repo/x", 42)  # type: ignore[arg-type]
    assert (result.status, result.error_type) == ("failed", "CorruptPartial")


def test_compare_weight_null_partial_is_recomputed(monkeypatch, tmp_path):
    """A partial file whose top-level JSON is valid but not an object (`null`) must be
    treated as absent by the resume loop — the target is recomputed, not a crash.
    """
    (tmp_path / "q8.json").write_text("null")

    calls = []

    def _fake_run(
        quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None
    ):
        calls.append(quant)
        env = _ok_envelope(quant, 0.01, 8000)
        partial_path.write_text(json.dumps(env))
        return env

    monkeypatch.setattr(cmp, "_run_weight_target", _fake_run)
    report = cmp.compare_weight_fidelity(["q8", "q6"], "ref", artifacts_dir=tmp_path)
    assert "q8" in calls  # non-dict partial triggered a re-run
    assert len(report.results) == 2


# ── regression: the schema-version constant split ─────────────────────────────


def test_weight_partials_survive_unrelated_schema_bumps(monkeypatch, tmp_path):
    """A weight partial whose stored schema_version matches the current
    _WEIGHT_PARTIAL_SCHEMA_VERSION constant resumes without a forced recompute — the KV
    partial schema (a separate constant) bumps on its own, independent cadence.
    """
    assert cmp._WEIGHT_PARTIAL_SCHEMA_VERSION == 2

    matching_env = _weight_ok_envelope_with_identity("q8", 0.01, 8000, schema_version=2)
    (tmp_path / "q8.json").write_text(json.dumps(matching_env))

    calls = []

    def _fake_run(
        quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None
    ):
        calls.append(quant)
        env = _weight_ok_envelope_with_identity(quant, 0.04, 6200)
        partial_path.write_text(json.dumps(env))
        return env

    monkeypatch.setattr(cmp, "_run_weight_target", _fake_run)

    cmp.compare_weight_fidelity(["q8", "q9"], "ref", artifacts_dir=tmp_path)

    assert "q8" not in calls, "q8 must resume: the weight schema version (2) did not change"


def test_compare_weight_old_schema_version_partial_recomputes(monkeypatch, tmp_path):
    """A partial written at schema_version=1 (pre-revision-identity) must not resume once the
    live constant is 2 — the identity dict shape changed (it gained quant_revision/
    reference_revision), so an old partial's identity can never equal the new expected one.
    """
    stale_env = _weight_ok_envelope_with_identity("q8", 0.01, 8000, schema_version=1)
    (tmp_path / "q8.json").write_text(json.dumps(stale_env))

    calls = []

    def _fake_run(
        quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None
    ):
        calls.append(quant)
        env = _weight_ok_envelope_with_identity(quant, 0.01, 8000)
        partial_path.write_text(json.dumps(env))
        return env

    monkeypatch.setattr(cmp, "_run_weight_target", _fake_run)

    cmp.compare_weight_fidelity(["q8", "q9"], "ref", artifacts_dir=tmp_path)

    assert "q8" in calls, "a schema-1 partial must recompute under the schema-2 constant"


def test_compare_weight_stale_quant_revision_partial_is_recomputed(monkeypatch, tmp_path):
    """A partial pinned to quant_revision='rev-A' must NOT resume for a 'rev-B' run.

    Mirrors test_compare_weight_stale_reference_partial_is_recomputed but for the quant
    side's own revision pin — the two repos revision independently.
    """
    stale_env = _weight_ok_envelope_with_identity("q8", 0.01, 8000, quant_revision="rev-A")
    (tmp_path / "q8.json").write_text(json.dumps(stale_env))

    calls = []

    def _fake_run(
        quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None
    ):
        calls.append((quant, quant_revision))
        env = _weight_ok_envelope_with_identity(quant, 0.01, 8000, quant_revision=quant_revision)
        partial_path.write_text(json.dumps(env))
        return env

    monkeypatch.setattr(cmp, "_run_weight_target", _fake_run)

    cmp.compare_weight_fidelity(["q8", "q9"], "ref", quant_revision="rev-B", artifacts_dir=tmp_path)

    assert any(quant == "q8" and rev == "rev-B" for quant, rev in calls)


def test_compare_weight_stale_reference_revision_partial_is_recomputed(monkeypatch, tmp_path):
    """A partial pinned to reference_revision='rev-A' must NOT resume for a 'rev-B' run."""
    stale_env = _weight_ok_envelope_with_identity("q8", 0.01, 8000, reference_revision="rev-A")
    (tmp_path / "q8.json").write_text(json.dumps(stale_env))

    calls = []

    def _fake_run(
        quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None
    ):
        calls.append((quant, reference_revision))
        env = _weight_ok_envelope_with_identity(
            quant, 0.01, 8000, reference_revision=reference_revision
        )
        partial_path.write_text(json.dumps(env))
        return env

    monkeypatch.setattr(cmp, "_run_weight_target", _fake_run)

    cmp.compare_weight_fidelity(
        ["q8", "q9"], "ref", reference_revision="rev-B", artifacts_dir=tmp_path
    )

    assert any(quant == "q8" and rev == "rev-B" for quant, rev in calls)


def test_compare_weight_matching_revisions_resume(monkeypatch, tmp_path):
    """A partial whose stored quant_revision/reference_revision match the current call is
    resumed, not re-run — proves the new identity fields don't force an unnecessary recompute
    when nothing changed.
    """
    matching_env = _weight_ok_envelope_with_identity(
        "q8", 0.01, 8000, quant_revision="rev-A", reference_revision="rev-B"
    )
    (tmp_path / "q8.json").write_text(json.dumps(matching_env))

    calls = []

    def _fake_run(
        quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None
    ):
        calls.append(quant)
        env = _weight_ok_envelope_with_identity(
            quant, 0.04, 6200, quant_revision=quant_revision, reference_revision=reference_revision
        )
        partial_path.write_text(json.dumps(env))
        return env

    monkeypatch.setattr(cmp, "_run_weight_target", _fake_run)

    cmp.compare_weight_fidelity(
        ["q8", "q9"],
        "ref",
        quant_revision="rev-A",
        reference_revision="rev-B",
        artifacts_dir=tmp_path,
    )

    assert "q8" not in calls, "q8 must resume: quant_revision/reference_revision match"


def test_run_weight_target_passes_revisions_to_worker_cmd_only_when_set(monkeypatch, tmp_path):
    """`_run_weight_target` appends --quant-revision/--reference-revision to the worker cmd
    only when set — mirrors how --max-chunks is optionally appended — so a run with no
    revision pin doesn't grow a redundant flag with an empty/None value.
    """

    class _FakeCompletedProcess:
        stdout = ""
        stderr = ""

    captured: dict[str, list[str]] = {}
    out = tmp_path / "out.json"

    def fake_run(cmd, check, capture_output, text):
        captured["cmd"] = cmd
        out.write_text(json.dumps({"status": "ok", "report": {}}))
        return _FakeCompletedProcess()

    monkeypatch.setattr(cmp.subprocess, "run", fake_run)

    cmp._run_weight_target(
        "quant/repo",
        "ref/repo",
        out,
        None,
        quant_revision="rev-q",
        reference_revision=None,
    )

    cmd = captured["cmd"]
    assert "--quant-revision" in cmd
    assert cmd[cmd.index("--quant-revision") + 1] == "rev-q"
    assert "--reference-revision" not in cmd


# ── repo[@revision] target grammar (Task 5) ──────────────────────────────────


def test_split_target_grammar(tmp_path):
    """Reds on: splitting at the last `@`, splitting an existing local path, accepting an empty
    or whitespace revision."""
    assert cmp.split_target("org/m-4bit") == ("org/m-4bit", None)
    assert cmp.split_target("org/m-4bit@abc123") == ("org/m-4bit", "abc123")
    assert cmp.split_target("org/m@refs/pr/3") == ("org/m", "refs/pr/3")
    assert cmp.split_target("org/m@x@y") == ("org/m", "x@y")
    local = tmp_path / "model@2026"
    local.mkdir()
    assert cmp.split_target(str(local)) == (str(local), None)
    for bad in ("org/m@", "@abc", "org/m@re v", "org/m@a\x00b"):
        with pytest.raises(CompareConfigError):
            cmp.split_target(bad)


def test_compare_weight_rejects_same_repo_at_two_revisions(tmp_path):
    with pytest.raises(CompareConfigError, match="duplicate"):
        cmp.compare_weight_fidelity(["org/m", "org/m@abc"], "ref", artifacts_dir=tmp_path)


def test_compare_weight_inline_revisions_win_over_flags_and_reach_worker_label_and_filename(
    monkeypatch, tmp_path
):
    """Reds if the raw `repo@rev` string leaks into the label, the partial filename, or the
    worker's --quant argument, or if a flag overrides an inline pin on either side."""
    seen = []

    def _fake_run(
        quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None
    ):
        seen.append((quant, quant_revision, reference, reference_revision, partial_path.name))
        env = _weight_ok_envelope_with_identity(
            quant,
            0.04,
            6200,
            quant_revision=quant_revision,
            reference_revision=reference_revision,
            reference=reference,
        )
        partial_path.write_text(json.dumps(env))
        return env

    monkeypatch.setattr(cmp, "_run_weight_target", _fake_run)
    report = cmp.compare_weight_fidelity(
        ["org/q4@rev-4", "org/q8"],
        "org/ref@rev-R",
        quant_revision="fallback",
        reference_revision="ref-fallback",
        artifacts_dir=tmp_path,
    )
    assert seen == [
        ("org/q4", "rev-4", "org/ref", "rev-R", "org_q4.json"),
        ("org/q8", "fallback", "org/ref", "rev-R", "org_q8.json"),
    ]
    assert {r.label for r in report.results} == {"org/q4", "org/q8"}
    assert report.reference == "org/ref"


def test_compare_weight_inline_revision_resumes_on_the_second_call(monkeypatch, tmp_path):
    """The resume-identity bug: if the orchestrator expects the call-level quant_revision (None)
    while the worker stored the inline one, every `@`-pinned target re-runs forever."""
    calls = []

    def _fake_run(
        quant, reference, partial_path, max_chunks, quant_revision=None, reference_revision=None
    ):
        calls.append(quant)
        env = _weight_ok_envelope_with_identity(
            quant,
            0.04,
            6200,
            quant_revision=quant_revision,
            reference_revision=reference_revision,
            reference=reference,
        )
        partial_path.write_text(json.dumps(env))
        return env

    monkeypatch.setattr(cmp, "_run_weight_target", _fake_run)
    targets = ["org/q4@rev-4", "org/q8@rev-8"]
    cmp.compare_weight_fidelity(targets, "org/ref@rev-R", artifacts_dir=tmp_path)
    cmp.compare_weight_fidelity(targets, "org/ref@rev-R", artifacts_dir=tmp_path)
    assert calls == ["org/q4", "org/q8"], "second call must resume every pinned target"
