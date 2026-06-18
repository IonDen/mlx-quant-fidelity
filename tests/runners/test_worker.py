import json

from tests.test_compare_report import _wreport

from mlx_quant_fidelity.runners import _worker


def test_worker_writes_ok_envelope(tmp_path, monkeypatch):
    out = tmp_path / "q4.json"
    monkeypatch.setattr(
        _worker, "measure_weight_fidelity", lambda *a, **k: _wreport("q4", 0.09, 4200)
    )
    rc = _worker.run_weight_worker(["--quant", "q4", "--reference", "ref", "--out", str(out)])
    assert rc == 0
    env = json.loads(out.read_text())
    assert env["status"] == "ok"
    assert env["report"]["quant_model_id"] == "q4"


def test_worker_writes_failed_envelope_on_domain_error(tmp_path, monkeypatch):
    from mlx_quant_fidelity.errors import ModelMismatchError

    out = tmp_path / "q2.json"

    def boom(*a, **k):
        raise ModelMismatchError("vocab_size mismatch")

    monkeypatch.setattr(_worker, "measure_weight_fidelity", boom)
    rc = _worker.run_weight_worker(["--quant", "q2", "--reference", "ref", "--out", str(out)])
    assert rc == 0  # a measurement failure is data, not a worker crash
    env = json.loads(out.read_text())
    assert env["status"] == "failed"
    assert env["error_type"] == "ModelMismatchError"
    assert "vocab_size" in env["message"]


def test_worker_isolates_non_domain_error(tmp_path, monkeypatch):
    """A non-QuantFidelityError from measure_weight_fidelity is isolated into a failed envelope.

    The worker writes a failed envelope with the exception's type name and message and returns 0
    (the same contract as a domain error), so any measurement failure is data, not a crash.
    KeyboardInterrupt / SystemExit are NOT caught — only Exception subclasses.
    """
    out = tmp_path / "q.json"

    def boom(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(_worker, "measure_weight_fidelity", boom)
    rc = _worker.run_weight_worker(["--quant", "q", "--reference", "r", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    env = json.loads(out.read_text())
    assert env["status"] == "failed"
    assert env["error_type"] == "RuntimeError"
    assert "unexpected" in env["message"]


def test_worker_ok_envelope_includes_run_identity(tmp_path, monkeypatch):
    """The worker's ok envelope must include a run_identity block with mode, quant, reference,
    max_chunks, and schema_version — so the orchestrator can validate it on resume.
    """
    from mlx_quant_fidelity.runners.compare import _PARTIAL_SCHEMA_VERSION

    out = tmp_path / "q4.json"
    monkeypatch.setattr(
        _worker, "measure_weight_fidelity", lambda *a, **k: _wreport("q4", 0.09, 4200)
    )
    _worker.run_weight_worker(
        ["--quant", "q4", "--reference", "ref", "--out", str(out), "--max-chunks", "7"]
    )
    env = json.loads(out.read_text())
    assert "run_identity" in env, "ok envelope must contain run_identity"
    identity = env["run_identity"]
    assert identity["mode"] == "weight"
    assert identity["quant"] == "q4"
    assert identity["reference"] == "ref"
    assert identity["max_chunks"] == 7
    assert identity["schema_version"] == _PARTIAL_SCHEMA_VERSION


def test_worker_failed_envelope_includes_run_identity(tmp_path, monkeypatch):
    """The worker's failed envelope must also include a run_identity block.

    This allows the orchestrator to distinguish a legitimately-failed run (same identity)
    from a stale partial from a different run.
    """
    from mlx_quant_fidelity.errors import ModelMismatchError
    from mlx_quant_fidelity.runners.compare import _PARTIAL_SCHEMA_VERSION

    out = tmp_path / "q2.json"

    def boom(*a, **k):
        raise ModelMismatchError("vocab_size mismatch")

    monkeypatch.setattr(_worker, "measure_weight_fidelity", boom)
    _worker.run_weight_worker(["--quant", "q2", "--reference", "ref-x", "--out", str(out)])
    env = json.loads(out.read_text())
    assert env["status"] == "failed"
    assert "run_identity" in env, "failed envelope must contain run_identity"
    identity = env["run_identity"]
    assert identity["quant"] == "q2"
    assert identity["reference"] == "ref-x"
    assert identity["max_chunks"] is None  # no --max-chunks arg passed
    assert identity["schema_version"] == _PARTIAL_SCHEMA_VERSION


def test_worker_forwards_quant_reference_and_max_chunks(tmp_path, monkeypatch):
    received: dict[str, object] = {}

    def fake(quant, reference, **kw):
        received["quant"] = quant
        received["reference"] = reference
        received["kw"] = kw
        return _wreport("q4", 0.09, 4200)

    monkeypatch.setattr(_worker, "measure_weight_fidelity", fake)
    out = tmp_path / "q4.json"
    rc = _worker.run_weight_worker(
        ["--quant", "q4", "--reference", "ref", "--out", str(out), "--max-chunks", "3"]
    )
    assert rc == 0
    assert received["quant"] == "q4"  # first positional is the quant repo
    assert received["reference"] == "ref"  # a quant/reference swap would fail here
    assert received["kw"]["max_chunks"] == 3  # type: ignore[index]
