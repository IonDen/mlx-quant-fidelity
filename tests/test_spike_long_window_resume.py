"""Offline unit test for the long-window spike script's resume predicate.

``scripts/spike_long_window_memory.py`` is not part of the installed package and carries no
coverage requirement, but its resume decision (skip vs re-run a length whose artifact already
exists) is a pure function worth pinning directly -- a prior version skipped ANY existing
artifact, including one recording a watchdog abort or wall-clock timeout, silently treating a
failed lane as done forever. No MLX import, no model, no network: importing the script module
only pulls in stdlib.
"""

import json

from scripts.spike_long_window_memory import _lane_is_complete, _lane_out_dir


def test_missing_artifact_is_not_complete(tmp_path):
    assert _lane_is_complete(tmp_path / "512.json") is False


def test_success_artifact_is_complete(tmp_path):
    out = tmp_path / "512.json"
    out.write_text(json.dumps({"returncode": 0, "result": "{}", "stderr": ""}))
    assert _lane_is_complete(out) is True


def test_watchdog_abort_artifact_is_not_complete(tmp_path):
    # returncode 3 is the WORKER's os._exit(3) on the active-memory watchdog firing.
    out = tmp_path / "4096.json"
    out.write_text(json.dumps({"returncode": 3, "result": "{}", "stderr": ""}))
    assert _lane_is_complete(out) is False


def test_wall_timeout_artifact_is_not_complete(tmp_path):
    # returncode -1 is what _run_worker records for the wall-clock backstop.
    out = tmp_path / "4096.json"
    out.write_text(json.dumps({"returncode": -1, "result": "{}", "stderr": ""}))
    assert _lane_is_complete(out) is False


def test_corrupt_artifact_is_not_complete(tmp_path):
    out = tmp_path / "4096.json"
    out.write_text("{not valid json")
    assert _lane_is_complete(out) is False


def test_artifact_missing_returncode_field_is_not_complete(tmp_path):
    out = tmp_path / "4096.json"
    out.write_text(json.dumps({"result": "{}"}))
    assert _lane_is_complete(out) is False


def test_lane_out_dir_stock_is_the_flat_base(tmp_path):
    # the stock lane keeps the legacy flat path so the committed 2026-08-09 baseline record
    # (<base>/<length>.json) is not orphaned by the method-lane split.
    assert _lane_out_dir("stock", base=tmp_path) == tmp_path


def test_lane_out_dir_nonstock_is_a_method_subdirectory(tmp_path):
    assert _lane_out_dir("turboquant", base=tmp_path) == tmp_path / "turboquant"
    assert _lane_out_dir("turboquant-vonly", base=tmp_path) == tmp_path / "turboquant-vonly"
    assert _lane_out_dir("affine", base=tmp_path) == tmp_path / "affine"
    assert _lane_out_dir("stock-control", base=tmp_path) == tmp_path / "stock-control"


def test_completed_turboquant_lane_does_not_mark_affine_lane_complete(tmp_path):
    # a completed turboquant/512.json must NOT cause affine/512.json to be skipped -- lane
    # directories isolate resume state per method, they don't share one flat namespace.
    tq_dir = _lane_out_dir("turboquant", base=tmp_path)
    tq_dir.mkdir(parents=True)
    (tq_dir / "512.json").write_text(json.dumps({"returncode": 0, "result": "{}", "stderr": ""}))
    affine_out = _lane_out_dir("affine", base=tmp_path) / "512.json"
    assert _lane_is_complete(affine_out) is False


def test_flat_legacy_artifact_does_not_satisfy_a_lane_resume_check(tmp_path):
    # a flat legacy 512.json (the stock lane's own path) must NOT satisfy a non-stock lane's
    # resume check just because it lives at the same base directory.
    (tmp_path / "512.json").write_text(json.dumps({"returncode": 0, "result": "{}", "stderr": ""}))
    lane_out = _lane_out_dir("turboquant", base=tmp_path) / "512.json"
    assert _lane_is_complete(lane_out) is False
