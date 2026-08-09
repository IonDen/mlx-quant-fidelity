"""Offline unit test for the long-window spike script's resume predicate.

``scripts/spike_long_window_memory.py`` is not part of the installed package and carries no
coverage requirement, but its resume decision (skip vs re-run a length whose artifact already
exists) is a pure function worth pinning directly -- a prior version skipped ANY existing
artifact, including one recording a watchdog abort or wall-clock timeout, silently treating a
failed lane as done forever. No MLX import, no model, no network: importing the script module
only pulls in stdlib.
"""

import json

from scripts.spike_long_window_memory import _lane_is_complete


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
