"""Measure KV-probe peak memory vs chunk length, per method lane.

One subprocess per length; resumable per lane. Each worker runs an active-memory watchdog (abort at ``memory_size - 4 GiB`` with an honest
artifact) and the orchestrator enforces a wall-clock backstop per length — required for any
heavy worker; the wired cap alone does not prevent pageable-memory paging storms (workspace
rule, amended 2026-07-11).

Feeds the ``MAX_CHUNK_LENGTH`` decision in ``src/mlx_quant_fidelity/probes/kv.py`` and the
long-window docs. Run on the MAIN THREAD only, one lane at a time (never in parallel — heavy
runs are serialized):

    uv run --no-sync python scripts/spike_long_window_memory.py --method stock
    uv run --no-sync python scripts/spike_long_window_memory.py --method turboquant
    uv run --no-sync python scripts/spike_long_window_memory.py --method turboquant-vonly
    uv run --no-sync python scripts/spike_long_window_memory.py --method affine
    uv run --no-sync python scripts/spike_long_window_memory.py --method stock-control

``--no-sync`` is required for every lane, including ``affine`` and ``stock-control`` (which
never import the TurboQuant-MLX port themselves): a bare ``uv run`` resyncs the venv to the
pinned lockfile and silently REMOVES the git-pinned port, so a later turboquant/turboquant-vonly
lane would fail with it missing.

Resumable per lane: a length whose artifact (``_artifacts/spike_long_window/<length>.json`` for
the ``stock`` lane, ``_artifacts/spike_long_window/<lane>/<length>.json`` for every other lane —
see :func:`_lane_out_dir`) records a successful prior run (``returncode == 0``) is skipped, so an
interrupted run only loses its in-flight length. An artifact recording a watchdog abort, a
wall-clock timeout, or that is missing/corrupt does NOT count as done -- that length is re-run
and its artifact overwritten. Lane directories are isolated from each other and from the legacy
flat path, so a completed run of one lane never marks another lane's length as done.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
# Current `main` commit SHA, resolved 2026-08-09 (`hf repo info mlx-community/Llama-3.2-1B-Instruct-4bit`).
MODEL_REVISION = "08231374eeacb049a0eade7922910865b8fce912"
LENGTHS = (512, 1024, 2048, 4096)
WALL_TIMEOUT_S = 900  # generous: the 4096 lane at 4 chunks is minutes, not a quarter hour
OUT = Path("_artifacts/spike_long_window")

# method: (parse_method_spec spec string, control=). "stock" is the pre-0.7.0 legacy lane;
# "stock-control" reuses the stock method but also runs the quantizer-only control forward.
LANE_SPECS: dict[str, tuple[str, bool]] = {
    "stock": ("4:64", False),
    "turboquant": ("turboquant:4", False),
    "turboquant-vonly": ("turboquant-vonly:4", False),
    "affine": ("affine:8:4", False),
    "stock-control": ("4:64", True),
}

WORKER = """
import json, os, sys, threading, time
import mlx.core as mx

length, model_id, revision, method_spec, control = (
    int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] == "1"
)
ceiling = int(mx.device_info()["memory_size"]) - 4 * 1024**3

# MLX does not return a dropped buffer to the OS -- it moves to the allocator's retained cache
# pool, which get_active_memory() does NOT count. Bound that pool so the watchdog's headroom is
# real, and measure against active + cache (the actual resident footprint).
mx.set_cache_limit(4 * 1024**3)

def _watchdog():
    while True:
        active, cached = int(mx.get_active_memory()), int(mx.get_cache_memory())
        if active + cached > ceiling:
            print(json.dumps({"chunk_length": length, "aborted": "active-memory watchdog",
                              "active_bytes": active, "cache_bytes": cached,
                              "resident_bytes": active + cached, "ceiling_bytes": ceiling}))
            sys.stdout.flush()
            os._exit(3)
        # 0.05s, not 0.5s: the poll thread runs while mx.eval holds no GIL, so it is cheap, and
        # a 0.5s window is long enough for a single large allocation to overshoot the ceiling.
        time.sleep(0.05)

threading.Thread(target=_watchdog, daemon=True).start()

from mlx_quant_fidelity.probes.kv import measure_kv_fidelity
from mlx_quant_fidelity.probes.kv_methods import parse_method_spec
method = parse_method_spec(method_spec)
report = measure_kv_fidelity(model_id, method=method, chunk_length=length, max_chunks=4,
                             model_revision=revision, control=control)
print(json.dumps({"chunk_length": length, "peak_bytes": report.peak_memory_bytes,
                  "kl_mean": report.kl.mean, "model_revision": revision,
                  "buckets": [b.kl_mean for b in report.kl_by_depth or ()]}))
"""


def _run_worker(length: int, method_spec: str, control: bool) -> tuple[int, str, str]:
    """Run one subprocess for ``length`` on the given method lane.

    Returns (returncode, last-stdout-line, stderr tail). Isolated in its own function (not
    inlined in the orchestrator loop) so the try/except for the wall-clock backstop is not
    lexically nested inside the ``for`` loop below.
    """
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                WORKER,
                str(length),
                MODEL,
                MODEL_REVISION,
                method_spec,
                "1" if control else "0",
            ],
            capture_output=True,
            text=True,
            timeout=WALL_TIMEOUT_S,
        )
        stdout = result.stdout
        stderr = result.stderr
        return result.returncode, stdout, stderr
    except subprocess.TimeoutExpired as exc:
        stdout = json.dumps(
            {"chunk_length": length, "aborted": f"wall-clock backstop ({WALL_TIMEOUT_S}s)"}
        )
        raw_stderr = exc.stderr
        stderr = (
            raw_stderr.decode(errors="replace")
            if isinstance(raw_stderr, bytes)
            else (raw_stderr or "")
        )
        return -1, stdout, stderr


def _recorded_returncode(path: Path) -> int | None:
    """Returncode recorded in an existing artifact, or None if missing, corrupt, or unreadable.

    Guards the resume read: any failure to parse the artifact as JSON, or a payload that isn't
    a dict with an integer ``returncode``, is treated as "not a recorded success" -- the caller
    re-runs rather than trusting a file it can't confidently read.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    rc = payload.get("returncode") if isinstance(payload, dict) else None
    return rc if isinstance(rc, int) else None


def _lane_is_complete(path: Path) -> bool:
    """True only when ``path`` records a successful (``returncode == 0``) prior run.

    A watchdog abort (rc 3), a wall-clock timeout (rc -1), or a missing/corrupt artifact must
    NOT count as complete -- skipping those silently would treat a failed lane as done forever.
    """
    return _recorded_returncode(path) == 0


def _lane_out_dir(method: str, *, base: Path = OUT) -> Path:
    """Artifact directory for ``method``: ``base`` itself for ``stock``, ``base/<method>`` otherwise.

    Keeping ``stock`` flat means the committed 2026-08-09 baseline record
    (``_artifacts/spike_long_window/<length>.json``) is not orphaned by the method-lane split.
    Every other lane (including ``stock-control``) gets its own subdirectory keyed by its lane
    name, so a completed length in one lane never satisfies another lane's resume check.
    """
    return base if method == "stock" else base / method


def main() -> int:
    """Measure peak memory for each length in ``LENGTHS`` on the selected ``--method`` lane.

    Resumes by skipping lengths whose lane-local artifact already records success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=sorted(LANE_SPECS),
        default="stock",
        help="method lane to measure (default: stock, the pre-0.7.0 legacy lane)",
    )
    args = parser.parse_args()
    method_spec, control = LANE_SPECS[args.method]
    lane_dir = _lane_out_dir(args.method)
    lane_dir.mkdir(parents=True, exist_ok=True)
    for length in LENGTHS:
        out = lane_dir / f"{length}.json"
        recorded_rc = _recorded_returncode(
            out
        )  # single read; _lane_is_complete(out) == (recorded_rc == 0)
        if recorded_rc == 0:
            print(f"skip {args.method}/{length} (exists, rc=0)")
            continue
        if out.exists():
            print(
                f"{args.method}/{length}: existing artifact recorded rc={recorded_rc!r} "
                "(not a success) -> re-running"
            )
        rc, stdout, stderr = _run_worker(length, method_spec, control)
        payload = (stdout.strip().splitlines() or ["{}"])[-1]
        out.write_text(json.dumps({"returncode": rc, "result": payload, "stderr": stderr[-2000:]}))
        print(f"{args.method}/{length}: rc={rc} {payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
