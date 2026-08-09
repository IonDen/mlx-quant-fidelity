"""Measure KV-probe peak memory vs chunk length. One subprocess per length; resumable.

Each worker runs an active-memory watchdog (abort at ``memory_size - 4 GiB`` with an honest
artifact) and the orchestrator enforces a wall-clock backstop per length — required for any
heavy worker; the wired cap alone does not prevent pageable-memory paging storms (workspace
rule, amended 2026-07-11).

Feeds the ``MAX_CHUNK_LENGTH`` decision in ``src/mlx_quant_fidelity/probes/kv.py`` and the
long-window docs. Run on the MAIN THREAD only:

    uv run python scripts/spike_long_window_memory.py

Resumable: a length whose ``_artifacts/spike_long_window/<length>.json`` already exists is
skipped, so an interrupted run only loses its in-flight length.
"""

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

WORKER = """
import json, os, sys, threading, time
import mlx.core as mx

length, model_id, revision = int(sys.argv[1]), sys.argv[2], sys.argv[3]
ceiling = int(mx.device_info()["memory_size"]) - 4 * 1024**3

def _watchdog():
    while True:
        active = int(mx.get_active_memory())
        if active > ceiling:
            print(json.dumps({"chunk_length": length, "aborted": "active-memory watchdog",
                              "active_bytes": active, "ceiling_bytes": ceiling}))
            sys.stdout.flush()
            os._exit(3)
        time.sleep(0.5)

threading.Thread(target=_watchdog, daemon=True).start()

from mlx_quant_fidelity.probes.kv import measure_kv_fidelity
report = measure_kv_fidelity(model_id, kv_bits=4, kv_group_size=64,
                             chunk_length=length, max_chunks=4, model_revision=revision)
print(json.dumps({"chunk_length": length, "peak_bytes": report.peak_memory_bytes,
                  "kl_mean": report.kl.mean, "model_revision": revision,
                  "buckets": [b.kl_mean for b in report.kl_by_depth or ()]}))
"""


def _run_worker(length: int) -> tuple[int, str, str]:
    """Run one subprocess for ``length``; return (returncode, last-stdout-line, stderr tail).

    Isolated in its own function (not inlined in the orchestrator loop) so the try/except for
    the wall-clock backstop is not lexically nested inside the ``for`` loop below.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", WORKER, str(length), MODEL, MODEL_REVISION],
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


def main() -> int:
    """Measure peak memory for each length in ``LENGTHS``, skipping already-written outputs."""
    OUT.mkdir(parents=True, exist_ok=True)
    for length in LENGTHS:
        out = OUT / f"{length}.json"
        if out.exists():
            print(f"skip {length} (exists)")
            continue
        rc, stdout, stderr = _run_worker(length)
        payload = (stdout.strip().splitlines() or ["{}"])[-1]
        out.write_text(json.dumps({"returncode": rc, "result": payload, "stderr": stderr[-2000:]}))
        print(f"{length}: rc={rc} {payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
