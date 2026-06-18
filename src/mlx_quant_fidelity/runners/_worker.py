"""Subprocess worker: run ONE weight-fidelity measurement and write a JSON envelope.

Isolation per target keeps MLX's lazy allocator from accumulating two models across N targets
(the subprocess-per-condition rule). The orchestrator spawns this and reads the envelope.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from mlx_quant_fidelity._memory_caps import install_memory_caps
from mlx_quant_fidelity.errors import QuantFidelityError
from mlx_quant_fidelity.probes.weights import measure_weight_fidelity


def run_weight_worker(argv: list[str] | None = None) -> int:
    """Measure one (quant, reference) pair; write {status, report|error} to --out. Returns 0."""
    install_memory_caps()  # before any model load
    parser = argparse.ArgumentParser(prog="mlx-quant-fidelity-worker")
    parser.add_argument("--quant", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-chunks", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        report = measure_weight_fidelity(args.quant, args.reference, max_chunks=args.max_chunks)
        envelope: dict[str, object] = {"status": "ok", "report": dataclasses.asdict(report)}
    except QuantFidelityError as exc:
        envelope = {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)}
    Path(args.out).write_text(json.dumps(envelope))
    return 0


def _console_entry() -> None:  # pragma: no cover - process-exit wrapper
    import os

    code = 1
    try:
        code = run_weight_worker()
    except Exception as exc:  # last resort, still write nothing and hard-exit
        print(f"worker error: {exc}", file=sys.stderr)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


if __name__ == "__main__":  # pragma: no cover
    _console_entry()
