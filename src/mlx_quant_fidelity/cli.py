"""Command-line interface. Installs the device-derived wired cap before any model load."""

from __future__ import annotations

import argparse
import os
import sys

from mlx_quant_fidelity._memory_caps import install_memory_caps
from mlx_quant_fidelity.errors import QuantFidelityError
from mlx_quant_fidelity.probes.kv import measure_kv_fidelity
from mlx_quant_fidelity.report import render_json, render_markdown


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code.

    Kept free of ``os._exit`` so it stays unit-testable; the console-script wrapper
    (:func:`_console_entry`) performs the hard exit that skips MLX's Metal teardown.
    """
    install_memory_caps()  # first action, before importing/loading any model

    parser = argparse.ArgumentParser(prog="mlx-quant-fidelity")
    sub = parser.add_subparsers(dest="command", required=True)
    kv = sub.add_parser("kv", help="measure KV-cache quantization fidelity")
    kv.add_argument("model")
    kv.add_argument("--kv-bits", type=int, default=4)
    kv.add_argument("--kv-group-size", type=int, default=64)
    kv.add_argument("--quantize-start", type=int, default=0)
    kv.add_argument("--max-chunks", type=int, default=None)
    kv.add_argument("--format", choices=["json", "md"], default="md")

    args = parser.parse_args(argv)
    try:
        report = measure_kv_fidelity(
            args.model,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            quantize_start=args.quantize_start,
            max_chunks=args.max_chunks,
        )
    except QuantFidelityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = render_json(report) if args.format == "json" else render_markdown(report)
    print(out)
    return 0


def _console_entry() -> None:  # pragma: no cover - process-exit wrapper
    """Console-script entry point.

    Runs :func:`main`, flushes output, then hard-exits via ``os._exit`` to skip MLX's
    Metal backend C++ destructor, which segfaults at interpreter shutdown on Apple
    Silicon ("Python quit unexpectedly"). The work and output are already complete.
    """
    try:
        code = main()
    except SystemExit as exc:  # argparse usage errors, etc.
        code = exc.code if isinstance(exc.code, int) else 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":  # pragma: no cover
    _console_entry()
