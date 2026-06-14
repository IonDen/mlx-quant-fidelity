"""Command-line interface. Installs the device-derived wired cap before any model load."""

from __future__ import annotations

import argparse
import sys

from mlx_quant_fidelity._memory_caps import install_memory_caps
from mlx_quant_fidelity.probes.kv import measure_kv_fidelity
from mlx_quant_fidelity.report import render_json, render_markdown


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `mlx-quant-fidelity` console script."""
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
    if args.command == "kv":
        report = measure_kv_fidelity(
            args.model,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            quantize_start=args.quantize_start,
            max_chunks=args.max_chunks,
        )
        out = render_json(report) if args.format == "json" else render_markdown(report)
        print(out)
        return 0
    return 2  # pragma: no cover - argparse enforces a subcommand


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
