"""Command-line interface. Installs the device-derived wired cap before any model load."""

from __future__ import annotations

import argparse
import os
import sys

from mlx_quant_fidelity._memory_caps import install_memory_caps
from mlx_quant_fidelity.errors import QuantFidelityError
from mlx_quant_fidelity.probes.kv import measure_kv_fidelity
from mlx_quant_fidelity.probes.weights import measure_weight_fidelity
from mlx_quant_fidelity.report import (
    render_comparison_json,
    render_comparison_markdown,
    render_json,
    render_markdown,
    render_weight_markdown,
)
from mlx_quant_fidelity.runners.compare import compare_kv_fidelity, compare_weight_fidelity


def _parse_kv_configs(raw: str) -> list[tuple[int, int]]:
    """Parse '4:32,4:64,8:64' -> [(4,32),(4,64),(8,64)]. Raises ValueError on a malformed entry."""
    configs = []
    for item in raw.split(","):
        bits_s, sep, gs_s = item.partition(":")
        if not sep or not bits_s.isdigit() or not gs_s.isdigit():
            raise ValueError(f"--configs entry {item!r} must be 'bits:group_size' (e.g. 4:64).")
        if int(bits_s) <= 0 or int(gs_s) <= 0:
            raise ValueError(
                f"--configs entry {item!r}: bits and group_size must be positive integers."
            )
        configs.append((int(bits_s), int(gs_s)))
    return configs


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

    weights = sub.add_parser("weights", help="measure weight-quantization fidelity")
    weights.add_argument("quant_model")
    weights.add_argument("--reference", required=True)
    weights.add_argument("--max-chunks", type=int, default=None)
    weights.add_argument("--format", choices=["json", "md"], default="md")

    compare = sub.add_parser("compare", help="rank N quantizations on a memory-normalized Pareto")
    csub = compare.add_subparsers(dest="compare_mode", required=True)

    cw = csub.add_parser("weights", help="rank N weight-quant repos vs a reference")
    cw.add_argument("quant_models", nargs="+")
    cw.add_argument("--reference", required=True)
    cw.add_argument("--max-chunks", type=int, default=None)
    cw.add_argument("--max-kld", type=float, default=None)
    cw.add_argument("--min-tier", choices=["good", "marginal", "bad"], default=None)
    cw.add_argument("--format", choices=["json", "md"], default="md")

    ck = csub.add_parser("kv", help="rank N (bits:group_size) KV configs on one model")
    ck.add_argument("model")
    ck.add_argument("--configs", required=True)
    ck.add_argument("--quantize-start", type=int, default=0)
    ck.add_argument("--max-chunks", type=int, default=None)
    ck.add_argument("--max-kld", type=float, default=None)
    ck.add_argument("--min-tier", choices=["good", "marginal", "bad"], default=None)
    ck.add_argument("--format", choices=["json", "md"], default="md")

    args = parser.parse_args(argv)
    try:
        if args.command == "kv":
            report = measure_kv_fidelity(
                args.model,
                kv_bits=args.kv_bits,
                kv_group_size=args.kv_group_size,
                quantize_start=args.quantize_start,
                max_chunks=args.max_chunks,
            )
            out = render_json(report) if args.format == "json" else render_markdown(report)
        elif args.command == "weights":
            wreport = measure_weight_fidelity(
                args.quant_model,
                args.reference,
                max_chunks=args.max_chunks,
            )
            out = render_json(wreport) if args.format == "json" else render_weight_markdown(wreport)
        elif args.command == "compare" and args.compare_mode == "weights":
            creport = compare_weight_fidelity(
                args.quant_models,
                args.reference,
                max_chunks=args.max_chunks,
                max_kld=args.max_kld,
                min_tier=args.min_tier,
            )
            out = (
                render_comparison_json(creport)
                if args.format == "json"
                else render_comparison_markdown(creport)
            )
        elif args.compare_mode == "kv":
            try:
                configs = _parse_kv_configs(args.configs)
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            creport = compare_kv_fidelity(
                args.model,
                configs,
                quantize_start=args.quantize_start,
                max_chunks=args.max_chunks,
                max_kld=args.max_kld,
                min_tier=args.min_tier,
            )
            out = (
                render_comparison_json(creport)
                if args.format == "json"
                else render_comparison_markdown(creport)
            )
        else:
            raise AssertionError(  # pragma: no cover
                f"unhandled compare_mode: {args.compare_mode!r}"
            )
    except QuantFidelityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(out)
    return 0


def _console_entry() -> None:  # pragma: no cover - process-exit wrapper
    """Console-script entry point.

    Runs :func:`main`, flushes output, then hard-exits via ``os._exit`` to skip MLX's
    Metal backend C++ destructor, which segfaults at interpreter shutdown on Apple
    Silicon ("Python quit unexpectedly"). The ``finally`` guarantees the hard exit on
    every path — including an unexpected error from ``main`` — so the teardown segfault
    cannot leak through on an error.
    """
    code = 1
    try:
        code = main()
    except SystemExit as exc:  # argparse usage errors, etc.
        code = exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:  # last resort: still hard-exit rather than crash in teardown
        print(f"internal error: {exc}", file=sys.stderr)
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


if __name__ == "__main__":  # pragma: no cover
    _console_entry()
