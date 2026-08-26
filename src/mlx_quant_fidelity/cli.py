"""Command-line interface. Installs the device-derived wired cap before any model load."""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

from mlx_quant_fidelity._memory_caps import install_memory_caps
from mlx_quant_fidelity.badge import render_badge_markdown
from mlx_quant_fidelity.errors import QuantFidelityError
from mlx_quant_fidelity.probes.kv import measure_kv_fidelity
from mlx_quant_fidelity.probes.kv_methods import (
    METHODS,
    TURBOQUANT_DEFAULT_SEED,
    StockKVMethod,
    TurboQuantKVMethod,
    TurboQuantVOnlyKVMethod,
    parse_method_spec,
)
from mlx_quant_fidelity.probes.weights import measure_weight_fidelity
from mlx_quant_fidelity.report import (
    render_comparison_json,
    render_comparison_markdown,
    render_json,
    render_markdown,
    render_weight_markdown,
)
from mlx_quant_fidelity.runners.compare import (
    compare_kv_fidelity,
    compare_weight_fidelity,
    filter_configs_by_kv_budget,
    generate_sweep_configs,
    kv_geometry_from_config,
)

if TYPE_CHECKING:
    from mlx_quant_fidelity.probes.kv_methods import KVCacheMethod


def _fetch_model_config(model_id: str) -> dict[str, object]:  # pragma: no cover - network
    """Fetch just ``config.json`` from a HuggingFace repo — no weight download."""
    import json as _json
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    with Path(hf_hub_download(model_id, "config.json")).open() as f:
        return _json.load(f)  # type: ignore[no-any-return]


def _parse_kv_configs(raw: str) -> list[KVCacheMethod]:
    """Parse '4:32,turboquant:3' -> [StockKVMethod(4,32), TurboQuantKVMethod(3)].

    Raises ValueError (via CompareConfigError) on a malformed entry.
    """
    return [parse_method_spec(item) for item in raw.split(",")]


def _resolve_kv_method(args: argparse.Namespace) -> KVCacheMethod:
    """Build the method from ``--kv-method`` (a name or a spec string) and the kv flags.

    A value containing ``:`` is a spec string (``'turboquant:4:7'``, ``'affine:8:4'``,
    a bare ``'4:64'``) parsed via :func:`parse_method_spec`; it may not be combined with
    any of ``--kv-bits``/``--kv-group-size``/``--kv-seed``. A bare name uses those flags
    directly, applying the 0.5.x legacy defaults when a flag is omitted (stock: bits 4,
    group_size 64; turboquant: bits 4, seed ``TURBOQUANT_DEFAULT_SEED``). ``affine`` has
    no flag-based form (it needs both k_bits and v_bits) and always errors, pointing at
    the spec grammar. ``turboquant-vonly`` reuses ``--kv-bits`` as its ``v_bits`` and
    requires it explicitly (there is no sensible default to guess). An unrecognized name
    raises, listing the known ``METHODS``.
    """
    value = args.kv_method
    if ":" in value:
        if args.kv_bits is not None or args.kv_group_size is not None or args.kv_seed is not None:
            raise ValueError(
                "--kv-method with a spec string replaces --kv-bits/--kv-group-size/--kv-seed"
            )
        return parse_method_spec(value)
    if value == "stock":
        if args.kv_seed is not None:
            raise ValueError("--kv-seed is only valid with --kv-method turboquant")
        bits = 4 if args.kv_bits is None else args.kv_bits
        gs = 64 if args.kv_group_size is None else args.kv_group_size
        return StockKVMethod(bits=bits, group_size=gs)
    if value == "turboquant":
        if args.kv_group_size is not None:
            raise ValueError("--kv-group-size is only valid with --kv-method stock")
        bits = 4 if args.kv_bits is None else args.kv_bits
        seed = TURBOQUANT_DEFAULT_SEED if args.kv_seed is None else args.kv_seed
        return TurboQuantKVMethod(bits=bits, seed=seed)
    if value == "affine":
        raise ValueError(
            "--kv-method affine has no flag-based form (it needs both k_bits and v_bits); "
            "use a spec string, e.g. --kv-method affine:8:4"
        )
    if value == "turboquant-vonly":
        if args.kv_bits is None:
            raise ValueError("--kv-method turboquant-vonly requires --kv-bits (used as v_bits)")
        seed = TURBOQUANT_DEFAULT_SEED if args.kv_seed is None else args.kv_seed
        return TurboQuantVOnlyKVMethod(v_bits=args.kv_bits, seed=seed)
    raise ValueError(f"unknown --kv-method {value!r}; known: {sorted(METHODS)}")


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
    kv.add_argument("--kv-bits", type=int, default=None)
    kv.add_argument(
        "--kv-method",
        default="stock",
        metavar="METHOD",
        help=(
            "cache method: a name (stock, turboquant, affine, turboquant-vonly) combined "
            "with --kv-bits/--kv-group-size/--kv-seed, or a self-contained spec string "
            "(e.g. 'turboquant:4:7', 'affine:8:4', 'turboquant-vonly:3') that replaces "
            "those flags (default stock)"
        ),
    )
    kv.add_argument("--kv-group-size", type=int, default=None)
    kv.add_argument("--kv-seed", type=int, default=None)
    kv.add_argument("--quantize-start", type=int, default=0)
    kv.add_argument("--max-chunks", type=int, default=None)
    kv.add_argument("--chunk-length", type=int, default=512)
    kv.add_argument("--model-revision", default=None)
    kv.add_argument(
        "--control",
        action="store_true",
        help="also run the stock method's quantizer-only control lane (see --kv-method stock)",
    )
    kv.add_argument("--format", choices=["json", "md", "badge"], default="md")

    weights = sub.add_parser("weights", help="measure weight-quantization fidelity")
    weights.add_argument("quant_model")
    weights.add_argument("--reference", required=True)
    weights.add_argument("--max-chunks", type=int, default=None)
    weights.add_argument("--format", choices=["json", "md", "badge"], default="md")

    compare = sub.add_parser("compare", help="rank N quantizations on a memory-normalized Pareto")
    csub = compare.add_subparsers(dest="compare_mode", required=True)

    cw = csub.add_parser("weights", help="rank N weight-quant repos vs a reference")
    cw.add_argument("quant_models", nargs="+")
    cw.add_argument("--reference", required=True)
    cw.add_argument("--max-chunks", type=int, default=None)
    cw.add_argument("--max-kld", type=float, default=None)
    cw.add_argument("--min-tier", choices=["good", "marginal", "bad"], default=None)
    cw.add_argument("--quant-revision", default=None)
    cw.add_argument("--reference-revision", default=None)
    cw.add_argument("--format", choices=["json", "md"], default="md")

    ck = csub.add_parser("kv", help="rank N (bits:group_size) KV configs on one model")
    ck.add_argument("model")
    ck.add_argument(
        "--configs",
        default=None,
        help="e.g. '4:32,4:64,8:64,turboquant:4' (methods: bits:group_size = stock; "
        "turboquant:bits[:seed], seed >= 1)",
    )
    ck.add_argument(
        "--sweep",
        action="store_true",
        help="auto-generate the config grid from the model's config.json (mutually "
        "exclusive with --configs)",
    )
    ck.add_argument(
        "--max-kv-bytes-per-token",
        type=int,
        default=None,
        help="--sweep only: drop configs whose KV bytes/token exceed this budget",
    )
    ck.add_argument("--quantize-start", type=int, default=0)
    ck.add_argument("--max-chunks", type=int, default=None)
    ck.add_argument("--chunk-length", type=int, default=512)
    ck.add_argument("--model-revision", default=None)
    ck.add_argument("--max-kld", type=float, default=None)
    ck.add_argument("--min-tier", choices=["good", "marginal", "bad"], default=None)
    ck.add_argument("--format", choices=["json", "md"], default="md")

    args = parser.parse_args(argv)
    try:
        if args.command == "kv":
            try:
                method = _resolve_kv_method(args)
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            report = measure_kv_fidelity(
                args.model,
                method=method,
                quantize_start=args.quantize_start,
                max_chunks=args.max_chunks,
                chunk_length=args.chunk_length,
                model_revision=args.model_revision,
                control=args.control,
            )
            if args.format == "json":
                out = render_json(report)
            elif args.format == "badge":
                out = render_badge_markdown(report)
            else:
                out = render_markdown(report)
        elif args.command == "weights":
            wreport = measure_weight_fidelity(
                args.quant_model,
                args.reference,
                max_chunks=args.max_chunks,
            )
            if args.format == "json":
                out = render_json(wreport)
            elif args.format == "badge":
                out = render_badge_markdown(wreport)
            else:
                out = render_weight_markdown(wreport)
        elif args.command == "compare" and args.compare_mode == "weights":
            creport = compare_weight_fidelity(
                args.quant_models,
                args.reference,
                max_chunks=args.max_chunks,
                max_kld=args.max_kld,
                min_tier=args.min_tier,
                quant_revision=args.quant_revision,
                reference_revision=args.reference_revision,
            )
            out = (
                render_comparison_json(creport)
                if args.format == "json"
                else render_comparison_markdown(creport)
            )
        elif args.compare_mode == "kv":
            if bool(args.configs) == bool(args.sweep):
                print("error: exactly one of --configs or --sweep is required", file=sys.stderr)
                return 2
            if args.max_kv_bytes_per_token is not None and not args.sweep:
                print(
                    "error: --max-kv-bytes-per-token is only valid with --sweep",
                    file=sys.stderr,
                )
                return 2
            skipped_configs: list[tuple[str, str]] = []
            configs: list[KVCacheMethod]
            if args.sweep:
                config_json = _fetch_model_config(args.model)
                n_layers, n_kv_heads, head_dim = kv_geometry_from_config(config_json)
                if head_dim is None:
                    print(
                        "error: cannot derive head_dim from config.json; pass --configs explicitly",
                        file=sys.stderr,
                    )
                    return 2
                grid, sweep_skipped = generate_sweep_configs(head_dim)
                if args.max_kv_bytes_per_token is not None:
                    if n_layers is None or n_kv_heads is None:
                        print(
                            "error: cannot apply --max-kv-bytes-per-token: config.json is "
                            "missing num_hidden_layers/num_key_value_heads",
                            file=sys.stderr,
                        )
                        return 2
                    kept, budget_skipped = filter_configs_by_kv_budget(
                        grid,
                        n_layers=n_layers,
                        n_kv_heads=n_kv_heads,
                        head_dim=head_dim,
                        max_kv_bytes_per_token=args.max_kv_bytes_per_token,
                    )
                else:
                    kept, budget_skipped = grid, []
                if len(kept) < 2:
                    print(
                        "error: fewer than 2 configs remain after "
                        f"--max-kv-bytes-per-token={args.max_kv_bytes_per_token}",
                        file=sys.stderr,
                    )
                    return 2
                configs = [StockKVMethod(bits=b, group_size=g) for b, g in kept]
                skipped_configs = sweep_skipped + budget_skipped
            else:
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
                chunk_length=args.chunk_length,
                model_revision=args.model_revision,
                max_kld=args.max_kld,
                min_tier=args.min_tier,
                skipped_configs=skipped_configs,
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
