"""Command-line interface. Installs the device-derived wired cap before any model load."""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import TYPE_CHECKING

from mlx_quant_fidelity._memory_caps import install_memory_caps
from mlx_quant_fidelity.badge import render_badge_markdown
from mlx_quant_fidelity.errors import CompareConfigError, QuantFidelityError
from mlx_quant_fidelity.probes._preload import read_model_config
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
    MAX_WORKER_TIMEOUT_S,
    compare_kv_fidelity,
    compare_weight_fidelity,
    filter_configs_by_kv_budget,
    generate_sweep_configs,
    kv_geometry_from_config,
    split_target,
)

if TYPE_CHECKING:
    from mlx_quant_fidelity.probes.kv_methods import KVCacheMethod
    from mlx_quant_fidelity.report import ComparisonReport


def _parse_kv_configs(raw: str) -> list[KVCacheMethod]:
    """Parse '4:32,turboquant:3' -> [StockKVMethod(4,32), TurboQuantKVMethod(3)].

    Raises ValueError (via CompareConfigError) on a malformed entry.
    """
    methods: list[KVCacheMethod] = []
    for item in raw.split(","):
        try:
            methods.append(parse_method_spec(item))
        except ValueError as exc:
            raise CompareConfigError(f"--configs entry: {exc}") from exc
    return methods


def _resolve_kv_method(args: argparse.Namespace) -> KVCacheMethod:
    """Build the method from ``--kv-method`` (a name or a spec string) and the kv flags.

    A value containing ``:`` is a spec string (``'turboquant:4:7'``, ``'affine:8:4'``,
    a bare ``'4:64'``) parsed via :func:`parse_method_spec`; it may not be combined with
    any of ``--kv-bits``/``--kv-group-size``/``--kv-seed``. A bare name uses those flags
    directly, applying the 0.5.x legacy defaults when a flag is omitted (stock: bits 4,
    group_size 64; turboquant: bits 4, seed ``TURBOQUANT_DEFAULT_SEED``). ``affine`` has
    no flag-based form (it needs both k_bits and v_bits) and always errors, pointing at
    the spec grammar. ``turboquant-vonly`` reuses ``--kv-bits`` as its ``v_bits`` and
    requires it explicitly (there is no sensible default to guess); it has no group_size
    parameter, so a simultaneous ``--kv-group-size`` errors the same way it does for
    ``turboquant``. An unrecognized name raises, listing the known ``METHODS``.
    """
    value = args.kv_method
    if ":" in value:
        if args.kv_bits is not None or args.kv_group_size is not None or args.kv_seed is not None:
            raise ValueError(
                "--kv-method with a spec string replaces --kv-bits/--kv-group-size/--kv-seed"
            )
        try:
            return parse_method_spec(value)
        except CompareConfigError as exc:
            raise CompareConfigError(f"--kv-method: {exc}") from exc
    if value == "stock":
        if args.kv_seed is not None:
            raise ValueError("--kv-seed is only valid with --kv-method turboquant")
        bits = 4 if args.kv_bits is None else args.kv_bits
        gs = 64 if args.kv_group_size is None else args.kv_group_size
        try:
            return StockKVMethod(bits=bits, group_size=gs)
        except CompareConfigError as exc:  # only the group size is validated at construction
            raise CompareConfigError(f"--kv-group-size: {exc}") from exc
    if value == "turboquant":
        if args.kv_group_size is not None:
            raise ValueError("--kv-group-size is only valid with --kv-method stock")
        bits = 4 if args.kv_bits is None else args.kv_bits
        seed = TURBOQUANT_DEFAULT_SEED if args.kv_seed is None else args.kv_seed
        try:
            return TurboQuantKVMethod(bits=bits, seed=seed)
        except CompareConfigError as exc:
            raise CompareConfigError(f"--kv-method: {exc}") from exc
    if value == "affine":
        raise ValueError(
            "--kv-method affine has no flag-based form (it needs both k_bits and v_bits); "
            "use a spec string, e.g. --kv-method affine:8:4"
        )
    if value == "turboquant-vonly":
        if args.kv_group_size is not None:
            raise ValueError("--kv-group-size is only valid with --kv-method stock")
        if args.kv_bits is None:
            raise ValueError("--kv-method turboquant-vonly requires --kv-bits (used as v_bits)")
        seed = TURBOQUANT_DEFAULT_SEED if args.kv_seed is None else args.kv_seed
        try:
            return TurboQuantVOnlyKVMethod(v_bits=args.kv_bits, seed=seed)
        except CompareConfigError as exc:
            raise CompareConfigError(f"--kv-method: {exc}") from exc
    raise ValueError(f"--kv-method: unknown method {value!r}; known: {sorted(METHODS)}")


def _stderr_progress(message: str) -> None:
    """CLI progress sink: stderr only, so stdout stays the pure report."""
    print(message, file=sys.stderr, flush=True)


def _non_negative_seconds(value: str) -> float:
    """Argparse type for --worker-timeout: a finite, non-negative number of seconds."""
    seconds = float(value)
    if not (math.isfinite(seconds) and seconds >= 0):  # rejects NaN and inf too
        raise argparse.ArgumentTypeError(f"must be >= 0 (0 disables the limit), got {value}")
    if seconds > MAX_WORKER_TIMEOUT_S:
        # macOS subprocess.run(timeout=...) overflows above ~2,147,483 s.
        raise argparse.ArgumentTypeError(
            f"must be <= {MAX_WORKER_TIMEOUT_S:,.0f} seconds (use 0 to disable the limit), got {value}"
        )
    return seconds


def _user_error_types() -> tuple[type[BaseException], ...]:
    """Exception types that mean "the model could not be found or read", not a library bug.

    Lazy-imports huggingface_hub so importing this module stays cheap.
    """
    from huggingface_hub import errors as hub_errors

    types: list[type[BaseException]] = [
        hub_errors.RepositoryNotFoundError,
        hub_errors.GatedRepoError,
        hub_errors.RevisionNotFoundError,
        hub_errors.EntryNotFoundError,
        hub_errors.LocalEntryNotFoundError,
        hub_errors.HFValidationError,
        hub_errors.HfHubHTTPError,
        FileNotFoundError,
    ]
    return tuple(types)


def _all_rows_failed(report: ComparisonReport) -> bool:
    """True when a comparison produced rows and none of them measured successfully."""
    return bool(report.results) and not any(r.status == "ok" for r in report.results)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code.

    Kept free of ``os._exit`` so it stays unit-testable; the console-script wrapper
    (:func:`_console_entry`) performs the hard exit that skips MLX's Metal teardown.
    """
    install_memory_caps()  # first action, before importing/loading any model

    max_chunks_help = "score at most this many corpus chunks (default: the whole corpus)"
    chunk_length_help = "tokens per scored chunk (default 512; hard ceiling 4096)"
    quantize_start_help = (
        "keep the first N tokens full precision (deployment mode); 0 quantizes from token 0 "
        "(stress mode, the default)"
    )
    model_revision_help = "Hub revision (branch, tag or commit sha) of the model to load"
    max_kld_help = "KL-divergence budget: the recommended pick must not exceed it"
    min_tier_help = "minimum verdict tier the recommended pick must reach"
    format_help = "output format (default md)"
    quant_revision_help = "Hub revision applied to every quant target without an inline @revision"
    reference_revision_help = (
        "Hub revision of the reference repo, unless the reference carries an inline @revision"
    )

    parser = argparse.ArgumentParser(prog="mlx-quant-fidelity")
    sub = parser.add_subparsers(dest="command", required=True)
    kv = sub.add_parser("kv", help="measure KV-cache quantization fidelity")
    kv.add_argument("model", help="model repo id or local path")
    kv.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        help="KV-cache bit width for a named method (stock/turboquant default 4; required "
        "by turboquant-vonly, where it is the V bit width)",
    )
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
    kv.add_argument(
        "--kv-group-size",
        type=int,
        default=None,
        help="quantization group size, --kv-method stock only (default 64)",
    )
    kv.add_argument(
        "--kv-seed",
        type=int,
        default=None,
        help="rotation seed (>= 1) for the turboquant methods (default: the port's default)",
    )
    kv.add_argument("--quantize-start", type=int, default=0, help=quantize_start_help)
    kv.add_argument("--max-chunks", type=int, default=None, help=max_chunks_help)
    kv.add_argument("--chunk-length", type=int, default=512, help=chunk_length_help)
    kv.add_argument("--model-revision", default=None, help=model_revision_help)
    kv.add_argument(
        "--control",
        action="store_true",
        help="also run the stock method's quantizer-only control lane (see --kv-method stock)",
    )
    kv.add_argument("--format", choices=["json", "md", "badge"], default="md", help=format_help)

    allow_code_help = (
        "allow a repo whose config.json names a model_file to run its own Python code "
        "(off by default: mlx-lm executes that file on load)"
    )
    kv.add_argument("--allow-custom-code", action="store_true", help=allow_code_help)

    repo_help = (
        "repo id or local path; repo@revision pins a Hub revision (an inline pin wins "
        "over --quant-revision / --reference-revision)"
    )

    weights = sub.add_parser("weights", help="measure weight-quantization fidelity")
    weights.add_argument("quant_model", help=repo_help)
    weights.add_argument("--reference", required=True, help=repo_help)
    weights.add_argument("--max-chunks", type=int, default=None, help=max_chunks_help)
    weights.add_argument("--quant-revision", default=None, help=quant_revision_help)
    weights.add_argument("--reference-revision", default=None, help=reference_revision_help)
    weights.add_argument("--allow-custom-code", action="store_true", help=allow_code_help)
    weights.add_argument(
        "--format", choices=["json", "md", "badge"], default="md", help=format_help
    )

    compare = sub.add_parser("compare", help="rank N quantizations on a memory-normalized Pareto")
    csub = compare.add_subparsers(dest="compare_mode", required=True)

    cw = csub.add_parser("weights", help="rank N weight-quant repos vs a reference")
    cw.add_argument("quant_models", nargs="+", help=repo_help)
    cw.add_argument("--reference", required=True, help=repo_help)
    cw.add_argument("--max-chunks", type=int, default=None, help=max_chunks_help)
    cw.add_argument("--max-kld", type=float, default=None, help=max_kld_help)
    cw.add_argument(
        "--min-tier", choices=["good", "marginal", "bad"], default=None, help=min_tier_help
    )
    cw.add_argument("--quant-revision", default=None, help=quant_revision_help)
    cw.add_argument("--reference-revision", default=None, help=reference_revision_help)
    cw.add_argument("--allow-custom-code", action="store_true", help=allow_code_help)
    cw.add_argument(
        "--worker-timeout",
        type=_non_negative_seconds,
        default=21600.0,
        metavar="SECONDS",
        help="wall-clock limit per target's worker process; a target over it becomes a failed "
        "row (default 21600 = 6 h; 0 disables)",
    )
    cw.add_argument("--format", choices=["json", "md"], default="md", help=format_help)

    ck = csub.add_parser("kv", help="rank N KV-cache methods/configs on one model")
    ck.add_argument("model", help="model repo id or local path")
    ck.add_argument(
        "--configs",
        default=None,
        help="e.g. '4:32,4:64,8:64,turboquant:4,turboquant-vonly:3,affine:8:4' (methods: "
        "bits:group_size = stock; turboquant:bits[:seed], seed >= 1; "
        "turboquant-vonly:v_bits[:seed]; affine:k_bits:v_bits[:group_size])",
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
    ck.add_argument("--quantize-start", type=int, default=0, help=quantize_start_help)
    ck.add_argument("--max-chunks", type=int, default=None, help=max_chunks_help)
    ck.add_argument("--chunk-length", type=int, default=512, help=chunk_length_help)
    ck.add_argument("--model-revision", default=None, help=model_revision_help)
    ck.add_argument("--max-kld", type=float, default=None, help=max_kld_help)
    ck.add_argument(
        "--min-tier", choices=["good", "marginal", "bad"], default=None, help=min_tier_help
    )
    ck.add_argument("--allow-custom-code", action="store_true", help=allow_code_help)
    ck.add_argument("--format", choices=["json", "md"], default="md", help=format_help)

    args = parser.parse_args(argv)
    exit_code = 0
    progress = _stderr_progress
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
                allow_custom_code=args.allow_custom_code,
                progress=progress,
            )
            if args.format == "json":
                out = render_json(report)
            elif args.format == "badge":
                out = render_badge_markdown(report)
            else:
                out = render_markdown(report)
        elif args.command == "weights":
            quant_id, quant_inline = split_target(args.quant_model)
            reference_id, reference_inline = split_target(args.reference)
            wreport = measure_weight_fidelity(
                quant_id,
                reference_id,
                max_chunks=args.max_chunks,
                quant_revision=quant_inline if quant_inline is not None else args.quant_revision,
                reference_revision=(
                    reference_inline if reference_inline is not None else args.reference_revision
                ),
                allow_custom_code=args.allow_custom_code,
                progress=progress,
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
                allow_custom_code=args.allow_custom_code,
                worker_timeout_s=args.worker_timeout if args.worker_timeout > 0 else None,
                progress=progress,
            )
            out = (
                render_comparison_json(creport)
                if args.format == "json"
                else render_comparison_markdown(creport)
            )
            if _all_rows_failed(creport):
                exit_code = 1
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
                config_json = read_model_config(args.model, args.model_revision)
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
                allow_custom_code=args.allow_custom_code,
                progress=progress,
            )
            out = (
                render_comparison_json(creport)
                if args.format == "json"
                else render_comparison_markdown(creport)
            )
            if _all_rows_failed(creport):
                exit_code = 1
        else:
            raise AssertionError(  # pragma: no cover
                f"unhandled compare_mode: {args.compare_mode!r}"
            )
    except QuantFidelityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except _user_error_types() as exc:
        # Raised from inside a probe, where the failing repo is unknown: stay neutral rather
        # than blame the first model id on the command line.
        first_line = (str(exc).splitlines() or [type(exc).__name__])[0]
        print(f"error: {first_line} (typo, gated repo, or offline?)", file=sys.stderr)
        return 2
    print(out)
    return exit_code


def _console_entry() -> None:
    """Console-script entry point.

    Runs :func:`main`, flushes output, then hard-exits via ``os._exit`` to skip MLX's
    Metal backend C++ destructor, which segfaults at interpreter shutdown on Apple
    Silicon ("Python quit unexpectedly"). The ``finally`` guarantees the hard exit on
    every path — including an unexpected error from ``main`` — so the teardown segfault
    cannot leak through on an error.
    """
    from mlx_quant_fidelity._watchdog import MemoryWatchdog

    code = 1
    try:
        install_memory_caps()
        MemoryWatchdog().start()  # console only: never started by main() or the library API
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
