"""Latent-cache reproducer for a deepseek_v3 (MLA) checkpoint: stock vs affine KV caches.

deepseek_v3 caches the compressed latent and the rope key (``cache.update_and_fetch(kv_latent,
k_pe)`` in ``mlx_lm/models/deepseek_v3.py``), then calls ``k_pe.swapaxes`` on what the cache
returns. mlx-lm's stock ``QuantizedKVCache`` returns a packed tuple there, so the first forward
fails; a dequantize-on-fetch cache returns plain arrays and runs. This script runs one forward
per condition in a single process and records what happens:

(a) the model's own ``make_prompt_cache`` caches (full precision): the reference logits, fp32;
(b) every layer on mlx-lm's stock ``QuantizedKVCache(group_size=64, bits=4)``: the exception
    type, message and innermost frame, or the logits and their KL if it runs;
(c) every layer on the package's affine cache (dequantize on fetch), two symmetric variants,
    ``affine:8:8`` and ``affine:4:4`` (group 64): full-vocab KL(P_ref || Q) per position,
    mean / median / max, reported per variant. Symmetric because on deepseek_v3 the key slot
    holds the latent and the value slot holds the rope key; an asymmetric split would mix them.

The input is a fixed synthetic token-id array, ``mx.arange(1, tokens + 1) % vocab`` (64 tokens
by default): the script tests cache mechanics, not text quality, so no tokenizer is loaded and
the KL figures are not a corpus measurement. Only the model's weights are loaded, with
``mlx_lm.utils.load_model`` from the locally cached snapshot (``local_files_only``; it never
downloads). A repo whose config names its own model code is refused before load.

Exit codes: 0 whenever the run ends by itself. Per-condition errors, a pre-load refusal (custom
model code, wrong model_type, snapshot not cached) and a load error are all recorded in the
``--out`` JSON (``conditions``, ``refused``, ``error``), never turned into an exit code. 3 only
for the memory watchdog or the 900 s wall backstop (the JSON keeps the finished conditions plus
``aborted`` and ``reason``).

This loads a 16B-parameter checkpoint (about 9 GB peak on the default model): run it alone on
the machine, never beside another model run. ``--plan`` prints the plan and the snapshot path if
it is cached, loads nothing and exits 0:

    uv run --no-sync python scripts/spike_mla_latent_cache.py --out <result.json> --plan
    uv run --no-sync python scripts/spike_mla_latent_cache.py --out <result.json>
"""

import argparse
import contextlib
import functools
import importlib.metadata
import json
import os
import statistics
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import mlx.core as mx

from mlx_quant_fidelity._memory_caps import compute_safe_caps_gb, device_string, install_memory_caps
from mlx_quant_fidelity._watchdog import EXIT_MEMORY_CEILING, MemoryWatchdog, default_ceiling

PROG = "spike_mla_latent_cache"
GIB = 1024**3
CACHE_LIMIT_BYTES = 4 * GIB
WALL_BACKSTOP_S = 900.0
EXPECTED_MODEL_TYPE = "deepseek_v3"
DEFAULT_REPO = "mlx-community/Moonlight-16B-A3B-Instruct-4-bit"
# The commit measured on 2026-10-24 (mlx 0.31.2, mlx-lm 0.31.3).
DEFAULT_REVISION = "8503ff1a2065f57fc9dc8fb95fabd864849c0744"
DEFAULT_TOKENS = 64
STOCK_BITS = 4
STOCK_GROUP_SIZE = 64
AFFINE_SPECS = ("affine:8:8", "affine:4:4")  # symmetric: key slot = latent, value slot = rope key

HELP_DESCRIPTION = (
    "Reproduce the deepseek_v3 (MLA) latent-cache behaviour: one forward on the full-precision "
    "cache, on mlx-lm's stock QuantizedKVCache, and on the affine dequantize-on-fetch cache. "
    "Loads the model's weights only (no tokenizer, never downloads; custom model code is "
    f"refused), scores {DEFAULT_TOKENS} synthetic tokens and reports cache mechanics (which "
    "cache runs, the error of the one that does not, KL against the reference), not a corpus "
    "number. Writes one JSON to --out."
)


class _RefusedError(Exception):
    """A pre-load refusal: recorded in the JSON as ``refused``; the run still exits 0."""


class _Record:
    """The JSON written to ``--out``: a partial after every step, the final one at the end.

    One lock serializes the main thread and the abort paths (watchdog thread, backstop timer),
    and every write is atomic (temp file + replace), so an abort never leaves half a file.
    """

    def __init__(self, out: Path, payload: dict[str, Any]) -> None:
        """Hold the output path and the initial payload; nothing is written yet."""
        self._out = out
        self._payload = payload
        self._lock = threading.Lock()

    def update(self, **fields: Any) -> None:
        """Merge ``fields`` into the payload and write it."""
        with self._lock:
            self._payload.update(fields)
            self._write_locked()

    def set_condition(self, name: str, result: dict[str, Any]) -> None:
        """Record one condition's result and write the partial JSON."""
        with self._lock:
            self._payload["conditions"][name] = result
            self._payload["peak_memory_bytes"] = int(mx.get_peak_memory())
            self._write_locked()

    def abort(self, aborted: str, reason: str, **fields: Any) -> None:
        """Write what finished plus ``aborted``/``reason``, then hard-exit with the watchdog code."""
        with self._lock:
            self._payload.update(fields, aborted=aborted, reason=reason)
            with contextlib.suppress(Exception):
                self._write_locked()
        print(f"{PROG}: aborted ({aborted}): {reason}", file=sys.stderr)
        sys.stderr.flush()
        os._exit(EXIT_MEMORY_CEILING)

    def _write_locked(self) -> None:
        self._out.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._out.with_name(self._out.name + ".tmp")
        tmp.write_text(json.dumps(self._payload, indent=2, sort_keys=True) + "\n")
        tmp.replace(self._out)


def _cached_snapshot(repo: str, revision: str) -> Path | None:
    """The local snapshot directory for ``repo@revision``, or None; never touches the network."""
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo, revision=revision, local_files_only=True))
    except Exception:
        return None


def _plan(args: argparse.Namespace) -> int:
    """Print what a real run would do and the cached snapshot (if any); load nothing."""
    from mlx_quant_fidelity.probes._preload import refuse_custom_code

    wired_gb, memory_gb = compute_safe_caps_gb()
    print(f"{PROG} plan (nothing is loaded)")
    print(f"  model:     {args.repo} @ {args.revision}")
    print(f"  out:       {args.out}")
    print(f"  input:     synthetic token ids mx.arange(1, {args.tokens + 1}) % vocab, batch 1")
    print(f"  caps:      wired {wired_gb} GiB, memory {memory_gb} GiB, cache 4 GiB (device caps)")
    print(
        f"  watchdog:  active + cache > {default_ceiling() / GIB:.2f} GiB -> exit 3; "
        f"wall backstop {WALL_BACKSTOP_S:.0f} s -> exit 3"
    )
    print(f"  device:    {device_string()}")
    print("  load:      preload_check(allow_custom_code=False) then mlx_lm.utils.load_model")
    print(f"  require:   model_type == {EXPECTED_MODEL_TYPE!r}")
    print("  (a) reference:  make_prompt_cache(model), fp32 logits")
    print(f"  (b) stock:      QuantizedKVCache(group_size={STOCK_GROUP_SIZE}, bits={STOCK_BITS})")
    print(f"  (c) affine:     {' and '.join(AFFINE_SPECS)}, group 64, KL vs (a), one per variant")
    snapshot = _cached_snapshot(args.repo, args.revision)
    if snapshot is None:
        print("  snapshot:  not cached (download the snapshot first)")
        return 0
    print(f"  snapshot:  {snapshot}")
    config_path = snapshot / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  config:    unreadable ({type(exc).__name__}: {exc})")
        return 0
    shards = sorted(p.name for p in snapshot.glob("model*.safetensors"))
    print(f"  config:    model_type={config.get('model_type')!r} vocab={config.get('vocab_size')}")
    print(f"  shards:    {len(shards)} {shards}")
    try:
        refuse_custom_code(config, model=args.repo, allow_custom_code=False)
        print("  code:      no model_file (would load)")
    except Exception as exc:
        print(f"  code:      REFUSED ({exc})")
    return 0


def _arm(record: _Record) -> tuple[MemoryWatchdog, threading.Timer]:
    """Start the active+cache watchdog and the wall backstop; both abort through ``record``."""

    def on_breach(active: int, cache: int, ceiling: int) -> None:
        record.abort(
            "memory-watchdog",
            f"active {active / GIB:.2f} GiB + cache {cache / GIB:.2f} GiB exceeds the "
            f"{ceiling / GIB:.2f} GiB ceiling",
            active_bytes=active,
            cache_bytes=cache,
            ceiling_bytes=ceiling,
        )

    def on_wall() -> None:
        record.abort("wall-clock-backstop", f"run exceeded {WALL_BACKSTOP_S:.0f} s")

    watchdog = MemoryWatchdog(on_breach=on_breach).start()
    timer = threading.Timer(WALL_BACKSTOP_S, on_wall)
    timer.daemon = True
    timer.start()
    return watchdog, timer


def _innermost_frame(exc: BaseException) -> str | None:
    """``file:line in function`` of the deepest traceback frame, or None."""
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return None
    last = frames[-1]
    return f"{last.filename}:{last.lineno} in {last.name}"


def _kl_stats(ref: mx.array, logits: mx.array) -> dict[str, float]:
    """Mean / median / max of the per-position full-vocab KL(P_ref || Q), in nats."""
    from mlx_quant_fidelity.metrics.kl import kl_divergence

    kl = kl_divergence(ref, logits)
    mx.eval(kl)
    values = [float(v) for v in cast("list[float]", kl.tolist())]  # [positions] -> flat list
    return {
        "kl_mean": statistics.fmean(values),
        "kl_median": statistics.median(values),
        "kl_max": max(values),
    }


def _forward(model: Any, tokens: mx.array, cache: list[Any]) -> mx.array:
    """One prefill forward over ``tokens``; returns evaluated fp32 logits ``[positions, vocab]``."""
    logits: mx.array = model(tokens, cache=cache)[0].astype(mx.float32)
    mx.eval(logits)
    return logits


def _condition(
    model: Any,
    tokens: mx.array,
    make_cache: Callable[[], list[Any]],
    ref: mx.array | None,
) -> tuple[dict[str, Any], mx.array | None]:
    """Run one condition on a fresh cache; return its JSON result and its logits (None on error)."""
    result: dict[str, Any] = {
        "status": "ok",
        "error_type": None,
        "error_message": None,
        "frame": None,
        "kl_mean": None,
        "kl_median": None,
        "kl_max": None,
    }
    logits: mx.array | None = None
    try:
        cache = make_cache()
        result["cache_types"] = sorted({type(c).__name__ for c in cache})
        logits = _forward(model, tokens, cache)
        result["logits_finite"] = bool(mx.all(mx.isfinite(logits)).item())
        if ref is not None:
            result.update(_kl_stats(ref, logits))
        del cache
    except Exception as exc:
        logits = None
        result.update(
            status="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            frame=_innermost_frame(exc),
        )
    mx.clear_cache()
    return result, logits


def _run(args: argparse.Namespace) -> int:
    """Arm the caps and watchdogs, load the weights, run the conditions, write the JSON."""
    # Bounds first, before anything can allocate: device caps, retained pool, watchdog, backstop.
    caps = install_memory_caps()
    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    mx.reset_peak_memory()
    started = time.monotonic()
    record = _Record(
        args.out,
        {
            "model": args.repo,
            "revision": args.revision,
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_lm_version": importlib.metadata.version("mlx-lm"),
            "device": device_string(),
            "caps_gb": {"wired": caps[0], "memory": caps[1], "cache": CACHE_LIMIT_BYTES // GIB},
            "input": f"synthetic: mx.arange(1, {args.tokens + 1}) % vocab, batch 1 (no tokenizer)",
            "tokens": args.tokens,
            "conditions": {},
            "peak_memory_bytes": 0,
            "wall_seconds": None,
            "complete": False,
        },
    )
    record.update()  # the --out file exists from here on, whatever happens next
    watchdog, timer = _arm(record)
    try:
        _measure(args, record, started)
    except _RefusedError as exc:
        record.update(refused=str(exc))
        print(f"{PROG}: refused: {exc}", file=sys.stderr)
    except Exception as exc:
        record.update(
            error={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "frame": _innermost_frame(exc),
            }
        )
        print(f"{PROG}: error: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        timer.cancel()
        watchdog.stop()
    record.update(
        peak_memory_bytes=int(mx.get_peak_memory()), wall_seconds=time.monotonic() - started
    )
    print(f"{PROG}: wrote {args.out}")
    return 0


def _measure(args: argparse.Namespace, record: _Record, started: float) -> None:
    """Check, load, and run the conditions; raise _RefusedError before load when refused."""
    from mlx_quant_fidelity.errors import QuantFidelityError
    from mlx_quant_fidelity.probes._preload import preload_check

    try:
        checked = preload_check(args.repo, args.revision, allow_custom_code=False)
    except QuantFidelityError as exc:
        raise _RefusedError(f"{type(exc).__name__}: {exc}") from exc
    model_type = checked.config.get("model_type")
    if model_type != EXPECTED_MODEL_TYPE:
        raise _RefusedError(f"model_type {model_type!r} is not {EXPECTED_MODEL_TYPE!r}")
    revision = checked.load_revision(args.revision) or args.revision
    snapshot = _cached_snapshot(args.repo, revision)
    if snapshot is None:
        raise _RefusedError(f"snapshot {args.repo}@{revision} is not cached")
    record.update(checked_revision=revision, model_type=model_type)

    from mlx_lm.models.cache import (
        QuantizedKVCache,  # noqa: TID251 -- the reproducer measures mlx-lm's stock cache itself
        make_prompt_cache,
    )
    from mlx_lm.utils import load_model

    from mlx_quant_fidelity.probes.kv_methods import parse_method_spec

    model, config = load_model(snapshot, lazy=False)  # weights only, never a tokenizer
    loaded_type = getattr(model, "model_type", None)
    if loaded_type != EXPECTED_MODEL_TYPE:
        raise _RefusedError(f"loaded model_type {loaded_type!r} is not {EXPECTED_MODEL_TYPE!r}")
    vocab = int(config["vocab_size"])
    n_layers = len(model.layers)
    tokens = (mx.arange(1, args.tokens + 1) % vocab)[None]
    mx.eval(tokens)
    record.update(vocab_size=vocab, n_layers=n_layers, load_seconds=time.monotonic() - started)

    ref_result, ref = _condition(model, tokens, lambda: make_prompt_cache(model), None)
    record.set_condition("reference", ref_result)

    stock_result, stock_logits = _condition(
        model,
        tokens,
        lambda: [
            QuantizedKVCache(group_size=STOCK_GROUP_SIZE, bits=STOCK_BITS) for _ in range(n_layers)
        ],
        ref,
    )
    del stock_logits
    mx.clear_cache()
    record.set_condition(f"stock_{STOCK_BITS}_{STOCK_GROUP_SIZE}", stock_result)

    for spec in AFFINE_SPECS:
        method = parse_method_spec(spec)
        affine_result, affine_logits = _condition(
            model, tokens, functools.partial(method.make_cache, n_layers=n_layers), ref
        )
        affine_result["method"] = spec
        del affine_logits
        mx.clear_cache()
        record.set_condition(spec.replace(":", "_"), affine_result)

    record.update(complete=True)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments; ``--plan`` prints the plan, anything else runs the reproducer."""
    parser = argparse.ArgumentParser(prog=PROG, description=HELP_DESCRIPTION)
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"Hub repo id of a deepseek_v3 checkpoint ({DEFAULT_REPO})",
    )
    parser.add_argument(
        "--revision", default=DEFAULT_REVISION, help=f"commit sha to load ({DEFAULT_REVISION})"
    )
    parser.add_argument("--out", required=True, type=Path, help="JSON result path")
    parser.add_argument(
        "--tokens",
        type=int,
        default=DEFAULT_TOKENS,
        help=f"synthetic prompt length ({DEFAULT_TOKENS})",
    )
    parser.add_argument("--plan", action="store_true", help="print the plan; load nothing")
    args = parser.parse_args(argv)
    if args.tokens < 2:
        parser.error("--tokens must be at least 2 (the prefill path needs L > 1)")
    if args.plan:
        return _plan(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
