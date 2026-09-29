"""Teacher-forced, streaming KV-quant fidelity probe."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.metadata
from typing import TYPE_CHECKING, cast

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache

from mlx_quant_fidelity._memory_caps import (
    caps_warning,
    compute_safe_caps_gb,
    device_string,
    install_memory_caps,
)
from mlx_quant_fidelity.corpora.provenance import scored_provenance
from mlx_quant_fidelity.errors import (
    CacheNotQuantizableError,
    CompareConfigError,
    CorpusError,
    LogitsBudgetError,
    QuantizeStartError,
)
from mlx_quant_fidelity.metrics import bucket_by_depth, kl_divergence, summarize, top_token_flips
from mlx_quant_fidelity.policy import verdict_for
from mlx_quant_fidelity.probes._paired import (
    ProgressFn,
    _Aggregate,
    _aggregate_chunks,
    _check_exact_zero,
    _reduce_pair,
    _require_finite,
    report_chunk_progress,
)
from mlx_quant_fidelity.probes._preload import preload_check
from mlx_quant_fidelity.probes.kv_methods import (
    ControlLaneMethod,
    PartialCoverageMethod,
    StockKVMethod,
    check_before_load,
)
from mlx_quant_fidelity.probes.kv_methods import (
    packed_width_mismatch as packed_width_mismatch,  # re-exported: historical import site
)
from mlx_quant_fidelity.report import FidelityReport

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mlx_quant_fidelity.corpora.provenance import Corpus, CorpusProvenance
    from mlx_quant_fidelity.metrics import DepthBucketSummary, ScalarSummary
    from mlx_quant_fidelity.probes.kv_methods import KVCacheMethod, LayerCoverage


# Hard ceiling on chunk_length (kernel-panic safety surface — paired fp32 logits scale with
# window x vocab, see the memory gate in score_kv_config). Validated 2026-08-09 by the
# long-window memory spike (docs/measurement-principles.md, scripts/spike_long_window_memory.py):
# measured peak at chunk_length=4096 on Llama-3.2-1B-4bit (vocab ~128k) was ~13.5 GiB, under the
# device wired cap. The ceiling is therefore only validated AT THAT VOCABULARY — peak memory
# scales linearly with vocab, so a 262k-vocab model at 4096 would need roughly twice that. Larger
# vocabularies are gated dynamically against the installed wired cap (_preflight_logits_budget),
# not by this constant.
MAX_CHUNK_LENGTH = 4096

# Fraction of the installed wired cap the per-chunk paired-logits estimate may occupy before the
# probe refuses to run. Leaves room for model weights, the KV cache, and the allocator's retained
# pool — the estimate covers only the transient logits/log-softmax working set.
LOGITS_BUDGET_FRACTION = 0.7

# Bytes-per-window multiplier, calibrated against the 2026-08-09 measured long-window spike
# (docs/measurement-principles.md): the measured peak slope between the 2048 and 4096
# chunk_length lanes on Llama-3.2-1B-4bit is ~6.6 [positions, vocab] fp32-array-equivalents
# per window (raw logits + log-softmax temporaries under the lazy graph), rounded up to 7 so
# the estimate keeps erring high. The slope excludes the model-weight/KV-cache memory
# intercept, so small models fire the warning slightly early -- the right direction for a
# safety warning.
_LOGITS_ARRAYS_PER_WINDOW = 7

# Estimate above which the report carries a memory warning (the band below the hard gate).
_LOGITS_WARN_BYTES = 4 * 1024**3

# Method names carrying a MEASURED long-window receipt in docs/measurement-principles.md (the
# long-window memory spike, scripts/spike_long_window_memory.py, run per method lane). A method
# name outside this set has no measured evidence that windows above 512 stay within the memory
# ceiling, so it still gets the interim >512 warning below.
RECEIPTED_METHODS = frozenset({"stock", "turboquant", "turboquant-vonly", "affine"})


# Headroom kept free of the device working set on top of weights + transient estimate.
RESIDENT_HEADROOM_BYTES = 2 * 1024**3

# Largest window gated without a known vocabulary: above it an underivable vocab_size refuses.
UNKNOWN_VOCAB_MAX_WINDOW = 512


def _model_arg(args: object, name: str) -> object:
    """Read a geometry field from ``args``, falling back to ``args.text_config``.

    mlx-lm wrapper architectures keep their language-model geometry under ``text_config``
    (a dict or an object). Returns None for a missing field or a non-positive int (0 is how
    some configs say "derive it").
    """

    def _clean(value: object) -> object:
        if isinstance(value, int) and not isinstance(value, bool) and value <= 0:
            return None
        return value

    value = _clean(getattr(args, name, None))
    if value is not None:
        return value
    text_config = getattr(args, "text_config", None)
    if text_config is None:
        return None
    if isinstance(text_config, dict):
        return _clean(text_config.get(name))
    return _clean(getattr(text_config, name, None))


def model_resident_bytes(model: object) -> int:
    """Bytes held by the model's parameters (metadata only, no evaluation); 0 if unknown.

    Arrays shared under several keys (tied embeddings) are counted once.
    """
    params = getattr(model, "parameters", None)
    if not callable(params):
        return 0
    try:
        flat = cast("list[tuple[str, mx.array]]", tree_flatten(params()))
        # Tied weights appear under several keys but occupy one buffer: count each once.
        unique = {id(leaf): leaf for _, leaf in flat if hasattr(leaf, "nbytes")}
        return sum(int(leaf.nbytes) for leaf in unique.values())
    except Exception:
        return 0


def _max_working_set_bytes() -> int:
    """The device's recommended working-set size in bytes; 0 when not reported."""
    try:
        info = mx.device_info()
    except Exception:
        return 0
    return int(info.get("max_recommended_working_set_size", 0) or 0)


def _paired_logits_bytes(window: int, vocab: int) -> int:
    """Estimated transient bytes held by the paired fp32 logits for one chunk."""
    return _LOGITS_ARRAYS_PER_WINDOW * (window - 1) * int(vocab) * 4


def _preflight_logits_budget(
    window: int | None,
    vocab: int | None,
    *,
    method_ws_bytes: int = 0,
    control_bytes: int = 0,
    cache_bytes: int = 0,
    resident_bytes: int = 0,
    remedy: str | None = None,
) -> str | None:
    """Refuse windows whose paired-logits + method/control working-set estimate exceeds the cap.

    ``method_ws_bytes`` (the scored method's own :meth:`KVCacheMethod.working_set_bytes`) and
    ``control_bytes`` (the control cache's stored + working-set bytes, when ``--control`` is on)
    both ADD to the paired-logits estimate, for both the gate comparison and the warn band —
    they are real resident/transient memory the same run holds, not a separate budget.

    Returns a warning string (or None) for the band below the gate; raises
    ``LogitsBudgetError`` above it. The gate runs BEFORE any cache construction or forward
    pass — a warning emitted in the report arrives after the allocation it was meant to
    prevent, which is exactly the pageable-allocation paging-storm path the ceiling exists to
    avoid.

    ``resident_bytes`` (the model's parameter bytes) is checked first and on its own (weights
    alone must fit in the device working set less ``RESIDENT_HEADROOM_BYTES``, whatever the
    window or vocab), then together with the estimate and ``cache_bytes`` (the KV caches the run
    holds). ``cache_bytes`` counts ONLY in that working-set rule: the 70%-of-wired logits budget
    is calibrated on measured total peaks that already held the caches, so adding them there
    would count them twice.

    Skipped (warning path only) when the device reports no working-set size, since there is
    no cap to measure against.
    """
    max_ws = _max_working_set_bytes()
    if max_ws and resident_bytes > max_ws - RESIDENT_HEADROOM_BYTES:
        # Independent of window and vocab: the weights alone leave no room, so no --chunk-length
        # can help.
        raise LogitsBudgetError(
            f"model weights {resident_bytes / 1024**3:.1f} GiB alone exceed the "
            f"{(max_ws - RESIDENT_HEADROOM_BYTES) / 1024**3:.1f} GiB limit (device working set "
            f"{max_ws / 1024**3:.1f} GiB less {RESIDENT_HEADROOM_BYTES / 1024**3:.0f} GiB "
            "headroom): this model does not fit this device; see docs/measurement-principles.md."
        )
    if window is not None and window > UNKNOWN_VOCAB_MAX_WINDOW and not vocab:
        raise LogitsBudgetError(
            f"chunk_length={window}: vocab_size is not derivable for this architecture, so a "
            f"window above {UNKNOWN_VOCAB_MAX_WINDOW} cannot be gated; use --chunk-length "
            f"{UNKNOWN_VOCAB_MAX_WINDOW} or smaller."
        )
    if not vocab or not window:
        return (
            f"per-chunk paired fp32 logits budget is UNCHECKED (vocab_size={vocab!r}, "
            f"chunk_length={window!r}): one of them is not derivable, so the estimate could "
            "not be gated against the installed wired cap. Prefer a smaller --chunk-length "
            "on an unfamiliar architecture."
        )
    est = _paired_logits_bytes(window, vocab) + method_ws_bytes + control_bytes
    if remedy is None:
        remedy = "Lower --chunk-length (halving it roughly halves the estimate)"
        if control_bytes > 0:
            remedy += " or drop --control"
    if max_ws and resident_bytes + est + cache_bytes > max_ws - RESIDENT_HEADROOM_BYTES:
        raise LogitsBudgetError(
            f"chunk_length={window} at vocab_size={int(vocab)}: model weights "
            f"{resident_bytes / 1024**3:.1f} GiB + per-chunk working set "
            f"{(est + cache_bytes) / 1024**3:.1f} GiB exceed the {(max_ws - RESIDENT_HEADROOM_BYTES) / 1024**3:.1f}"
            f" GiB limit (device working set {max_ws / 1024**3:.1f} GiB less "
            f"{RESIDENT_HEADROOM_BYTES / 1024**3:.0f} GiB headroom). {remedy}; "
            "see docs/measurement-principles.md."
        )
    wired_gb, _ = compute_safe_caps_gb()
    if wired_gb:
        gate = int(LOGITS_BUDGET_FRACTION * wired_gb * 1024**3)
        if est > gate:
            raise LogitsBudgetError(
                f"chunk_length={window} at vocab_size={int(vocab)}: paired fp32 logits peak ≈ "
                f"{est / 1024**3:.1f} GiB per chunk, above the {gate / 1024**3:.1f} GiB "
                f"pre-flight budget ({LOGITS_BUDGET_FRACTION:.0%} of the {wired_gb} GiB "
                f"installed wired cap). {remedy}; see docs/measurement-principles.md."
            )
    if est > _LOGITS_WARN_BYTES:
        return (
            f"chunk_length={window}: paired fp32 logits peak ≈ {est / 1024**3:.1f} GiB per "
            "chunk; see docs/measurement-principles.md (Drift by position depth) for "
            "measured ceilings."
        )
    return None


def _kv_head_dim(model: object) -> int | None:
    """Best-effort per-head KV dim for the group-size gate. None if not derivable.

    ``ModelArgs.head_dim`` is frequently None (llama populates the real value as an
    Attention-layer local, not on args), so the hidden//heads fallback is the primary
    path. Field names are llama-family-specific; unusual archs return None.
    """
    args = getattr(model, "args", None)
    head_dim = _model_arg(args, "head_dim")
    if isinstance(head_dim, int):
        return head_dim
    hidden = _model_arg(args, "hidden_size")
    heads = _model_arg(args, "num_attention_heads")
    if isinstance(hidden, int) and isinstance(heads, int):
        return hidden // heads
    return None


def _score_chunk(
    model: object,
    ids: mx.array,
    ref_cache: list[object],
    quant_cache: list[object],
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """One teacher-forced chunk: forward twice on identical tokens, reduce to per-position scalars.

    Returns (kl[positions-1], flips[positions-1], ref_nll[positions-1], quant_nll[positions-1]).
    The caller must mx.eval the returns and let the vocab-wide logits leave scope before the
    next chunk; this function holds no logits beyond its own frame.
    """
    inp = ids[None, :-1]
    targets = ids[1:]
    ref_logits = model(inp, cache=ref_cache)[0].astype(mx.float32)  # type: ignore[operator]
    quant_logits = model(inp, cache=quant_cache)[0].astype(mx.float32)  # type: ignore[operator]
    return _reduce_pair(ref_logits, quant_logits, targets)


def _score_chunk_control(
    model: object,
    ids: mx.array,
    ref_logits: mx.array,
    control_cache: list[object],
) -> tuple[mx.array, mx.array]:
    """Third forward against already-computed reference logits; KL + flips only."""
    control_logits = model(ids[None, :-1], cache=control_cache)[0].astype(mx.float32)  # type: ignore[operator]
    return kl_divergence(ref_logits, control_logits), top_token_flips(ref_logits, control_logits)


def _require_teacher_forced_positions(chunks: Sequence[mx.array]) -> None:
    """Every scored chunk needs at least one teacher-forced position (2 tokens)."""
    if any(c.shape[0] < 2 for c in chunks):
        raise CorpusError(
            "every corpus chunk must have at least 2 tokens (one teacher-forced position)."
        )


def _resolve_method(
    method: KVCacheMethod | None, kv_bits: int, kv_group_size: int
) -> KVCacheMethod:
    """Explicit ``method`` wins; the 0.5.x ``kv_bits``/``kv_group_size`` sugar builds stock."""
    if method is None:
        return StockKVMethod(bits=kv_bits, group_size=kv_group_size)
    if kv_bits != 4 or kv_group_size != 64:
        raise CompareConfigError(
            "pass either method= or kv_bits/kv_group_size, not both "
            "(the sugar only builds the stock method)."
        )
    return method


def _measured_bytes_per_token(method: KVCacheMethod, cache: list[object]) -> int | None:
    """Stored bytes / stored positions of a filled cache list; None if the cache has no offset."""
    offset = getattr(cache[0], "offset", None) if cache else None
    if not isinstance(offset, int) or offset <= 0:
        return None
    return round(method.measured_bytes(cache) / offset)


def _score_chunk_deployment(
    model: object,
    ids: mx.array,
    ref_cache: list[object],
    quant_cache: list[object],
    *,
    quantize_start: int,
    method: KVCacheMethod,
    control_method: KVCacheMethod | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array | None, mx.array | None]:
    """Deployment split: compute the prefix in full precision, then convert the stored cache.

    Returns per-position (kl, flips, ref_nll, quant_nll) over ONLY the post-boundary
    [n:L-1) prediction positions, plus (control_kl, control_flips) over the same region when
    ``control_method`` is given (``(None, None)`` otherwise). `quant_cache` starts
    full-precision (make_prompt_cache) and is converted at the boundary via
    ``method.convert_prefix`` (mirrors mlx-lm's maybe_quantize_kv_cache). Segment 1 (the
    full-precision prefix [0:n)) exists only to fill the cache: it is evaluated for its cache
    state and its logits are never kept or reduced, so no lm_head output, fp32 cast or
    log-softmax is spent on positions the report excludes. The control lane likewise never
    scores the prefix; it exists to isolate quantizer error at the boundary, not to double
    as a stress-mode run over the whole chunk.
    """
    n = quantize_start
    targets = ids[1:]
    ref_logits = model(ids[None, :-1], cache=ref_cache)[0].astype(mx.float32)  # type: ignore[operator]
    mx.eval(ref_logits)
    # Segment 1: prefix [0:n) through the full-precision quant_cache (identical to ref there).
    # Only the cache state is needed; collapse the seg-1 graph before the boundary.
    model(ids[None, :n], cache=quant_cache)  # type: ignore[operator]
    mx.eval([c.state for c in quant_cache])  # type: ignore[attr-defined]
    # Boundary: control conversion FIRST, while quant_cache still holds fp state -- the
    # bundled conversion below mutates quant_cache in place and would leave nothing for the
    # control lane to replay from if it ran second.
    control_caches = control_method.convert_prefix(quant_cache) if control_method else None
    quant_cache[:] = method.convert_prefix(quant_cache)  # in place: the caller's list sees it
    # Segment 2: [n:L-1) through the now-quantized cache. NOTE ids[n:-1], not ids[n:].
    seg2 = model(ids[None, n:-1], cache=quant_cache)[0].astype(mx.float32)  # type: ignore[operator]
    kl, flip, ref_nll, quant_nll = _reduce_pair(ref_logits[n:], seg2, targets[n:])
    mx.eval(kl, flip, ref_nll, quant_nll)
    del seg2  # free the bundled seg-2 logits before the control forward (two-tensor peak, spec §5)
    control_kl: mx.array | None = None
    control_flip: mx.array | None = None
    if control_caches is not None:
        control_logits = model(ids[None, n:-1], cache=control_caches)[0].astype(mx.float32)  # type: ignore[operator]
        control_kl = kl_divergence(ref_logits[n:], control_logits)
        control_flip = top_token_flips(ref_logits[n:], control_logits)
        mx.eval(control_kl, control_flip)
    return kl, flip, ref_nll, quant_nll, control_kl, control_flip


# mlx-lm KVCache growth step (mlx_lm/models/cache.py, ``KVCache.step``).
_KV_STEP = 256


def _gate_extra_bytes(
    method: KVCacheMethod,
    control_m: KVCacheMethod | None,
    *,
    window: object,
    args: object,
    head_dim: int | None,
    quantize_start: int = 0,
) -> tuple[int, int, int]:
    """(method working-set, control-cache, KV-cache) bytes for the pre-load memory gate.

    The third figure is the KV caches the run really holds, counted only against the device
    working set (see :func:`_preflight_logits_budget`): the reference run's fp16 cache (mlx-lm's ``KVCache`` grows in ``step = 256`` blocks, see
    ``mlx_lm/models/cache.py``), the scored method's stored cache, and, in deployment mode
    (``quantize_start > 0``), one more fp16 cache for the full-precision prefix held before
    conversion. A method that cannot size its cache contributes 0 for the stored term.

    Geometry is model.args-derived and computed BEFORE any cache construction (the gate's whole
    point is running before cache/model allocations). Each figure degrades to 0 (the 0.6.0
    estimate) when any one piece of geometry is unknown, rather than guessing.
    """
    n_layers = _model_arg(args, "num_hidden_layers")
    n_kv = _model_arg(args, "num_key_value_heads") or _model_arg(args, "num_attention_heads")
    if not (
        isinstance(window, int)
        and isinstance(n_layers, int)
        and isinstance(n_kv, int)
        and head_dim is not None
    ):
        return 0, 0, 0
    method_ws_bytes = method.working_set_bytes(
        window=window, n_layers=n_layers, n_kv_heads=n_kv, head_dim=head_dim, dtype_bytes=2
    )
    kv_fp = 2 * n_layers * n_kv * head_dim * 2 * (-(-window // _KV_STEP) * _KV_STEP)
    cache_bytes = kv_fp * (2 if quantize_start > 0 else 1)
    with contextlib.suppress(CacheNotQuantizableError):
        cache_bytes += (
            method.bytes_per_token(n_layers=n_layers, n_kv_heads=n_kv, head_dim=head_dim) * window
        )
    control_bytes = 0
    if control_m is not None:
        control_bytes = control_m.bytes_per_token(
            n_layers=n_layers, n_kv_heads=n_kv, head_dim=head_dim
        ) * window + control_m.working_set_bytes(
            window=window, n_layers=n_layers, n_kv_heads=n_kv, head_dim=head_dim, dtype_bytes=2
        )
    return method_ws_bytes, control_bytes, cache_bytes


def _resolve_partial_coverage(
    model: object,
    method: KVCacheMethod,
    *,
    quantize_start: int,
    control_m: KVCacheMethod | None,
    probe_warnings: list[str],
) -> tuple[int, LayerCoverage | None]:
    """Probe the cache once: layer count, capability gate, and per-layer partial coverage.

    A hybrid model passes ``probe_capability`` as long as ONE layer is quantizable, so this
    records which layers to quantize (stock only; quantizer-only methods do not implement the
    capability). A partial model is refused in deployment mode and with a control lane, and is
    smoke-tested with a one-token forward on the mixed cache; the coverage note is appended
    to ``probe_warnings``. ``make_prompt_cache`` is called through the module global so tests
    that monkeypatch ``kv.make_prompt_cache`` keep binding.
    """
    probe_cache = make_prompt_cache(model)
    n_layers = len(probe_cache)
    method.probe_capability(probe_cache)
    partial_coverage: LayerCoverage | None = None
    if isinstance(method, PartialCoverageMethod):
        cov = method.layer_coverage(probe_cache)
        if cov.is_partial:
            partial_coverage = cov
    del probe_cache

    if partial_coverage is not None and isinstance(method, PartialCoverageMethod):
        if quantize_start > 0:
            raise CacheNotQuantizableError(
                "partial KV coverage (a hybrid model with layers that cannot be quantized, "
                "e.g. sliding-window) is measured only in stress mode; measure it with the "
                "`kv` command without --quantize-start."
            )
        if control_m is not None:
            raise CacheNotQuantizableError(
                "partial KV coverage (a hybrid model with sliding-window or state-space layers) "
                "is measured only by the `kv` command without a control lane; it is not available "
                "in `compare kv` or with `kv --control`."
            )
        probe_warnings.append(partial_coverage.note())
        # Fail fast on an architecture whose forward cannot consume a mixed quantized /
        # full-precision cache: probe_capability only proves each layer quantizes in isolation,
        # not that the model runs over the heterogeneous list. A one-token forward here turns a
        # mid-loop crash into a clean, package-rooted error before any chunk is scored.
        smoke_cache = method.make_partial_cache(make_prompt_cache(model), partial_coverage)
        try:
            mx.eval(model(mx.array([[0]]), cache=smoke_cache))  # type: ignore[operator]
        except Exception as exc:
            raise CacheNotQuantizableError(
                "this model's forward does not run on a mixed quantized/full-precision KV cache "
                f"(partial coverage): {exc}"
            ) from exc
        del smoke_cache
        mx.clear_cache()
    return n_layers, partial_coverage


def _score_chunk_stress_control(
    model: object,
    ids: mx.array,
    ref_cache: list[object],
    quant_cache: list[object],
    control_m: KVCacheMethod,
    n_layers: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Stress-mode chunk with a third, quantizer-only forward.

    Peak is two vocab-wide tensors at once (spec §5): ref forward -> bundled forward ->
    reduce+eval bundled -> release bundled logits -> control forward -> reduce+eval control ->
    release ref logits. Returns (kl, flips, ref_nll, quant_nll, control_kl, control_flips).
    """
    inp, targets = ids[None, :-1], ids[1:]
    ref_logits = model(inp, cache=ref_cache)[0].astype(mx.float32)  # type: ignore[operator]
    quant_logits = model(inp, cache=quant_cache)[0].astype(mx.float32)  # type: ignore[operator]
    kl, flip, ref_nll, quant_nll = _reduce_pair(ref_logits, quant_logits, targets)
    mx.eval(kl, flip, ref_nll, quant_nll)
    del quant_logits  # bundled logits out of scope before the third forward
    control_cache = control_m.make_cache(n_layers=n_layers)
    kl_c, flip_c = _score_chunk_control(model, ids, ref_logits, control_cache)
    mx.eval(kl_c, flip_c)
    del control_cache, ref_logits
    return kl, flip, ref_nll, quant_nll, kl_c, flip_c


def _control_method_for(method: KVCacheMethod, control: bool) -> KVCacheMethod | None:
    """The quantizer-only control method when ``control`` is on; refuse a method without one.

    Model-free, so ``measure_kv_fidelity`` can call it before any load.
    """
    if not control:
        return None
    if not isinstance(method, ControlLaneMethod):
        raise CompareConfigError(
            f"--control only applies to a method with a bundled quantized-attention path "
            f"(stock); method {method.name!r} is already quantizer-only "
            "(dequantize-on-fetch, standard SDPA)."
        )
    return method.control_method()


def _assemble_kv_report(
    *,
    model_id: str,
    model_revision: str | None,
    method: KVCacheMethod,
    quantize_start: int,
    agg: _Aggregate,
    n_scored: int,
    provenance: CorpusProvenance,
    warnings: list[str],
    kl_by_depth: tuple[DepthBucketSummary, ...] | None,
    measured_bpt: int | None,
    control_summary: ScalarSummary | None,
    control_flip_rate: float | None,
    working_set_bytes_per_token: int | None,
    partial_coverage: LayerCoverage | None,
) -> FidelityReport:
    """Build the frozen report from the scored aggregates (pure; no MLX work)."""
    return FidelityReport(
        model_id=model_id,
        model_revision=model_revision,
        kv_bits=method.params.get("bits"),
        kv_group_size=method.params.get("group_size"),
        quantize_start=quantize_start,
        quantize_mode="stress" if quantize_start == 0 else "deployment",
        kl=agg.kl,
        flip_rate=agg.flip_rate,
        perplexity_ref=agg.perplexity_ref,
        perplexity_quant=agg.perplexity_quant,
        perplexity_delta=agg.perplexity_quant - agg.perplexity_ref,
        n_positions=agg.n_positions,
        n_chunks=n_scored,
        corpus=provenance,
        mlx_version=importlib.metadata.version("mlx"),
        mlx_lm_version=importlib.metadata.version("mlx-lm"),
        peak_memory_bytes=int(mx.get_peak_memory()),
        cache_supported=True,
        verdict=verdict_for(agg.kl.mean, agg.kl.p99, agg.flip_rate),
        warnings=tuple(warnings),
        device=device_string(),
        kl_by_depth=kl_by_depth,
        kv_method=method.name,
        kv_method_params=dict(method.params),
        kv_method_provenance=method.provenance(),
        measured_kv_bytes_per_token=measured_bpt,
        drift_footing="bundled" if isinstance(method, ControlLaneMethod) else "quantizer_only",
        control_kl=control_summary,
        control_flip_rate=control_flip_rate,
        working_set_bytes_per_token=working_set_bytes_per_token,
        kv_partial=partial_coverage is not None,
        kv_layers_total=partial_coverage.total if partial_coverage is not None else None,
        kv_layers_quantized=(
            len(partial_coverage.quantized_indices) if partial_coverage is not None else None
        ),
        kv_layers_skipped=(
            dict(partial_coverage.skipped_types) if partial_coverage is not None else None
        ),
    )


def score_kv_config(
    model: object,
    corpus: Corpus,
    *,
    model_id: str,
    model_revision: str | None = None,
    kv_bits: int = 4,
    kv_group_size: int = 64,
    method: KVCacheMethod | None = None,
    quantize_start: int = 0,
    max_chunks: int | None = None,
    control: bool = False,
    gate_remedy: str | None = None,
    progress: ProgressFn | None = None,
) -> FidelityReport:
    """Score one KV config on an ALREADY-LOADED model (no load, no caps install).

    Shared by ``measure_kv_fidelity`` (load -> delegate) and the KV ``compare`` adapter
    (load once -> loop configs). Applies ``max_chunks`` to the provided corpus,
    so a caller-supplied corpus is capped identically to the weight probe.

    ``progress`` receives ``chunk i/n`` lines about every tenth chunk (silent when ``None``).

    ``gate_remedy`` overrides the remedy text of a memory-gate refusal for callers whose
    command lacks the default's flags.

    ``control=True`` runs a third, quantizer-only forward per chunk (via
    ``method.control_method()``) so the report can separate quantizer error from
    quantized-attention-kernel numerics; raises CompareConfigError if ``method`` has no
    ``control_method``. In stress mode (``quantize_start=0``) the control forward covers
    the whole chunk; with ``quantize_start > 0`` the control cache converts from the same
    full-precision prefix as the bundled path and scores only the post-boundary region.
    """
    method = _resolve_method(method, kv_bits, kv_group_size)
    control_m = _control_method_for(method, control)
    probe_warnings: list[str] = []
    args = getattr(model, "args", None)
    model_type = str(getattr(args, "model_type", "unknown"))
    head_dim = _kv_head_dim(model)
    # Pure pre-flight first (keeps the 0.5.x warnings order: head_dim, then budget). It allocates
    # nothing, so the kernel-panic budget gate below still precedes any cache construction.
    probe_warnings.extend(method.check(head_dim=head_dim, model_type=model_type))
    window = getattr(corpus.provenance, "chunk_length", None)
    _require_teacher_forced_positions(
        corpus.chunks[:max_chunks] if max_chunks is not None else corpus.chunks
    )
    # Gate geometry is model.args-derived and computed BEFORE any cache construction (the gate's
    # whole point is running before cache/model allocations — see make_prompt_cache below, which
    # this precedes). Each of method_ws_bytes/control_bytes degrades to 0 (the 0.6.0 estimate)
    # when any one piece of geometry is unknown, rather than guessing.
    gate_vocab = _model_arg(args, "vocab_size")
    resident = model_resident_bytes(model)
    if resident == 0 and callable(getattr(model, "parameters", None)):
        probe_warnings.append(
            "model weight size unknown; the weights were not counted in the memory gate."
        )
    method_ws_bytes, control_bytes, cache_bytes = _gate_extra_bytes(
        method,
        control_m,
        window=window,
        args=args,
        head_dim=head_dim,
        quantize_start=quantize_start,
    )
    budget_warning = _preflight_logits_budget(
        window,
        gate_vocab if isinstance(gate_vocab, int) else None,
        method_ws_bytes=method_ws_bytes,
        control_bytes=control_bytes,
        cache_bytes=cache_bytes,
        resident_bytes=resident,
        remedy=gate_remedy,
    )
    if budget_warning is not None:
        probe_warnings.append(budget_warning)
    # The model is loaded lazily so the gate above sizes the weights before they are read.
    # Read them now, once, so a bad shard fails here with its own error rather than inside the
    # partial-coverage smoke forward or a scored chunk.
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        mx.eval(parameters())
    if method.name not in RECEIPTED_METHODS and isinstance(window, int) and window > 512:
        probe_warnings.append(
            f"chunk_length={window}: the memory ceiling was validated on the stock method; "
            f"method {method.name!r} retains additional full-precision working buffers, so "
            "validate peak memory before trusting large windows."
        )

    n_layers, partial_coverage = _resolve_partial_coverage(
        model,
        method,
        quantize_start=quantize_start,
        control_m=control_m,
        probe_warnings=probe_warnings,
    )

    mode = "stress" if quantize_start == 0 else "deployment"
    chunks = corpus.chunks[:max_chunks] if max_chunks is not None else corpus.chunks
    vocab_size = gate_vocab
    if isinstance(vocab_size, int) and vocab_size > 0 and chunks:
        max_id = int(mx.max(mx.stack([mx.max(ids) for ids in chunks])))
        if max_id >= vocab_size:
            raise CorpusError(
                f"corpus contains token id {max_id} but the model's vocab_size is {vocab_size}; "
                "a mismatched tokenizer or corpus would gather out of range and produce "
                "garbage perplexity silently."
            )
    kls: list[mx.array] = []
    flips: list[mx.array] = []
    ref_nlls: list[mx.array] = []
    quant_nlls: list[mx.array] = []
    control_kls: list[mx.array] = []
    control_flips: list[mx.array] = []
    n_scored = 0
    measured_bpt: int | None = None
    for chunk_index, ids in enumerate(chunks, start=1):
        report_chunk_progress(progress, chunk_index, len(chunks))
        if quantize_start > 0 and int(ids.size) < quantize_start + 2:
            continue  # too short to have a post-boundary position; skip
        ref_cache = make_prompt_cache(model)
        kl_c: mx.array | None = None
        flip_c: mx.array | None = None
        with method.guard():
            if quantize_start == 0:
                quant_cache: list[object]
                if partial_coverage is not None and isinstance(method, PartialCoverageMethod):
                    quant_cache = method.make_partial_cache(
                        make_prompt_cache(model), partial_coverage
                    )
                else:
                    quant_cache = method.make_cache(n_layers=n_layers)
                if control_m is None:
                    kl, flip, ref_nll, quant_nll = _score_chunk(model, ids, ref_cache, quant_cache)
                else:
                    kl, flip, ref_nll, quant_nll, kl_c, flip_c = _score_chunk_stress_control(
                        model, ids, ref_cache, quant_cache, control_m, n_layers
                    )
                    control_kls.append(kl_c)
                    control_flips.append(flip_c)
            else:
                quant_cache = make_prompt_cache(model)
                kl, flip, ref_nll, quant_nll, kl_c, flip_c = _score_chunk_deployment(
                    model,
                    ids,
                    ref_cache,
                    quant_cache,
                    quantize_start=quantize_start,
                    method=method,
                    control_method=control_m,
                )
                if kl_c is not None and flip_c is not None:
                    control_kls.append(kl_c)
                    control_flips.append(flip_c)
        mx.eval(kl, flip, ref_nll, quant_nll)
        if measured_bpt is None:
            measured_bpt = _measured_bytes_per_token(method, quant_cache)
        kls.append(kl)
        flips.append(flip)
        ref_nlls.append(ref_nll)
        quant_nlls.append(quant_nll)
        n_scored += 1
        del ref_cache, quant_cache
        mx.clear_cache()

    if quantize_start > 0 and n_scored == 0:
        raise QuantizeStartError(
            f"quantize_start={quantize_start} exceeds every scored chunk's length "
            "— no position was quantized; use a smaller boundary or longer chunks."
        )

    agg = _aggregate_chunks(kls, flips, ref_nlls, quant_nlls)
    _check_exact_zero(
        kl_mean=agg.kl.mean,
        flip_rate=agg.flip_rate,
        context=(
            f"quantization did not engage (quantize_start={quantize_start}; chunk may be "
            "shorter than the keep-first-N boundary, or the quantized cache was bypassed)"
        ),
    )

    control_summary = None
    control_flip_rate = None
    if control_m is not None:
        kl_all = np.concatenate([np.asarray(k, dtype=np.float64) for k in control_kls])
        flip_all = np.concatenate(
            [np.asarray(f.astype(mx.float32), dtype=np.float64) for f in control_flips]
        )
        _require_finite("control-lane KL", kl_all, allow_inf=True)
        control_summary = summarize(kl_all)
        control_flip_rate = float(flip_all.mean())
        _check_exact_zero(
            kl_mean=control_summary.mean,
            flip_rate=control_flip_rate,
            context="the control lane saw reference logits — its cache was bypassed "
            "(control did not engage)",
        )

    kl_by_depth = None
    if mode == "stress" and kls:
        # This np.asarray conversion of kls is built exactly once here and reused for both
        # the uniform-window check and bucket_by_depth below. _aggregate_chunks (called just
        # above) does its own equivalent conversion of the SAME kls internally, so the array
        # data is technically walked twice overall — but sharing the two would mean either
        # changing _aggregate_chunks's list[mx.array] signature (it's exercised directly,
        # with that exact signature, by tests/probes/test_kv_fakeforward.py) or passing it
        # pre-converted numpy arrays where mypy --strict expects mx.array (an ignore-driven
        # type lie). _aggregate_chunks is also shared verbatim with probes/weights.py, which
        # has no depth-bucket concept at all. Not worth that churn for one extra host-side
        # float64 cast over a per-chunk KLD list.
        arrays = [np.asarray(k, dtype=np.float64) for k in kls]
        if len({a.shape[0] for a in arrays}) == 1:  # uniform windows only
            kl_by_depth = bucket_by_depth(arrays)
        else:
            probe_warnings.append(
                "depth table omitted: scored chunks have unequal lengths "
                "(drift-by-depth requires a fixed-window corpus)."
            )

    n_kv = _model_arg(args, "num_key_value_heads") or _model_arg(args, "num_attention_heads")
    # Skip this sanity check for a partial report: the analytic figure assumes every layer is
    # quantized, while measured_bpt is the real mixed cache (a few packed layers + many
    # full-precision ones), so the two never match and the dtype attribution would be wrong.
    # partial_coverage.note() already explains the mixed footprint.
    if (
        partial_coverage is None
        and measured_bpt is not None
        and head_dim is not None
        and isinstance(n_kv, int)
    ):
        analytic = method.bytes_per_token(n_layers=n_layers, n_kv_heads=n_kv, head_dim=head_dim)
        if analytic != measured_bpt:
            probe_warnings.append(
                f"measured KV bytes/token {measured_bpt} differs from the analytic {analytic} for "
                f"method {method.name!r} (ranking uses the analytic figure; a scale/bias dtype other "
                "than fp16/bf16 is the usual cause)."
            )
    working_set_bytes_per_token: int | None = None
    if isinstance(window, int) and window > 0 and head_dim is not None and isinstance(n_kv, int):
        working_set_bytes_per_token = (
            method.working_set_bytes(
                window=window,
                n_layers=n_layers,
                n_kv_heads=n_kv,
                head_dim=head_dim,
                dtype_bytes=2,
            )
            // window
        )
    probe_warnings.extend(method.report_warnings())

    return _assemble_kv_report(
        model_id=model_id,
        model_revision=model_revision,
        method=method,
        quantize_start=quantize_start,
        agg=agg,
        n_scored=n_scored,
        provenance=scored_provenance(corpus, len(chunks)),
        warnings=probe_warnings,
        kl_by_depth=kl_by_depth,
        measured_bpt=measured_bpt,
        control_summary=control_summary,
        control_flip_rate=control_flip_rate,
        working_set_bytes_per_token=working_set_bytes_per_token,
        partial_coverage=partial_coverage,
    )


def measure_kv_fidelity(
    model_id: str,
    *,
    kv_bits: int = 4,
    kv_group_size: int = 64,
    method: KVCacheMethod | None = None,
    quantize_start: int = 0,
    corpus: Corpus | None = None,
    max_chunks: int | None = None,
    model_revision: str | None = None,
    chunk_length: int = 512,
    control: bool = False,
    allow_custom_code: bool = False,
    progress: ProgressFn | None = None,
) -> FidelityReport:
    """Measure how much KV-cache quantization costs, via teacher-forced paired scoring.

    Args:
        model_id: HuggingFace model ID (e.g. ``mlx-community/Llama-3.2-1B-Instruct-4bit``).
        kv_bits: KV-cache quantization bits (default 4).
        kv_group_size: KV-cache quantization group size (default 64).
        method: an explicit :class:`KVCacheMethod`; when given, ``kv_bits``/``kv_group_size``
            must stay at their defaults.
        quantize_start: 0 = stress mode (default); ``1 ≤ N ≤ chunk_length-2`` = deployment
            mode (first N positions computed with a full-precision cache, then the stored prefix
            converts too; metrics cover the post-boundary region).
        corpus: Pre-built corpus to score. If None, WikiText-2 test split is fetched (requires
            network access and the ``--run-network`` marker in tests). ``chunk_length`` must be
            left at its default when a corpus is supplied — the corpus already fixes its window.
            ``corpus.provenance.chunk_length`` is still checked against ``MAX_CHUNK_LENGTH``
            (the safety ceiling applies to every window, not only the auto-loaded one).
        max_chunks: Score at most this many corpus chunks (applies to both the auto-loaded and
            a caller-provided corpus).
        model_revision: HuggingFace model revision (commit SHA or tag).
        chunk_length: Tokens per chunk for the auto-loaded corpus (default 512). Must be in
            ``[2, MAX_CHUNK_LENGTH]`` — the upper bound is a hard safety ceiling, not a mere
            recommendation. ``MAX_CHUNK_LENGTH`` alone is vocabulary-blind, so a second,
            vocabulary-aware pre-flight also refuses any window whose paired fp32 logits
            would exceed a fraction of the installed wired cap.
        control: Run a third, quantizer-only forward per chunk so the report can separate
            quantizer error from quantized-attention-kernel numerics. With
            ``quantize_start > 0`` the control lane converts from the same full-precision
            prefix and scores only the post-boundary region. See
            :func:`~mlx_quant_fidelity.probes.kv.score_kv_config`'s ``control`` docs.

        allow_custom_code: Let a repo whose config.json names a ``model_file`` run its own
            Python code on load (off by default; mlx-lm executes that file).
        progress: Optional callback receiving `chunk i/n` lines about every tenth chunk.
            Silent when None (the default).

    Returns:
        A :class:`~mlx_quant_fidelity.report.FidelityReport` with all metrics and provenance.

    Raises:
        UntrustedModelCodeError: If the repo ships its own model code and
            ``allow_custom_code`` is False.
        QuantizeStartError: If quantize_start is out of range for the corpus window.
        CacheNotQuantizableError: If the model's KV cache does not support quantization.
        ExactZeroError: If KLD and flip rate are exactly 0 (quantization did not engage).
        CorpusError: If chunk_length is out of range, or is set together with a caller-provided
            corpus, or a caller-provided corpus's own chunk_length exceeds MAX_CHUNK_LENGTH, or
            the corpus/max_chunks combination yields no chunks.
        LogitsBudgetError: If the window x vocabulary paired-logits estimate (plus the scored
            method's and, when ``--control`` is on, the control cache's working-set bytes)
            exceeds the installed wired-memory budget. A CorpusError subclass.
    """
    from mlx_lm import load

    if corpus is not None and chunk_length != 512:
        raise CorpusError(
            "chunk_length applies to the auto-loaded corpus; the provided corpus already "
            "fixes its own window (pass corpus without chunk_length, or omit corpus and let "
            "chunk_length build one)."
        )
    if not (2 <= chunk_length <= MAX_CHUNK_LENGTH):
        raise CorpusError(
            f"chunk_length={chunk_length} must be between 2 and MAX_CHUNK_LENGTH="
            f"{MAX_CHUNK_LENGTH} (a hard safety ceiling — larger windows hold larger paired "
            "fp32 logits per chunk and risk a kernel panic — raising it requires re-validating "
            "peak memory; see docs/measurement-principles.md)."
        )
    if corpus is not None and corpus.provenance.chunk_length > MAX_CHUNK_LENGTH:
        raise CorpusError(
            f"corpus.provenance.chunk_length={corpus.provenance.chunk_length} exceeds "
            f"MAX_CHUNK_LENGTH={MAX_CHUNK_LENGTH} (a hard safety ceiling — larger windows hold "
            "larger paired fp32 logits per chunk and risk a kernel panic — raising it requires "
            "re-validating peak memory; see docs/measurement-principles.md)."
        )
    if quantize_start != 0:
        window = corpus.provenance.chunk_length if corpus is not None else chunk_length
        if not (1 <= quantize_start <= window - 2):
            raise QuantizeStartError(
                f"quantize_start={quantize_start} must be in [1, {window - 2}] "
                f"(0 = stress mode; N computes the first N of {window} positions "
                "before conversion)."
            )
    if max_chunks is not None and max_chunks < 1:
        raise CorpusError(f"max_chunks must be >= 1 (got {max_chunks}).")
    if corpus is not None and len(corpus.chunks) == 0:
        raise CorpusError("the provided corpus has no chunks; at least one is required.")
    if corpus is not None:
        _require_teacher_forced_positions(corpus.chunks)
    # Model-free method checks (bit widths, installed runtime) need no network: refuse here,
    # before the pre-load config fetch and long before the model download.
    method = _resolve_method(method, kv_bits, kv_group_size)
    check_before_load(method)
    _control_method_for(method, control)
    checked = preload_check(model_id, model_revision, allow_custom_code=allow_custom_code)
    caps_note = caps_warning(install_memory_caps())  # install must precede model load
    # Load exactly the commit that was checked, not a re-resolved branch head.
    # lazy=True: mlx-lm's eager load evaluates every parameter before returning, which would
    # make the "weights alone exceed" refusal in score_kv_config fire after they are resident.
    # model_resident_bytes reads nbytes metadata; the first forward materialises the weights.
    _loaded = load(  # pragma: no cover
        model_id, revision=checked.load_revision(model_revision), lazy=True
    )
    model, tokenizer = _loaded[0], _loaded[1]  # pragma: no cover

    if corpus is None:  # pragma: no cover
        from mlx_quant_fidelity.corpora.wikitext import load_wikitext2

        corpus = load_wikitext2(
            tokenizer, chunk_length=chunk_length, max_chunks=max_chunks, tokenizer_id=model_id
        )
        if len(corpus.chunks) == 0:
            raise CorpusError("the evaluation corpus yielded no chunks; at least one is required.")

    report = score_kv_config(
        model,
        corpus,
        model_id=model_id,
        model_revision=model_revision,
        method=method,
        quantize_start=quantize_start,
        max_chunks=max_chunks,
        control=control,
        progress=progress,
    )
    if caps_note is not None:
        report = dataclasses.replace(report, warnings=(*report.warnings, caps_note))
    return report
