"""Teacher-forced, streaming KV-quant fidelity probe."""

from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING

import mlx.core as mx
from mlx_lm.models.cache import QuantizedKVCache, make_prompt_cache

from mlx_quant_fidelity._memory_caps import device_string, install_memory_caps
from mlx_quant_fidelity.errors import (
    CacheNotQuantizableError,
    CorpusError,
    QuantizeStartError,
)
from mlx_quant_fidelity.policy import verdict_for
from mlx_quant_fidelity.probes._paired import _aggregate_chunks, _check_exact_zero, _reduce_pair
from mlx_quant_fidelity.report import FidelityReport

if TYPE_CHECKING:
    from mlx_quant_fidelity.corpora.provenance import Corpus


def _kv_head_dim(model: object) -> int | None:
    """Best-effort per-head KV dim for the group-size gate. None if not derivable.

    ``ModelArgs.head_dim`` is frequently None (llama populates the real value as an
    Attention-layer local, not on args), so the hidden//heads fallback is the primary
    path. Field names are llama-family-specific; unusual archs return None.
    """
    args = getattr(model, "args", None)
    head_dim = getattr(args, "head_dim", None)
    if head_dim:
        return int(head_dim)
    hidden = getattr(args, "hidden_size", None)
    heads = getattr(args, "num_attention_heads", None)
    if hidden and heads:
        return int(hidden) // int(heads)
    return None


def _head_dim_gate(*, head_dim: int | None, kv_group_size: int, model_type: str) -> str | None:
    """Validate kv_group_size divides the per-head KV dim. Raise if it doesn't; warn if unknown.

    The quantity is the per-head head_dim (QuantizedKVCache quantizes the per-head last axis;
    mlx_lm/models/cache.py). Returns a warning string when head_dim can't be derived (do not
    silently pass on odd architectures), else None.
    """
    if head_dim is None:
        return (
            f"head_dim/kv_group_size compatibility unverified for '{model_type}'; "
            "relying on MLX to surface a mismatch at first use."
        )
    if head_dim % kv_group_size != 0:
        raise CacheNotQuantizableError(
            f"kv_group_size={kv_group_size} does not divide the model's KV head_dim={head_dim}; "
            f"choose a group size that divides {head_dim} (e.g. 32 or 64)."
        )
    return None


def packed_width_mismatch(head_dim: int, bits: int) -> bool:
    """True when mlx-lm's QuantizedKVCache pre-allocation disagrees with mx.quantize.

    The cache pre-allocates packed buffers of width ``head_dim // (32 // bits)``
    (el_per_int truncation, mlx-lm models/cache.py) while ``mx.quantize`` packs to
    ``head_dim * bits // 32``; a first append with mismatched widths dies in
    broadcast_shapes. Affects bits=6 at e.g. head_dim=128 on mlx-lm 0.31.x.
    """
    return head_dim // (32 // bits) != head_dim * bits // 32


def _cache_is_quantizable(cache: list[object], *, group_size: int, bits: int) -> bool:
    """Return True if every layer cache has a NON-RAISING to_quantized; else raise, naming the type.

    Validates ``bits`` up front (MLX only supports 2/3/4/6/8). An attribute check alone is
    insufficient: RotatingKVCache / BatchRotatingKVCache (sliding-window) HAVE to_quantized but
    raise NotImplementedError. So probe by actually calling it on the (empty, cheap) cache.
    """
    if bits not in (2, 3, 4, 6, 8):
        raise CacheNotQuantizableError(f"unsupported kv_bits={bits}; MLX supports 2/3/4/6/8.")
    for layer in cache:
        to_q = getattr(layer, "to_quantized", None)
        if to_q is None:
            raise CacheNotQuantizableError(
                f"cache layer {type(layer).__name__} has no to_quantized; "
                "this model's KV cache cannot be quantized (e.g. sliding-window / MLA)."
            )
        try:
            to_q(group_size=group_size, bits=bits)
        except NotImplementedError as exc:
            raise CacheNotQuantizableError(
                f"cache layer {type(layer).__name__} declares to_quantized but it is NYI: {exc}"
            ) from exc
        except (ValueError, RuntimeError) as exc:
            raise CacheNotQuantizableError(
                f"cache layer {type(layer).__name__} cannot quantize at "
                f"group_size={group_size}, bits={bits}: {exc}"
            ) from exc
    return True


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


def _score_chunk_deployment(
    model: object,
    ids: mx.array,
    ref_cache: list[object],
    quant_cache: list[object],
    *,
    quantize_start: int,
    group_size: int,
    bits: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Deployment split: compute the prefix in full precision, then convert the stored cache.

    Returns per-position (kl, flips, ref_nll, quant_nll) over ALL L-1 prediction positions.
    `quant_cache` starts full-precision (make_prompt_cache) and is converted to quantized at the
    boundary via `to_quantized` (mirrors mlx-lm's maybe_quantize_kv_cache). The caller aggregates
    only [quantize_start:] for the reported metrics.
    """
    n = quantize_start
    targets = ids[1:]
    ref_logits = model(ids[None, :-1], cache=ref_cache)[0].astype(mx.float32)  # type: ignore[operator]
    mx.eval(ref_logits)
    # Segment 1: prefix [0:n) through the full-precision quant_cache (identical to ref there).
    seg1 = model(ids[None, :n], cache=quant_cache)[0].astype(mx.float32)  # type: ignore[operator]
    kl1, flip1, refnll1, qnll1 = _reduce_pair(ref_logits[:n], seg1, targets[:n])
    mx.eval(kl1, flip1, refnll1, qnll1)
    del seg1  # free the prefix logits before the boundary + seg2 forward (one segment live)
    mx.eval([c.state for c in quant_cache])  # type: ignore[attr-defined]  # collapse seg-1 graph before boundary
    # Boundary: convert each layer cache (quantizes the stored [0:n) prefix).
    for i in range(len(quant_cache)):
        quant_cache[i] = quant_cache[i].to_quantized(group_size=group_size, bits=bits)  # type: ignore[attr-defined]
    # Segment 2: [n:L-1) through the now-quantized cache. NOTE ids[n:-1], not ids[n:].
    seg2 = model(ids[None, n:-1], cache=quant_cache)[0].astype(mx.float32)  # type: ignore[operator]
    kl2, flip2, refnll2, qnll2 = _reduce_pair(ref_logits[n:], seg2, targets[n:])
    mx.eval(kl2, flip2, refnll2, qnll2)
    return (
        mx.concatenate([kl1, kl2]),
        mx.concatenate([flip1, flip2]),
        mx.concatenate([refnll1, refnll2]),
        mx.concatenate([qnll1, qnll2]),
    )


def score_kv_config(
    model: object,
    corpus: Corpus,
    *,
    model_id: str,
    model_revision: str | None = None,
    kv_bits: int = 4,
    kv_group_size: int = 64,
    quantize_start: int = 0,
    max_chunks: int | None = None,
) -> FidelityReport:
    """Score one KV config on an ALREADY-LOADED model (no load, no caps install).

    Shared by ``measure_kv_fidelity`` (load -> delegate) and the KV ``compare`` adapter
    (load once -> loop configs). Applies ``max_chunks`` to the provided corpus,
    so a caller-supplied corpus is capped identically to the weight probe.
    """
    probe_warnings: list[str] = []
    model_type = str(getattr(getattr(model, "args", None), "model_type", "unknown"))
    head_dim = _kv_head_dim(model)
    head_dim_warning = _head_dim_gate(
        head_dim=head_dim, kv_group_size=kv_group_size, model_type=model_type
    )
    if head_dim_warning is not None:
        probe_warnings.append(head_dim_warning)
    if kv_bits not in (2, 3, 4, 6, 8):
        raise CacheNotQuantizableError(f"unsupported kv_bits={kv_bits}; MLX supports 2/3/4/6/8.")
    if head_dim is not None and packed_width_mismatch(head_dim, kv_bits):
        raise CacheNotQuantizableError(
            f"kv_bits={kv_bits} cannot append to a fresh QuantizedKVCache at "
            f"head_dim={head_dim} on this mlx-lm version (packed-width truncation bug); "
            "use bits 2/3/4/8 or a group-compatible head_dim."
        )

    probe_cache = make_prompt_cache(model)
    n_layers = len(probe_cache)
    _cache_is_quantizable(probe_cache, group_size=kv_group_size, bits=kv_bits)
    del probe_cache

    mode = "stress" if quantize_start == 0 else "deployment"
    chunks = corpus.chunks[:max_chunks] if max_chunks is not None else corpus.chunks
    kls: list[mx.array] = []
    flips: list[mx.array] = []
    ref_nlls: list[mx.array] = []
    quant_nlls: list[mx.array] = []
    n_scored = 0
    for ids in chunks:
        if quantize_start > 0 and int(ids.size) < quantize_start + 2:
            continue  # too short to have a post-boundary position; skip
        ref_cache = make_prompt_cache(model)
        if quantize_start == 0:
            quant_cache: list[object] = [
                QuantizedKVCache(group_size=kv_group_size, bits=kv_bits) for _ in range(n_layers)
            ]
            try:
                kl, flip, ref_nll, quant_nll = _score_chunk(model, ids, ref_cache, quant_cache)
            except ValueError as exc:
                if "broadcast_shapes" in str(exc):
                    raise CacheNotQuantizableError(
                        f"kv_bits={kv_bits} crashed appending to a fresh QuantizedKVCache "
                        f"(mlx-lm packed-width truncation bug): {exc}"
                    ) from exc
                raise
        else:
            quant_cache = make_prompt_cache(model)
            kl, flip, ref_nll, quant_nll = _score_chunk_deployment(
                model,
                ids,
                ref_cache,
                quant_cache,
                quantize_start=quantize_start,
                group_size=kv_group_size,
                bits=kv_bits,
            )
            kl, flip = kl[quantize_start:], flip[quantize_start:]
            ref_nll, quant_nll = ref_nll[quantize_start:], quant_nll[quantize_start:]
        mx.eval(kl, flip, ref_nll, quant_nll)
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
    return FidelityReport(
        model_id=model_id,
        model_revision=model_revision,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantize_start=quantize_start,
        quantize_mode=mode,
        kl=agg.kl,
        flip_rate=agg.flip_rate,
        perplexity_ref=agg.perplexity_ref,
        perplexity_quant=agg.perplexity_quant,
        perplexity_delta=agg.perplexity_quant - agg.perplexity_ref,
        n_positions=agg.n_positions,
        n_chunks=n_scored,
        corpus=corpus.provenance,
        mlx_version=importlib.metadata.version("mlx"),
        mlx_lm_version=importlib.metadata.version("mlx-lm"),
        peak_memory_bytes=int(mx.get_peak_memory()),
        cache_supported=True,
        verdict=verdict_for(agg.kl.mean, agg.kl.p99, agg.flip_rate),
        warnings=tuple(probe_warnings),
        device=device_string(),
    )


def measure_kv_fidelity(
    model_id: str,
    *,
    kv_bits: int = 4,
    kv_group_size: int = 64,
    quantize_start: int = 0,
    corpus: Corpus | None = None,
    max_chunks: int | None = None,
    model_revision: str | None = None,
) -> FidelityReport:
    """Measure how much KV-cache quantization costs, via teacher-forced paired scoring.

    Args:
        model_id: HuggingFace model ID (e.g. ``mlx-community/Llama-3.2-1B-Instruct-4bit``).
        kv_bits: KV-cache quantization bits (default 4).
        kv_group_size: KV-cache quantization group size (default 64).
        quantize_start: 0 = stress mode (default); ``1 ≤ N ≤ chunk_length-2`` = deployment
            mode (first N positions computed with a full-precision cache, then the stored prefix
            converts too; metrics cover the post-boundary region).
        corpus: Pre-built corpus to score. If None, WikiText-2 test split is fetched (requires
            network access and the ``--run-network`` marker in tests).
        max_chunks: Score at most this many corpus chunks (applies to both the auto-loaded and
            a caller-provided corpus).
        model_revision: HuggingFace model revision (commit SHA or tag).

    Returns:
        A :class:`~mlx_quant_fidelity.report.FidelityReport` with all metrics and provenance.

    Raises:
        QuantizeStartError: If quantize_start is out of range for the corpus window.
        CacheNotQuantizableError: If the model's KV cache does not support quantization.
        ExactZeroError: If KLD and flip rate are exactly 0 (quantization did not engage).
    """
    from mlx_lm import load

    if quantize_start != 0:
        window = corpus.provenance.chunk_length if corpus is not None else 512  # wikitext default
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
    install_memory_caps()  # must precede model load
    _loaded = load(model_id, revision=model_revision)  # pragma: no cover
    model, tokenizer = _loaded[0], _loaded[1]  # pragma: no cover

    if corpus is None:  # pragma: no cover
        from mlx_quant_fidelity.corpora.wikitext import load_wikitext2

        corpus = load_wikitext2(tokenizer, max_chunks=max_chunks, tokenizer_id=model_id)
        if len(corpus.chunks) == 0:
            raise CorpusError("the evaluation corpus yielded no chunks; at least one is required.")

    return score_kv_config(  # pragma: no cover
        model,
        corpus,
        model_id=model_id,
        model_revision=model_revision,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantize_start=quantize_start,
        max_chunks=max_chunks,
    )
