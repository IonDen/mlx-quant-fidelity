"""Shared paired-scoring helpers for the fidelity probes (model-agnostic)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_quant_fidelity.errors import ExactZeroError, NonFiniteMetricError
from mlx_quant_fidelity.metrics import ScalarSummary, perplexity, summarize
from mlx_quant_fidelity.metrics.flip import top_token_flips
from mlx_quant_fidelity.metrics.kl import kl_divergence
from mlx_quant_fidelity.metrics.perplexity import token_nll

ProgressFn = Callable[[str], None]


def report_chunk_progress(progress: ProgressFn | None, index: int, total: int) -> None:
    """Emit ``chunk {index}/{total}`` (1-based) about every tenth chunk and on the last one."""
    if progress is not None and (index % max(1, total // 10) == 0 or index == total):
        progress(f"chunk {index}/{total}")


@dataclass(frozen=True, slots=True)
class _Aggregate:
    """Reduced result across all scored chunks."""

    kl: ScalarSummary
    flip_rate: float
    perplexity_ref: float
    perplexity_quant: float
    n_positions: int


def _reduce_pair(
    ref_logits: mx.array, quant_logits: mx.array, targets: mx.array
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Reduce two fp32 logit arrays [positions, vocab] on identical targets to per-position scalars.

    Returns (kl, flips, ref_nll, quant_nll). Holds no logits beyond its own frame; the caller
    must mx.eval the returns and let the vocab-wide logits leave scope before the next chunk.
    """
    kl = kl_divergence(ref_logits, quant_logits)
    flips = top_token_flips(ref_logits, quant_logits)
    ref_nll = token_nll(ref_logits, targets)
    quant_nll = token_nll(quant_logits, targets)
    return kl, flips, ref_nll, quant_nll


def _require_finite(name: str, values: np.ndarray, *, allow_inf: bool = False) -> None:
    """Raise NonFiniteMetricError on NaN and -inf, and on +inf unless ``allow_inf``.

    A per-position KL of +inf is the documented zero-probability policy, so KL passes
    ``allow_inf=True``; -inf is never legal. NLLs must be fully finite (they feed perplexity).
    """
    arr = np.asarray(values, dtype=np.float64)
    bad = (np.isnan(arr) | np.isneginf(arr)).any() if allow_inf else not np.isfinite(arr).all()
    if bad:
        raise NonFiniteMetricError(
            f"{name} contains non-finite values (NaN"
            + (" or -inf" if allow_inf else " or inf")
            + "); a NaN would pass every threshold and could be ranked as a real "
            "measurement. Likely an fp16 overflow in the forward pass or a third-party kernel."
        )


def _aggregate_chunks(
    kls: list[mx.array],
    flips: list[mx.array],
    ref_nlls: list[mx.array],
    quant_nlls: list[mx.array],
) -> _Aggregate:
    """Concatenate per-chunk per-position scalars and reduce (host-side numpy)."""
    kl_all = np.concatenate([np.asarray(k, dtype=np.float64) for k in kls])
    flip_all = np.concatenate([np.asarray(f.astype(mx.float32), dtype=np.float64) for f in flips])
    ref_all = mx.concatenate(ref_nlls)
    quant_all = mx.concatenate(quant_nlls)
    _require_finite("per-position KL", kl_all, allow_inf=True)
    _require_finite("reference NLL", np.asarray(ref_all.astype(mx.float32)))
    _require_finite("quantized NLL", np.asarray(quant_all.astype(mx.float32)))
    return _Aggregate(
        kl=summarize(kl_all),
        flip_rate=float(flip_all.mean()),
        perplexity_ref=perplexity(ref_all),
        perplexity_quant=perplexity(quant_all),
        n_positions=int(kl_all.size),
    )


def _check_exact_zero(*, kl_mean: float, flip_rate: float, context: str) -> None:
    """Raise ExactZeroError (with a caller-supplied context message) if KLD and flip are both 0."""
    if kl_mean == 0.0 and flip_rate == 0.0:
        raise ExactZeroError(
            f"KLD and flip were exactly 0 — {context}. This is never a silent 'perfect fidelity'."
        )
