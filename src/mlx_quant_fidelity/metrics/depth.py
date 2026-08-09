"""Depth-resolved aggregation: bucket per-position scalars by position depth."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DepthBucketSummary:
    """Aggregate over positions [start, end) pooled across all scored chunks."""

    start: int
    end: int
    kl_mean: float
    kl_p99: float
    n_positions: int


def bucket_by_depth(
    per_chunk_values: list[NDArray[np.float64]], *, n_buckets: int = 8
) -> tuple[DepthBucketSummary, ...]:
    """Bucket per-position values by depth, pooled across equal-length chunks.

    Buckets are equal-width over the scored positions (the last bucket absorbs the
    remainder when the count does not divide evenly). All chunks must carry the same
    number of positions; ``n_buckets`` is clamped to that count.
    """
    if not per_chunk_values:
        raise ValueError("at least one chunk is required")
    length = int(per_chunk_values[0].shape[0])
    if any(int(c.shape[0]) != length for c in per_chunk_values):
        raise ValueError("all chunks must have equal scored-position counts")
    stacked = np.stack(per_chunk_values)  # (chunks, positions)
    edges = np.linspace(0, length, min(n_buckets, length) + 1, dtype=int)
    buckets = []
    for start, end in itertools.pairwise(edges):
        vals = stacked[:, start:end].reshape(-1)
        buckets.append(
            DepthBucketSummary(
                start=int(start),
                end=int(end),
                kl_mean=float(vals.mean()),
                kl_p99=float(np.percentile(vals, 99)),
                n_positions=int(vals.size),
            )
        )
    return tuple(buckets)
