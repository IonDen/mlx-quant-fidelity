"""Property tests: invariants that must hold for every input, not just the hand-picked ones."""

import itertools
import math

import mlx.core as mx
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mlx_quant_fidelity.metrics import bucket_by_depth, kl_divergence
from mlx_quant_fidelity.ranking import RankPoint, dominates, pareto_frontier

_SETTINGS = settings(deadline=None, max_examples=200)

# Few distinct values on purpose: small grids make exact ties (equal cost / equal quality)
# common, and ties are where a `<` vs `<=` slip in `dominates` lives.
_points = st.lists(
    st.tuples(st.integers(0, 6), st.sampled_from([0.0, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0])),
    min_size=1,
    max_size=12,
).map(lambda rows: [RankPoint(f"p{i}", q, c) for i, (c, q) in enumerate(rows)])


@_SETTINGS
@given(_points)
def test_frontier_points_never_dominate_each_other(points):
    """Bug caught: a dominated point (or a mutually-dominating pair) survives on the frontier."""
    front = [p for p in points if p.label in set(pareto_frontier(points))]
    assert front, "a non-empty set always has a non-dominated point"
    assert not any(dominates(a, b) for a in front for b in front)


@_SETTINGS
@given(_points)
def test_every_off_frontier_point_is_dominated_by_a_frontier_point(points):
    """Bug caught: a point is dropped from the frontier without any frontier point beating it."""
    on = set(pareto_frontier(points))
    front = [p for p in points if p.label in on]
    for p in points:
        if p.label not in on:
            assert any(dominates(f, p) for f in front)


@_SETTINGS
@given(_points, st.data())
def test_frontier_is_invariant_under_input_order(points, data):
    """Bug caught: the frontier (or its order) depends on the order targets were passed in."""
    shuffled = data.draw(st.permutations(points))
    assert pareto_frontier(list(shuffled)) == pareto_frontier(points)


@_SETTINGS
@given(_points)
def test_frontier_matches_an_independent_pareto_definition(points):
    """Bug caught: `dominates` itself is wrong (e.g. ignores an axis). The two properties above
    only check the frontier against `dominates`; this one checks it against the textbook
    definition written out inline: p is dropped iff some o is no worse on BOTH axes and strictly
    better on at least one."""

    def beaten(p):
        return any(
            o.cost_bytes <= p.cost_bytes
            and o.quality <= p.quality
            and (o.cost_bytes < p.cost_bytes or o.quality < p.quality)
            for o in points
        )

    assert set(pareto_frontier(points)) == {p.label for p in points if not beaten(p)}


@st.composite
def _chunks(draw):
    n_chunks = draw(st.integers(1, 3))
    length = draw(st.integers(1, 64))
    vals = st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False)
    return [
        np.array(draw(st.lists(vals, min_size=length, max_size=length)), dtype=np.float64)
        for _ in range(n_chunks)
    ]


@_SETTINGS
@given(_chunks(), st.integers(1, 12))
def test_depth_buckets_partition_positions_and_match_slice_means(chunks, n_buckets):
    """Bug caught: a position is dropped or double-counted, or a bucket mean is not the mean of
    exactly the pooled values in its [start, end) span."""
    length = chunks[0].shape[0]
    buckets = bucket_by_depth(chunks, n_buckets=n_buckets)
    assert sum(b.n_positions for b in buckets) == len(chunks) * length
    assert buckets[0].start == 0
    assert buckets[-1].end == length
    assert all(a.end == b.start for a, b in itertools.pairwise(buckets))
    stacked = np.stack(chunks)
    for b in buckets:
        assert math.isclose(
            b.kl_mean, float(stacked[:, b.start : b.end].mean()), rel_tol=1e-9, abs_tol=1e-12
        )


@st.composite
def _logit_pair(draw):
    positions = draw(st.integers(1, 6))
    vocab = draw(st.integers(2, 16))
    vals = st.floats(-20.0, 20.0, width=32, allow_nan=False, allow_infinity=False)
    flat = st.lists(vals, min_size=positions * vocab, max_size=positions * vocab)
    ref = np.array(draw(flat), dtype=np.float32).reshape(positions, vocab)
    quant = np.array(draw(flat), dtype=np.float32).reshape(positions, vocab)
    return ref, quant


@_SETTINGS
@given(_logit_pair())
def test_kl_is_non_negative_and_finite(pair):
    """Bug caught: a wrong sign / direction / missing normalization yields negative or non-finite
    KL on finite logits. The floor is not 0 because log_softmax subtracts a logsumexp that reaches
    ~23 for logits in [-20, 20] over 16 tokens; one fp32 ulp there is 2^-19 ~ 1.9e-6, and the
    p-weighted sum of per-token log differences can be off by a few such ulps (-2.8e-6 measured on
    exactly shifted pairs, whose true KL is 0). Five ulps, ~1e-5, bounds that."""
    ref, quant = pair
    kl = np.array(kl_divergence(mx.array(ref), mx.array(quant)))
    assert np.all(np.isfinite(kl))
    assert kl.min() >= -1e-5


def test_default_hypothesis_profile_is_derandomized():
    """Bug: a randomized profile on these tests lets a rare draw fail one CI run and not the next.
    Checks the profile registered in conftest (the tests' @given picks it up when no explicit
    --hypothesis-profile overrides it), not the local _SETTINGS decorator."""
    assert settings.get_profile("derandomized").derandomize is True
