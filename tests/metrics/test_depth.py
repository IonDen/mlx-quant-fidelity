import numpy as np
import pytest

from mlx_quant_fidelity.metrics import bucket_by_depth


def test_rising_drift_lands_in_later_buckets():
    # 2 chunks, 8 positions, values == position index. Buckets of 2:
    # [0,2)->mean 0.5, [2,4)->2.5, [4,6)->4.5, [6,8)->6.5 (hand-derived).
    chunks = [np.arange(8, dtype=np.float64), np.arange(8, dtype=np.float64)]
    buckets = bucket_by_depth(chunks, n_buckets=4)
    assert [b.kl_mean for b in buckets] == [0.5, 2.5, 4.5, 6.5]
    assert [(b.start, b.end) for b in buckets] == [(0, 2), (2, 4), (4, 6), (6, 8)]
    assert all(b.n_positions == 4 for b in buckets)  # 2 positions x 2 chunks


def test_rising_drift_with_differing_chunks_pools_not_averages():
    # 2 chunks with DIFFERENT values (not two copies of the same chunk): chunk1 = [0..7],
    # chunk2 = [100..107]. Because mean is linear, per-chunk-averaged-then-combined means would
    # happen to match the pooled mean here too — but p99 is NOT linear, so a per-chunk-averaged
    # implementation (compute each chunk's bucket p99, then average across chunks) would diverge
    # from the correct pooled-then-percentile value. Hand-derived for bucket [0,2):
    # pooled values sorted [0,1,100,101], p99 index = 0.99*3 = 2.97 -> 100 + 0.97*(101-100) =
    # 100.97. A wrong per-chunk-average would give (0.99 + 100.99) / 2 = 50.99 instead.
    chunk1 = np.arange(8, dtype=np.float64)
    chunk2 = np.arange(8, dtype=np.float64) + 100
    buckets = bucket_by_depth([chunk1, chunk2], n_buckets=4)
    assert [b.kl_mean for b in buckets] == [50.5, 52.5, 54.5, 56.5]
    assert [b.kl_p99 for b in buckets] == [
        pytest.approx(100.97),
        pytest.approx(102.97),
        pytest.approx(104.97),
        pytest.approx(106.97),
    ]
    assert all(b.n_positions == 4 for b in buckets)


def test_flat_drift_gives_equal_buckets():
    # A constant-near-zero implementation must FAIL the rising test above;
    # this one pins the flat case: every bucket mean identical.
    chunks = [np.full(8, 0.25)]
    buckets = bucket_by_depth(chunks, n_buckets=4)
    assert all(b.kl_mean == 0.25 for b in buckets)


def test_p99_within_bucket():
    # bucket [0,4) over values [0,0,0,100]: p99 = 97.0 (numpy linear interpolation,
    # hand-derived: sorted [0,0,0,100], p99 index = 0.99*3 = 2.97 -> 0 + 0.97*100).
    buckets = bucket_by_depth([np.array([0.0, 0.0, 0.0, 100.0])], n_buckets=1)
    assert buckets[0].kl_p99 == pytest.approx(97.0)


def test_n_buckets_clamped_to_positions():
    buckets = bucket_by_depth([np.array([1.0, 2.0])], n_buckets=8)
    assert len(buckets) == 2


def test_unequal_chunk_lengths_raise():
    with pytest.raises(ValueError, match="equal"):
        bucket_by_depth([np.zeros(4), np.zeros(5)])


def test_empty_input_raises():
    with pytest.raises(ValueError, match="at least one"):
        bucket_by_depth([])
