"""_AffineCache: affine-quantized storage, dequantize-on-fetch, standard-SDPA routing."""

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from mlx_quant_fidelity.probes.kv_methods import _AffineCache


def _kv(seed: int, s: int = 16):
    """Deterministic fp16 K/V pair shaped (1, 2, s, 64)."""
    k = mx.random.normal((1, 2, s, 64), key=mx.random.key(seed)).astype(mx.float16)
    v = mx.random.normal((1, 2, s, 64), key=mx.random.key(seed + 1)).astype(mx.float16)
    return k, v


def test_equal_bits_state_identical_to_stock_to_quantized():
    """Reds if the cache quantizes with different params/layout than stock's to_quantized."""
    k, v = _kv(0)
    base = KVCache()
    base.update_and_fetch(k, v)
    stock_q = base.to_quantized(group_size=64, bits=4)
    (sk, sv) = stock_q.state  # each side a (wq, scales, biases) triple sliced to offset
    aff = _AffineCache(k_bits=4, v_bits=4, group_size=64)
    aff.update_and_fetch(k, v)
    ak_wq, ak_s, ak_b, av_wq, av_s, av_b = aff.state
    for ours, theirs in (
        (ak_wq, sk[0]),
        (ak_s, sk[1]),
        (ak_b, sk[2]),
        (av_wq, sv[0]),
        (av_s, sv[1]),
        (av_b, sv[2]),
    ):
        assert mx.array_equal(ours, theirs)


def test_fetch_returns_dequantized_roundtrip():
    """Reds if fetch returns raw fp inputs (quantization bypassed) or the wrong side's bits."""
    k, v = _kv(2)
    aff = _AffineCache(k_bits=8, v_bits=4, group_size=64)
    k_out, v_out = aff.update_and_fetch(k, v)
    mx.eval(k_out, v_out)
    assert k_out.dtype == mx.float16
    assert v_out.dtype == mx.float16
    kq = mx.quantize(k, group_size=64, bits=8)
    assert mx.array_equal(k_out, mx.dequantize(*kq, group_size=64, bits=8))
    vq = mx.quantize(v, group_size=64, bits=4)
    assert mx.array_equal(v_out, mx.dequantize(*vq, group_size=64, bits=4))


def test_never_exposes_bits_attribute():
    """Reds if someone renames k_bits to bits — mlx-lm would silently route quantized SDPA."""
    assert not hasattr(_AffineCache(k_bits=8, v_bits=4, group_size=64), "bits")


def test_append_extends_offset_and_returns_full_history():
    """Reds if append overwrites instead of concatenating, or re-quantizes the old segment."""
    k1, v1 = _kv(4, s=8)
    k2, v2 = _kv(6, s=8)
    aff = _AffineCache(k_bits=4, v_bits=4, group_size=64)
    aff.update_and_fetch(k1, v1)
    k_out, _ = aff.update_and_fetch(k2, v2)
    assert aff.offset == 16
    assert k_out.shape[-2] == 16
    first = mx.dequantize(*mx.quantize(k1, group_size=64, bits=4), group_size=64, bits=4)
    assert mx.array_equal(k_out[..., :8, :], first)


def test_trim_drops_from_the_end():
    """Reds if trim forgets to slice the stored arrays or to update offset."""
    k, v = _kv(8)
    aff = _AffineCache(k_bits=4, v_bits=4, group_size=64)
    aff.update_and_fetch(k, v)
    assert aff.is_trimmable()
    assert aff.trim(6) == 6
    assert aff.offset == 10
    assert aff.state[0].shape[-2] == 10
