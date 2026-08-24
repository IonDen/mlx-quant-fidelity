"""A sys.modules fake of arozanov's turboquant_mlx, shaped like commit 6e928d7's verified surface.

Lets the adapter's wiring run offline. ``state`` stores uint32 packed codes + fp32 norms with the
real byte geometry so the behavioural contract (bytes == formula) is satisfiable by the fake.
"""

import sys
import types

import mlx.core as mx


def _packed_dim(head_dim, bits):
    vals = {1: 32, 2: 16, 3: 10, 4: 8}[bits]
    return -(-head_dim // vals)


class FakeTurboQuantKVCache:
    # signature matches the pinned port: (bits=3, seed=42, fused=False, sparse_v_threshold=None, v_only=False)
    def __init__(self, bits=3, seed=42, fused=False, sparse_v_threshold=None, v_only=False):
        self.quant_bits = bits
        self.seed = seed
        self.fused = fused
        self.sparse_v_threshold = sparse_v_threshold
        self.v_only = v_only
        self.offset = 0
        self._shape = None
        self.trim_calls = []
        self._k_deq_buf = None
        self._v_deq_buf = None

    def update_and_fetch(self, keys, values):
        b, h, s, d = keys.shape
        self._shape = (b, h, d)
        self.offset += s
        self._k_deq_buf = keys  # the real port retains dequant buffers until trim()
        self._v_deq_buf = values
        # "dequantize": a lossy but deterministic round-trip so KLD is non-zero downstream
        return mx.round(keys, 1), mx.round(values, 1)

    @property
    def state(self):
        if self._shape is None:
            return []
        b, h, d = self._shape
        packed = mx.zeros((b, h, self.offset, _packed_dim(d, self.quant_bits)), dtype=mx.uint32)
        norms = mx.zeros((b, h, self.offset), dtype=mx.float32)
        return [packed, norms, packed, norms]

    def trim(self, n):
        self.trim_calls.append(n)
        self._k_deq_buf = None
        self._v_deq_buf = None
        return 0


class _WrongOffsetCache(FakeTurboQuantKVCache):
    def update_and_fetch(self, keys, values):
        out = super().update_and_fetch(keys, values)
        self.offset -= 1  # under-counts: the behavioural contract must catch this
        return out


class _StickyTrimCache(FakeTurboQuantKVCache):
    """A port whose trim() records the call but leaves the dequant buffers resident."""

    def trim(self, n):
        self.trim_calls.append(n)
        return 0


class _NoDeqBuffersCache(FakeTurboQuantKVCache):
    """A port whose update_and_fetch never populates the dequant working buffers."""

    def update_and_fetch(self, keys, values):
        out = super().update_and_fetch(keys, values)
        self._k_deq_buf = None
        self._v_deq_buf = None
        return out


def _raising_property(name):
    def _get(self):
        raise AttributeError(name)

    return property(_get)


def install_fake_port(
    monkeypatch,
    *,
    version="0.3.0",
    no_version=False,
    missing=(),
    no_cache_module=False,
    with_bits_attr=False,
    wrong_behaviour=False,
    bad_trim=False,
    no_deq_buffers=False,
):
    """Install a fake ``turboquant_mlx`` into sys.modules; returns the fake cache class."""
    pkg = types.ModuleType("turboquant_mlx")
    if not no_version:
        pkg.__version__ = version
    if bad_trim:
        base = _StickyTrimCache
    elif wrong_behaviour:
        base = _WrongOffsetCache
    elif no_deq_buffers:
        base = _NoDeqBuffersCache
    else:
        base = FakeTurboQuantKVCache
    cls = type("TurboQuantKVCache", (base,), {})
    for name in missing:
        setattr(cls, name, _raising_property(name))
    if with_bits_attr:
        cls.bits = 4
    cache_mod = types.ModuleType("turboquant_mlx.cache")
    cache_mod.TurboQuantKVCache = cls
    monkeypatch.setitem(sys.modules, "turboquant_mlx", pkg)
    if no_cache_module:
        monkeypatch.delitem(sys.modules, "turboquant_mlx.cache", raising=False)
    else:
        pkg.cache = cache_mod
        monkeypatch.setitem(sys.modules, "turboquant_mlx.cache", cache_mod)
    return cls
