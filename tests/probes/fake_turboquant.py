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


class _FakeVTq:
    """Stand-in for ``VOnlyTurboQuantCache._v_tq`` (an inner ``TurboQuantKVCache``, v_only=True)."""

    def __init__(self, bits):
        self.quant_bits = bits
        self._v_deq_buf = None


class FakeVOnlyTurboQuantCache:
    """Fakes ``turboquant_mlx.v_only_cache.VOnlyTurboQuantCache`` at the pinned commit's shape.

    ``state`` returns 4 arrays: fp16 K, fp16 V (the port's unused duplicate, stored only because
    K reuses a plain ``KVCache`` that always stores both sides), packed V, fp32 norms — the same
    4-array shape :func:`turboquant_bytes_per_token`'s v-only sibling formula counts.
    """

    # signature matches the pinned port: (bits=3, seed=42, no_v_buffer=False)
    def __init__(self, bits=3, seed=42, no_v_buffer=False):
        self.v_bits = bits
        self.seed = seed
        self.no_v_buffer = no_v_buffer
        self._offset_value = 0
        self._shape = None
        self._dtype = None  # the real port's inner KVCache stores whatever dtype it's fed
        self.trim_calls = []
        self._v_tq = _FakeVTq(bits)

    @property
    def offset(self):
        # The real port exposes `offset` as a class-level property too (`return
        # self._k_cache.offset`) -- a required-attrs guard that checks `hasattr(cls, "offset")`
        # needs the fake's shape to match, not just its instance-level behavior (task-6 F1).
        return self._offset_value

    def update_and_fetch(self, keys, values):
        b, h, s, d = keys.shape
        self._shape = (b, h, d)
        self._dtype = keys.dtype
        self._offset_value += s
        self._v_tq._v_deq_buf = values  # the real port retains a dequant buffer until trim()
        return mx.round(keys, 1), mx.round(values, 1)

    @property
    def state(self):
        if self._shape is None:
            # Mirrors the REAL port's bug: VOnlyTurboQuantCache.state does
            # `list(self._k_cache.state) if self._k_cache.state else []`, and mlx-lm's
            # KVCache.state dereferences `self.keys.shape` unconditionally -- on a fresh
            # (empty) cache `keys` is None, so this raises before the `else []` can ever
            # apply. A guard that probes `hasattr(instance, "state")` swallows this raise
            # and misreports the genuine port as missing the attribute (task-6 F1).
            raise AttributeError("'NoneType' object has no attribute 'shape'")
        b, h, d = self._shape
        # K and the unused duplicate V are stored in whatever dtype update_and_fetch received
        # (the real port's inner plain KVCache does the same) -- NOT hardcoded fp16, so a
        # probe that feeds fp32 zeros desyncs from bytes_per_token's fp16 assumption and this
        # fake catches it, matching what direct verification against the real port found.
        k = mx.zeros((b, h, self.offset, d), dtype=self._dtype)
        v_dup = mx.zeros((b, h, self.offset, d), dtype=self._dtype)
        packed = mx.zeros(
            (b, h, self.offset, _packed_dim(d, self._v_tq.quant_bits)), dtype=mx.uint32
        )
        norms = mx.zeros((b, h, self.offset), dtype=mx.float32)
        return [k, v_dup, packed, norms]

    def trim(self, n):
        self.trim_calls.append(n)
        self._v_tq._v_deq_buf = None
        return 0


class _VOnlyWrongOffsetCache(FakeVOnlyTurboQuantCache):
    def update_and_fetch(self, keys, values):
        # Explicit unbound-method call, not zero-arg super(): install_fake_port lifts this
        # function into a flattened, unrelated class namespace (see _flatten_class_dict) whose
        # MRO does not include _VOnlyWrongOffsetCache, so the `__class__`-cell zero-arg super()
        # form would raise (self is not an instance of the closed-over class).
        out = FakeVOnlyTurboQuantCache.update_and_fetch(self, keys, values)
        self._offset_value -= 1  # under-counts: the behavioural contract must catch this
        return out


class _VOnlyStickyTrimCache(FakeVOnlyTurboQuantCache):
    """A port whose trim() records the call but leaves the V dequant buffer resident."""

    def trim(self, n):
        self.trim_calls.append(n)
        return 0


class _VOnlyNoDeqBufferCache(FakeVOnlyTurboQuantCache):
    """A port whose update_and_fetch never populates the V dequant working buffer."""

    def update_and_fetch(self, keys, values):
        out = FakeVOnlyTurboQuantCache.update_and_fetch(self, keys, values)  # see note above
        self._v_tq._v_deq_buf = None
        return out


def _flatten_class_dict(cls):
    """Every class-level attribute ``cls`` exposes (own + inherited via its MRO).

    Most-derived definition wins. Used to build a "genuinely missing an attribute" fake for
    ``TurboQuantVOnlyKVMethod``: its ``_cache_cls`` checks required attrs at the CLASS level
    (``hasattr(cls, attr)``, task-6 F1), so a raising ``property`` (the trick used for the
    uniform method below, which still checks at the instance level) would NOT simulate
    "missing" there -- the property descriptor is still present on the class either way.
    """
    merged: dict[str, object] = {}
    for klass in reversed(cls.__mro__):
        if klass is object:
            continue
        merged.update(vars(klass))
    return merged


def install_fake_port(
    monkeypatch,
    *,
    version="0.3.0",
    no_version=False,
    missing=(),
    no_cache_module=False,
    no_vonly_module=False,
    with_bits_attr=False,
    wrong_behaviour=False,
    bad_trim=False,
    no_deq_buffers=False,
):
    """Install a fake ``turboquant_mlx`` into sys.modules; returns the fake uniform cache class.

    Also installs a fake ``turboquant_mlx.v_only_cache`` module (mirroring the ``cache`` module
    below), built from the same failure-mode flags, so ``TurboQuantVOnlyKVMethod._cache_cls``'s
    failure paths test offline too. Fetch the fake V-only class via
    ``sys.modules["turboquant_mlx.v_only_cache"].VOnlyTurboQuantCache``.
    """
    pkg = types.ModuleType("turboquant_mlx")
    if not no_version:
        pkg.__version__ = version
    if bad_trim:
        base = _StickyTrimCache
        vonly_base = _VOnlyStickyTrimCache
    elif wrong_behaviour:
        base = _WrongOffsetCache
        vonly_base = _VOnlyWrongOffsetCache
    elif no_deq_buffers:
        base = _NoDeqBuffersCache
        vonly_base = _VOnlyNoDeqBufferCache
    else:
        base = FakeTurboQuantKVCache
        vonly_base = FakeVOnlyTurboQuantCache
    cls = type("TurboQuantKVCache", (base,), {})
    # vonly_cls is built from a flattened, `missing`-filtered namespace (see
    # _flatten_class_dict) rather than a subclass + raising-property, so `missing` attributes
    # are genuinely absent from the class -- required for TurboQuantVOnlyKVMethod's
    # class-level required-attrs check to see them as missing.
    vonly_namespace = {k: v for k, v in _flatten_class_dict(vonly_base).items() if k not in missing}
    vonly_cls = type("VOnlyTurboQuantCache", (object,), vonly_namespace)
    for name in missing:
        setattr(cls, name, _raising_property(name))
    if with_bits_attr:
        cls.bits = 4
        vonly_cls.bits = 4
    cache_mod = types.ModuleType("turboquant_mlx.cache")
    cache_mod.TurboQuantKVCache = cls
    vonly_mod = types.ModuleType("turboquant_mlx.v_only_cache")
    vonly_mod.VOnlyTurboQuantCache = vonly_cls
    monkeypatch.setitem(sys.modules, "turboquant_mlx", pkg)
    if no_cache_module:
        monkeypatch.delitem(sys.modules, "turboquant_mlx.cache", raising=False)
    else:
        pkg.cache = cache_mod
        monkeypatch.setitem(sys.modules, "turboquant_mlx.cache", cache_mod)
    if no_vonly_module:
        monkeypatch.delitem(sys.modules, "turboquant_mlx.v_only_cache", raising=False)
    else:
        pkg.v_only_cache = vonly_mod
        monkeypatch.setitem(sys.modules, "turboquant_mlx.v_only_cache", vonly_mod)
    return cls
