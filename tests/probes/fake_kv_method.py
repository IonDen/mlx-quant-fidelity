"""A FakeKVMethod + caches for offline seam tests. Returns objects that produce a hand-computable signal."""

from contextlib import nullcontext

import mlx.core as mx


class FakeQuantCache:
    """Marks the quantized path: the fake model bumps its logit peak when it sees ``.marker``."""

    marker = True
    offset = 4  # a fixed stored-token count so measured bytes / offset is hand-computable

    @property
    def state(self):
        return (mx.zeros((1, 1, 4, 8), dtype=mx.uint32), mx.zeros((1, 1, 4), dtype=mx.float32))


class FakeFullCache:
    """Full-precision stand-in: no ``.marker`` -> the fake model keeps its reference peak."""

    @property
    def state(self):
        return ()


class FakeKVMethod:
    """Drives score_kv_config with no MLX cache code; records what the probe asked of it."""

    def __init__(self, *, convert_noop: bool = False, bytes_per_chunk: int = 144):
        self.convert_noop = convert_noop
        self.bytes_per_chunk = bytes_per_chunk
        self.calls: list[str] = []

    @property
    def name(self):
        return "fake"

    @property
    def label(self):
        return "fake:1"

    @property
    def params(self):
        return {"bits": 1}

    def check(self, *, head_dim, model_type):
        self.calls.append("check")
        return []

    def probe_capability(self, empty_cache):
        self.calls.append("probe_capability")

    def make_cache(self, *, n_layers):
        self.calls.append("make_cache")
        return [FakeQuantCache() for _ in range(n_layers)]

    def convert_prefix(self, fp_cache):
        self.calls.append("convert_prefix")
        if self.convert_noop:
            return list(fp_cache)
        return [FakeQuantCache() for _ in fp_cache]

    def guard(self):
        self.calls.append("guard")
        return nullcontext()

    def bytes_per_token(self, *, n_layers, n_kv_heads, head_dim):
        return 100 * n_layers

    def measured_bytes(self, cache):
        # default: 4 tokens * (8 words * 4 B + 4 B norm) = 144 B per layer; grows per call so a
        # "measure every chunk / keep the last" bug changes the recorded number
        self.calls.append("measured_bytes")
        n = self.calls.count("measured_bytes")
        return self.bytes_per_chunk * n * len(cache)

    def provenance(self):
        return {"package": "fake", "version": "0"}

    def report_warnings(self):
        return ["fake method note"]


class FakeMethodModel:
    """Peak on token 0 for a cache without ``.marker``; peak on token 1 with it."""

    def __init__(self, head_dim=64, kv_heads=None):
        self.args = type(
            "A",
            (),
            {
                "model_type": "llama",
                "head_dim": head_dim,
                "hidden_size": None,
                "num_attention_heads": None,
                "num_key_value_heads": kv_heads,
                "vocab_size": 3,
            },
        )()

    def __call__(self, inp, cache=None):
        bump = 1 if (cache is not None and getattr(cache[0], "marker", False)) else 0
        out = mx.zeros((1, inp.shape[1], 3))
        out[:, :, bump] = 5.0
        return out
