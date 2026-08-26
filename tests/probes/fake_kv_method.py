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


class FakeControlCache:
    """Control-lane stand-in: distinct marker so the fake model perturbs it separately."""

    control_marker = True
    offset = 4

    @property
    def state(self):
        """Empty state — the fake model never reads cache contents."""
        return ()


class FakeControlMethod:
    """What a stock-like FakeKVMethod's control_method() returns."""

    name = "fake-control"
    label = "fake-control"

    def make_cache(self, *, n_layers):
        """Fresh control caches."""
        return [FakeControlCache() for _ in range(n_layers)]

    def convert_prefix(self, fp_cache):
        """Deployment boundary stand-in."""
        return [FakeControlCache() for _ in fp_cache]


class FakeKVMethod:
    """Drives score_kv_config with no MLX cache code; records what the probe asked of it."""

    def __init__(
        self,
        *,
        convert_noop: bool = False,
        bytes_per_chunk: int = 144,
        stock_like: bool = False,
    ):
        self.convert_noop = convert_noop
        self.bytes_per_chunk = bytes_per_chunk
        self.calls: list[str] = []
        self._stock_like = stock_like
        if stock_like:

            def _control():
                self.calls.append("control_method")
                return FakeControlMethod()

            self.control_method = _control

    @property
    def name(self):
        return "stock" if self._stock_like else "fake"

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

    def working_set_bytes(self, *, window, n_layers, n_kv_heads, head_dim, dtype_bytes):
        """No working-set model for the fake method; the seam's division must still handle 0."""
        return 0

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
    """Peak on token 0 with no marker; token 1 with ``.marker``; control-lane logic with
    ``.control_marker`` (uniform ``control_gain`` at ``control_peak``, or position-keyed
    when ``control_gain`` is the sentinel ``None``, for Task 5's slicing oracle).
    """

    def __init__(self, head_dim=64, kv_heads=None, control_peak: int = 0, control_gain=5.0):
        self.control_peak = control_peak
        self.control_gain = control_gain
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
        out = mx.zeros((1, inp.shape[1], 3))
        if cache is not None and getattr(cache[0], "control_marker", False):
            if self.control_gain is None:
                token_ids = inp[0]
                gains = 4.0 + 0.5 * (token_ids % 2).astype(mx.float32)
                out[:, :, 0] = gains
            else:
                out[:, :, self.control_peak] = self.control_gain
        elif cache is not None and getattr(cache[0], "marker", False):
            out[:, :, 1] = 5.0
        else:
            out[:, :, 0] = 5.0
        return out
