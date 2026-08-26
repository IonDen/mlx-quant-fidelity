"""KV-cache method seam: the probe measures any per-layer cache through this Protocol.

Stock mlx-lm ``QuantizedKVCache`` is the reference implementation; third-party caches
(TurboQuant-MLX first) plug in beside it. Everything method-specific lives here so
``probes/kv.py`` stays method-agnostic (enforced by a ruff TID251 rule).
"""

import importlib.metadata
import json
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from mlx_quant_fidelity.costs import kv_bytes_per_token
from mlx_quant_fidelity.errors import (
    CacheNotQuantizableError,
    CompareConfigError,
    MethodUnavailableError,
)

TURBOQUANT_PINNED_COMMIT = "6e928d715595dee9f6b6cc3968baa44e1f408d28"
TURBOQUANT_DEFAULT_SEED = 42
TURBOQUANT_INSTALL_HINT = (
    'pip install "turboquant-mlx @ git+https://github.com/arozanov/turboquant-mlx'
    f'@{TURBOQUANT_PINNED_COMMIT}"'
)

_STOCK_BITS: tuple[int, ...] = (2, 3, 4, 6, 8)


class KVCacheMethod(Protocol):
    """One way of quantizing the per-layer KV cache, as the probe consumes it.

    Pure members (``name``, ``label``, ``params``, ``check``, ``bytes_per_token``,
    ``working_set_bytes``, ``provenance``, ``report_warnings``) run without a model. Impure
    members build or inspect real cache objects. Implementations must not force evaluation in
    ``make_cache``, ``measured_bytes``, ``bytes_per_token`` or ``guard`` (they run inside
    the probe's chunk loop); ``convert_prefix`` must return caches whose ``state`` has
    been evaluated.
    """

    @property
    def name(self) -> str:
        """Method id: ``'stock'`` | ``'turboquant'``."""
        ...

    @property
    def label(self) -> str:
        """Report / partial label, e.g. ``'4:64'`` or ``'turboquant:4'``."""
        ...

    @property
    def params(self) -> dict[str, int]:
        """JSON-stable (int-valued) parameters; part of the compare partial identity."""
        ...

    def check(self, *, head_dim: int | None, model_type: str) -> list[str]:
        """Pure pre-flight: raise CacheNotQuantizableError if impossible; return warnings."""
        ...

    def probe_capability(self, empty_cache: list[object]) -> None:
        """Prove the method can run on this model's (empty) per-layer cache list; raise if not."""
        ...

    def make_cache(self, *, n_layers: int) -> list[object]:
        """Fresh quantized per-layer caches for stress mode."""
        ...

    def convert_prefix(self, fp_cache: list[object]) -> list[object]:
        """Deployment boundary: return caches holding the quantized prefix of ``fp_cache``."""
        ...

    def guard(self) -> AbstractContextManager[None]:
        """Context manager wrapping cache construction + forward (re-raises known crashes)."""
        ...

    def bytes_per_token(self, *, n_layers: int, n_kv_heads: int, head_dim: int) -> int:
        """Analytic KV bytes per token — the ranking cost."""
        ...

    def working_set_bytes(
        self, *, window: int, n_layers: int, n_kv_heads: int, head_dim: int, dtype_bytes: int
    ) -> int:
        """Resident+transient bytes the method's fetch path needs beyond stored cache state and logits."""
        ...

    def measured_bytes(self, cache: list[object]) -> int:
        """Bytes of the STORED tokens in a filled cache list (trimmed state, never ``.nbytes``)."""
        ...

    def provenance(self) -> dict[str, str]:
        """Package / version / commit / seed facts for the report."""
        ...

    def report_warnings(self) -> list[str]:
        """Method-specific notes the report must carry (numerics path, pin mismatch)."""
        ...


def _iter_arrays(obj: object) -> Iterator[mx.array]:
    """Yield every mx.array inside a (possibly nested) state structure; skip everything else."""
    if isinstance(obj, mx.array):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_arrays(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _iter_arrays(item)


def stored_state_bytes(cache: list[object]) -> int:
    """Sum the bytes of every array in each layer's ``state`` (the trimmed stored tokens).

    Never use a cache's own ``.nbytes``: mlx-lm's ``QuantizedKVCache.nbytes`` sums the
    step-256 padded buffers, TurboQuant's trims to ``offset`` — the two disagree by up to
    2x at windows that are not a multiple of 256. ``state`` slices to ``offset`` on both.
    ``nbytes`` is shape metadata, so this forces no evaluation.
    """
    return sum(int(a.nbytes) for layer in cache for a in _iter_arrays(layer.state))  # type: ignore[attr-defined, misc]


@contextmanager
def _packed_width_belt(kv_bits: int) -> Iterator[None]:
    """Re-raise mlx-lm's packed-width broadcast_shapes crash as a package-rooted error.

    Belt to the pre-flight ``packed_width_mismatch`` gate, for architectures whose head_dim
    is not derivable. Unrelated ValueErrors propagate untouched.
    """
    try:
        yield
    except ValueError as exc:
        if "broadcast_shapes" in str(exc):
            raise CacheNotQuantizableError(
                f"kv_bits={kv_bits} crashed appending to a fresh QuantizedKVCache "
                f"(mlx-lm packed-width truncation bug): {exc}"
            ) from exc
        raise


@dataclass(frozen=True, slots=True)
class StockKVMethod:
    """mlx-lm's ``QuantizedKVCache`` at (bits, group_size) — the reference implementation."""

    bits: int
    group_size: int

    @property
    def name(self) -> str:
        """``'stock'``."""
        return "stock"

    @property
    def label(self) -> str:
        """``'bits:group_size'`` — unchanged from 0.5.x."""
        return f"{self.bits}:{self.group_size}"

    @property
    def params(self) -> dict[str, int]:
        """``{"bits", "group_size"}``."""
        return {"bits": self.bits, "group_size": self.group_size}

    def check(self, *, head_dim: int | None, model_type: str) -> list[str]:
        """Head-dim divisibility / bits / packed-width gates, in 0.5.x order."""
        from mlx_quant_fidelity.probes.kv import packed_width_mismatch

        warnings: list[str] = []
        if head_dim is None:
            warnings.append(
                f"head_dim/kv_group_size compatibility unverified for '{model_type}'; "
                "relying on MLX to surface a mismatch at first use."
            )
        elif head_dim % self.group_size != 0:
            raise CacheNotQuantizableError(
                f"kv_group_size={self.group_size} does not divide the model's KV "
                f"head_dim={head_dim}; choose a group size that divides {head_dim} (e.g. 32 or 64)."
            )
        if self.bits not in _STOCK_BITS:
            raise CacheNotQuantizableError(
                f"unsupported kv_bits={self.bits}; MLX supports 2/3/4/6/8."
            )
        if head_dim is not None and packed_width_mismatch(head_dim, self.bits):
            usable = [b for b in _STOCK_BITS if not packed_width_mismatch(head_dim, b)]
            remedy = (
                f"use bits {'/'.join(str(b) for b in usable)} or a group-compatible head_dim."
                if usable
                else "no supported bit width packs cleanly at this head_dim."
            )
            raise CacheNotQuantizableError(
                f"kv_bits={self.bits} cannot append to a fresh QuantizedKVCache at "
                f"head_dim={head_dim} on this mlx-lm version (packed-width truncation bug); "
                f"{remedy}"
            )
        return warnings

    def probe_capability(self, empty_cache: list[object]) -> None:
        """Call ``to_quantized`` on every (empty) layer; name the type if it is absent or NYI."""
        for layer in empty_cache:
            to_q = getattr(layer, "to_quantized", None)
            if to_q is None:
                raise CacheNotQuantizableError(
                    f"cache layer {type(layer).__name__} has no to_quantized; "
                    "this model's KV cache cannot be quantized (e.g. sliding-window / MLA)."
                )
            try:
                to_q(group_size=self.group_size, bits=self.bits)
            except NotImplementedError as exc:
                raise CacheNotQuantizableError(
                    f"cache layer {type(layer).__name__} declares to_quantized but it is NYI: {exc}"
                ) from exc
            except (ValueError, RuntimeError) as exc:
                raise CacheNotQuantizableError(
                    f"cache layer {type(layer).__name__} cannot quantize at "
                    f"group_size={self.group_size}, bits={self.bits}: {exc}"
                ) from exc

    def make_cache(self, *, n_layers: int) -> list[object]:
        """``QuantizedKVCache(group_size, bits)`` per layer."""
        from mlx_lm.models.cache import QuantizedKVCache

        return [
            QuantizedKVCache(group_size=self.group_size, bits=self.bits) for _ in range(n_layers)
        ]

    def convert_prefix(self, fp_cache: list[object]) -> list[object]:
        """``to_quantized`` on each stored full-precision layer (mirrors mlx-lm's boundary).

        Evaluates every converted layer's state immediately: the values are unchanged, only
        the materialization point moves earlier, releasing the full-precision prefix before
        segment 2 (the seam's contract requires ``convert_prefix`` to return already-evaluated
        state; see :class:`KVCacheMethod`).
        """
        out = [
            c.to_quantized(group_size=self.group_size, bits=self.bits)  # type: ignore[attr-defined]
            for c in fp_cache
        ]
        mx.eval([c.state for c in out])
        return out

    def guard(self) -> AbstractContextManager[None]:
        """The packed-width belt."""
        return _packed_width_belt(self.bits)

    def bytes_per_token(self, *, n_layers: int, n_kv_heads: int, head_dim: int) -> int:
        """Delegates to :func:`~mlx_quant_fidelity.costs.kv_bytes_per_token`."""
        return kv_bytes_per_token(
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            bits=self.bits,
            group_size=self.group_size,
        )

    def working_set_bytes(
        self, *, window: int, n_layers: int, n_kv_heads: int, head_dim: int, dtype_bytes: int
    ) -> int:
        """Zero: mlx-lm's quantized attention reads the packed cache directly, no extra buffer."""
        return 0

    def measured_bytes(self, cache: list[object]) -> int:
        """Trimmed stored bytes (see :func:`stored_state_bytes`)."""
        return stored_state_bytes(cache)

    def provenance(self) -> dict[str, str]:
        """``{"package": "mlx-lm", "version": <installed>}``."""
        return {"package": "mlx-lm", "version": importlib.metadata.version("mlx-lm")}

    def report_warnings(self) -> list[str]:
        """Stock carries no method note (its attention-path note is already in the docs)."""
        return []

    def control_method(self) -> "AffineKVMethod":
        """The quantizer-only control: same bits both sides, dequantize-on-fetch, standard SDPA."""
        return AffineKVMethod(k_bits=self.bits, v_bits=self.bits, group_size=self.group_size)


class _AffineCache:
    """Per-layer affine-quantized K/V storage that dequantizes on fetch.

    The engine behind both the ``affine`` method and stock's quantizer-only control lane:
    ``mx.quantize`` on append (per side, own bit-width, shared group size), ``mx.dequantize``
    on fetch, so the model runs standard SDPA on full-width tensors. Deliberately exposes NO
    ``bits`` attribute — mlx-lm's SDPA routing checks ``hasattr(cache, "bits")`` and must
    keep taking the standard path.
    """

    def __init__(self, *, k_bits: int, v_bits: int, group_size: int) -> None:
        """Store the per-side bit widths and shared group size; start empty."""
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.offset = 0
        self._k: tuple[mx.array, mx.array, mx.array] | None = None
        self._v: tuple[mx.array, mx.array, mx.array] | None = None

    def _append(
        self,
        stored: tuple[mx.array, mx.array, mx.array] | None,
        new: mx.array,
        *,
        bits: int,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Quantize ``new`` and concatenate it onto ``stored`` along the token axis."""
        q = mx.quantize(new, group_size=self.group_size, bits=bits)
        if stored is None:
            return q
        return tuple(  # type: ignore[return-value]
            mx.concatenate([s, n], axis=-2) for s, n in zip(stored, q, strict=True)
        )

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Quantize-append the new segment per side; return the full dequantized history."""
        self._k = self._append(self._k, keys, bits=self.k_bits)
        self._v = self._append(self._v, values, bits=self.v_bits)
        self.offset += int(keys.shape[-2])
        k_out = mx.dequantize(*self._k, group_size=self.group_size, bits=self.k_bits)
        v_out = mx.dequantize(*self._v, group_size=self.group_size, bits=self.v_bits)
        return k_out, v_out

    @property
    def state(self) -> tuple[mx.array, ...]:
        """The six stored arrays (k triple + v triple); exact token count, no step padding."""
        if self._k is None or self._v is None:
            return ()
        return (*self._k, *self._v)

    def is_trimmable(self) -> bool:
        """Always trimmable (mirrors KVCache)."""
        return True

    def trim(self, n: int) -> int:
        """Drop the last ``n`` stored tokens; return how many were actually dropped."""
        n = min(n, self.offset)
        if n > 0 and self._k is not None and self._v is not None:
            keep = self.offset - n
            self._k = tuple(a[..., :keep, :] for a in self._k)  # type: ignore[assignment]
            self._v = tuple(a[..., :keep, :] for a in self._v)  # type: ignore[assignment]
            self.offset = keep
        return n


_TURBOQUANT_BITS: tuple[int, ...] = (2, 3, 4)
_VALS_PER_WORD: dict[int, int] = {1: 32, 2: 16, 3: 10, 4: 8}  # vendored from turboquant_mlx.packing
_TURBOQUANT_MAX_HEAD_DIM = 256
_TURBOQUANT_REQUIRED = ("update_and_fetch", "state", "trim", "offset")


def _packed_dim(head_dim: int, bits: int) -> int:
    return -(-head_dim // _VALS_PER_WORD[bits])


_TURBOQUANT_STEP = 256


def _step_padded_window(window: int) -> int:
    """Round ``window`` up to the port's step-256 working-buffer granularity.

    The port pre-allocates its retained dequantization buffers (and its prefill transients)
    to the next multiple of 256 tokens, never the exact window — this is what makes a
    window=300 run cost the same working set as window=512.
    """
    return -(-window // _TURBOQUANT_STEP) * _TURBOQUANT_STEP


def _gate_turboquant_head_dim(head_dim: int | None, *, model_type: str | None = None) -> None:
    """Raise unless head_dim is known, a power of two, and <= 256.

    The port's fused Metal kernels have no dimension assert: a non-power-of-two head_dim returns
    plausible but wrong values and > 256 overruns the threadgroup arrays. Nothing downstream can
    catch either, so this pure gate is the only line of defence. ``model_type`` names the model in
    the message when the caller has one (``check``); ``turboquant_bytes_per_token`` is called with
    only a geometry, so it passes None and the clause naming the model is omitted rather than
    printing a placeholder like ``model '?'``.
    """
    if (
        head_dim is None
        or head_dim <= 0
        or head_dim & (head_dim - 1)
        or head_dim > _TURBOQUANT_MAX_HEAD_DIM
    ):
        model_clause = f" model '{model_type}' has" if model_type is not None else " got"
        raise CacheNotQuantizableError(
            f"TurboQuant-MLX kernels require a known power-of-two head_dim <= "
            f"{_TURBOQUANT_MAX_HEAD_DIM};{model_clause} head_dim={head_dim}. This is a "
            "limitation of the port's kernels, not of the model."
        )


def turboquant_bytes_per_token(*, n_layers: int, n_kv_heads: int, head_dim: int, bits: int) -> int:
    """Stored bytes per token for TurboQuant's uniform-bit cache.

    Per (token, head): ``packed_dim(head_dim, bits)`` uint32 words of codes plus one fp32 norm;
    K and V; every layer. Reproduces 9216 / 8192 / 5120 B/token at 4/3/2 bits on Llama-3.2-1B
    (16 layers x 8 heads x 64 dims), including the 3-bit round-up from ``VALS_PER_WORD[3] == 10``.
    """
    _gate_turboquant_head_dim(head_dim)
    per_token_head = _packed_dim(head_dim, bits) * 4 + 4
    return per_token_head * 2 * n_kv_heads * n_layers


def _turboquant_dist_version() -> str:
    try:
        return importlib.metadata.version("turboquant-mlx")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _turboquant_direct_url() -> str | None:
    try:
        return importlib.metadata.distribution("turboquant-mlx").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None


def _installed_commit() -> str:
    raw = _turboquant_direct_url()
    if not raw:
        return "unknown"
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        return "unknown"
    vcs = info.get("vcs_info") if isinstance(info, dict) else None
    commit = vcs.get("commit_id") if isinstance(vcs, dict) else None
    return commit if isinstance(commit, str) and commit else "unknown"


@dataclass(frozen=True, slots=True)
class TurboQuantKVMethod:
    """arozanov/turboquant-mlx's uniform-bit ``TurboQuantKVCache`` (dequantize-on-fetch path).

    Install: ``TURBOQUANT_INSTALL_HINT``. Never ``pip install turboquant-mlx`` by name — the PyPI
    package of that name is an unrelated squatter.
    """

    bits: int
    seed: int = TURBOQUANT_DEFAULT_SEED

    def __post_init__(self) -> None:
        """Reject bit widths the port does not implement, and a non-positive rotation seed."""
        if self.bits not in _TURBOQUANT_BITS:
            raise ValueError(f"turboquant supports bits 2/3/4, got bits={self.bits}")
        if self.seed < 1:
            raise ValueError(f"turboquant seed must be >= 1, got seed={self.seed}")

    @property
    def name(self) -> str:
        """``'turboquant'``."""
        return "turboquant"

    @property
    def label(self) -> str:
        """``'turboquant:bits'``, plus ``':seed'`` only when the seed is not the port's default."""
        base = f"turboquant:{self.bits}"
        return base if self.seed == TURBOQUANT_DEFAULT_SEED else f"{base}:{self.seed}"

    @property
    def params(self) -> dict[str, int]:
        """``{"bits", "seed"}``."""
        return {"bits": self.bits, "seed": self.seed}

    def check(self, *, head_dim: int | None, model_type: str) -> list[str]:
        """The head_dim gate (see :func:`_gate_turboquant_head_dim`); no warnings."""
        _gate_turboquant_head_dim(head_dim, model_type=model_type)
        return []

    def _cache_cls(self) -> type:
        """Import the port and verify its API shape; raise MethodUnavailableError with the pin."""
        try:
            import turboquant_mlx
        except ImportError as exc:
            raise MethodUnavailableError(
                f"turboquant_mlx is not installed. Install the pinned port: {TURBOQUANT_INSTALL_HINT}"
            ) from exc
        try:
            from turboquant_mlx.cache import TurboQuantKVCache
        except ImportError as exc:
            raise MethodUnavailableError(
                "the installed `turboquant_mlx` has no `cache` module — this is the PyPI "
                "`turboquant-mlx` squatter, not the arozanov port. Uninstall it and run: "
                f"{TURBOQUANT_INSTALL_HINT}"
            ) from exc
        if not hasattr(turboquant_mlx, "__version__"):
            raise MethodUnavailableError(
                f"turboquant_mlx has no __version__; expected the port at {TURBOQUANT_PINNED_COMMIT}."
            )
        inst = TurboQuantKVCache()
        missing = [a for a in _TURBOQUANT_REQUIRED if not hasattr(inst, a)]
        if missing:
            raise MethodUnavailableError(
                f"TurboQuantKVCache is missing {missing}; expected the port at "
                f"{TURBOQUANT_PINNED_COMMIT} ({TURBOQUANT_INSTALL_HINT})."
            )
        if hasattr(inst, "bits"):
            raise MethodUnavailableError(
                "TurboQuantKVCache exposes `bits`, which would route it through mlx-lm's quantized "
                "attention instead of standard SDPA; the measured path would no longer be the "
                f"quantizer alone. Expected the port at {TURBOQUANT_PINNED_COMMIT}."
            )
        return TurboQuantKVCache  # type: ignore[no-any-return]

    def _new(self, cls: type) -> object:
        return cls(
            bits=self.bits, seed=self.seed, fused=False, v_only=False, sparse_v_threshold=None
        )

    def probe_capability(self, empty_cache: list[object]) -> None:
        """Every layer must be a plain mlx-lm ``KVCache``; then run and EVALUATE the contract.

        The contract feeds 8 tokens through a throwaway cache and forces the graph, so a Metal
        compile/dispatch failure surfaces here as MethodUnavailableError instead of mid-run
        (where ``guard()`` is a no-op for this method). Costs one JIT per process (~50 ms),
        then well under a millisecond per call. Also proves the trim contract on both halves
        (the basis of the 2.3x memory note): the port must actually populate the dequantized
        working buffers after ``update_and_fetch``, and ``trim(0)`` must then release them. A
        port whose trim leaves them resident would silently make ``convert_prefix`` retain
        full-precision working copies of the prefix past the boundary; a port that never
        populates them in the first place would make that same memory note groundless.
        """
        from mlx_lm.models.cache import KVCache

        for layer in empty_cache:
            if type(layer) is not KVCache:
                raise CacheNotQuantizableError(
                    f"cache layer {type(layer).__name__} is not a plain KVCache; TurboQuant-MLX "
                    "only replaces plain per-layer caches (sliding-window / MLA / mixed models are "
                    "not supported)."
                )
        cls = self._cache_cls()
        probe = self._new(cls)
        try:
            probe.update_and_fetch(mx.zeros((1, 1, 8, 64)), mx.zeros((1, 1, 8, 64)))  # type: ignore[attr-defined]
            mx.eval(probe.state)  # type: ignore[attr-defined]
        except Exception as exc:
            raise MethodUnavailableError(
                f"TurboQuantKVCache failed to execute its kernels: {exc}; expected the port at "
                f"{TURBOQUANT_PINNED_COMMIT}."
            ) from exc
        expected = self.bytes_per_token(n_layers=1, n_kv_heads=1, head_dim=64) * 8
        got_offset = getattr(probe, "offset", None)
        got_bytes = stored_state_bytes([probe])
        if got_offset != 8 or got_bytes != expected:
            raise MethodUnavailableError(
                f"TurboQuantKVCache behaviour differs from the pinned port (offset {got_offset} != 8 "
                f"or stored bytes {got_bytes} != {expected}); expected {TURBOQUANT_PINNED_COMMIT}."
            )
        for attr in ("_k_deq_buf", "_v_deq_buf"):
            if getattr(probe, attr, None) is None:
                raise MethodUnavailableError(
                    f"TurboQuantKVCache has no populated {attr} after update_and_fetch; "
                    f"expected the port at {TURBOQUANT_PINNED_COMMIT}."
                )
        probe.trim(0)  # type: ignore[attr-defined]
        if (
            getattr(probe, "_k_deq_buf", None) is not None
            or getattr(probe, "_v_deq_buf", None) is not None
        ):
            raise MethodUnavailableError(
                "TurboQuantKVCache.trim(0) did not release the dequantized working buffers; "
                f"expected the port at {TURBOQUANT_PINNED_COMMIT}."
            )

    def make_cache(self, *, n_layers: int) -> list[object]:
        """Fresh uniform-bit caches with every knob pinned (fused/v_only off, no sparse V)."""
        cls = self._cache_cls()
        return [self._new(cls) for _ in range(n_layers)]

    def convert_prefix(self, fp_cache: list[object]) -> list[object]:
        """Replay each layer's stored full-precision prefix through a fresh cache, then trim.

        ``KVCache.state`` is sliced to ``offset`` (never the step-padded buffer). ``trim(0)`` drops
        the dequantized working buffers the port retains after ``update_and_fetch`` (otherwise a
        step-padded fp16 copy of the prefix stays resident per layer and short segments take a
        different, fp16-output kernel). One batched eval for all layers, as kv.py does.
        """
        cls = self._cache_cls()
        out: list[object] = []
        for layer in fp_cache:
            k, v = layer.state  # type: ignore[attr-defined]
            new = self._new(cls)
            new.update_and_fetch(k, v)  # type: ignore[attr-defined]
            new.trim(0)  # type: ignore[attr-defined]
            out.append(new)
        mx.eval([c.state for c in out])  # type: ignore[attr-defined]
        return out

    def guard(self) -> AbstractContextManager[None]:
        """No known crash to translate."""
        return nullcontext()

    def bytes_per_token(self, *, n_layers: int, n_kv_heads: int, head_dim: int) -> int:
        """Analytic stored bytes (see :func:`turboquant_bytes_per_token`); gates geometry too."""
        return turboquant_bytes_per_token(
            n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim, bits=self.bits
        )

    def working_set_bytes(
        self, *, window: int, n_layers: int, n_kv_heads: int, head_dim: int, dtype_bytes: int
    ) -> int:
        """Prefill's ``all_k``+``all_v`` transients plus the two retained step-padded dequant buffers.

        Four buffers, each shaped ``(n_layers, n_kv_heads, step-256-padded window, head_dim)`` at
        ``dtype_bytes``: the prefill forward's full-window dequantized K and V (transient, freed
        after the forward), plus the ``_k_deq_buf``/``_v_deq_buf`` working copies the port retains
        after ``update_and_fetch`` (see :meth:`probe_capability`'s trim-release contract).
        """
        padded_window = _step_padded_window(window)
        return 4 * n_layers * n_kv_heads * padded_window * head_dim * dtype_bytes

    def measured_bytes(self, cache: list[object]) -> int:
        """Trimmed stored bytes (see :func:`stored_state_bytes`)."""
        return stored_state_bytes(cache)

    def provenance(self) -> dict[str, str]:
        """Package / module versions, installed vs pinned commit, seeds and pinned knobs."""
        try:
            import turboquant_mlx
        except ImportError as exc:
            raise MethodUnavailableError(
                f"turboquant_mlx is not installed. Install the pinned port: {TURBOQUANT_INSTALL_HINT}"
            ) from exc

        return {
            "package": "turboquant-mlx",
            "dist_version": _turboquant_dist_version(),
            "module_version": str(getattr(turboquant_mlx, "__version__", "unknown")),
            "commit": _installed_commit(),
            "pinned_commit": TURBOQUANT_PINNED_COMMIT,
            "k_seed": str(self.seed),
            "v_seed": str(self.seed + 1),
            "fused": "false",
            "v_only": "false",
            "sparse_v_threshold": "none",
        }

    def report_warnings(self) -> list[str]:
        """The numerics/scope/memory note, plus a pin-mismatch warning when the commit differs."""
        notes = [
            "turboquant: the quantized run dequantizes on fetch and rides standard SDPA in prefill "
            "(the port's fused kernel is decode-only and not exercised), so drift measures the "
            "quantizer round-trip only, while stock bundles quantizer + quantized-attention "
            "numerics; uniform-bit cache at the port's default seed — its V-only configuration is "
            "measured separately (`turboquant-vonly`); its layer-adaptive configuration is not — "
            "and its `make_adaptive_cache` silently ignores the documented `k_bits`/`v_bits` "
            "parameters at the pinned commit; resident memory in this path is the stored bytes "
            "plus two full-precision working copies — roughly 2.3x an fp16 cache on Llama-3.2-1B "
            "geometry, derived from the port's retained dequantization buffers — so peak memory "
            "does not show the compression."
        ]
        commit = _installed_commit()
        if commit != TURBOQUANT_PINNED_COMMIT:
            notes.append(
                f"turboquant: installed commit {commit} is not the pinned {TURBOQUANT_PINNED_COMMIT}; "
                "numbers may not reproduce the committed sample."
            )
        return notes


@dataclass(frozen=True, slots=True)
class TurboQuantVOnlyKVMethod:
    """arozanov/turboquant-mlx's ``VOnlyTurboQuantCache`` — fp16 K, TurboQuant-compressed V only.

    K stays in a plain fp16 KVCache (standard SDPA); only V is quantized-and-dequantized on
    fetch. The port's own inner cache still stores a full fp16 copy of V as a side effect of
    reusing ``KVCache`` for K (see :meth:`report_warnings`) — so this method's stored bytes can
    exceed a plain fp16 cache's, even though V itself compresses. Install: see
    ``TURBOQUANT_INSTALL_HINT``.
    """

    v_bits: int
    seed: int = TURBOQUANT_DEFAULT_SEED

    def __post_init__(self) -> None:
        """Reject V bit widths the port does not implement, and a non-positive rotation seed."""
        if self.v_bits not in _TURBOQUANT_BITS:
            raise ValueError(f"turboquant-vonly supports v_bits 2/3/4, got v_bits={self.v_bits}")
        if self.seed < 1:
            raise ValueError(f"turboquant-vonly seed must be >= 1, got seed={self.seed}")

    @property
    def name(self) -> str:
        """``'turboquant-vonly'``."""
        return "turboquant-vonly"

    @property
    def label(self) -> str:
        """``'turboquant-vonly:v_bits'``, plus ``':seed'`` when the seed is not the default."""
        base = f"turboquant-vonly:{self.v_bits}"
        return base if self.seed == TURBOQUANT_DEFAULT_SEED else f"{base}:{self.seed}"

    @property
    def params(self) -> dict[str, int]:
        """``{"v_bits", "seed"}``."""
        return {"v_bits": self.v_bits, "seed": self.seed}

    def check(self, *, head_dim: int | None, model_type: str) -> list[str]:
        """The head_dim gate (see :func:`_gate_turboquant_head_dim`); no warnings.

        The V path still runs the port's fused Metal kernels, so the same power-of-two/<=256
        constraint applies even though K stays in plain fp16.
        """
        _gate_turboquant_head_dim(head_dim, model_type=model_type)
        return []

    def _cache_cls(self) -> type:
        """Import the port and verify its V-only API shape; raise MethodUnavailableError with the pin."""
        try:
            import turboquant_mlx
        except ImportError as exc:
            raise MethodUnavailableError(
                f"turboquant_mlx is not installed. Install the pinned port: {TURBOQUANT_INSTALL_HINT}"
            ) from exc
        try:
            from turboquant_mlx.v_only_cache import VOnlyTurboQuantCache
        except ImportError as exc:
            raise MethodUnavailableError(
                "the installed `turboquant_mlx` has no `v_only_cache` module — this is the PyPI "
                "`turboquant-mlx` squatter, not the arozanov port, or a pinned commit predating "
                f"V-only support. Uninstall it and run: {TURBOQUANT_INSTALL_HINT}"
            ) from exc
        if not hasattr(turboquant_mlx, "__version__"):
            raise MethodUnavailableError(
                f"turboquant_mlx has no __version__; expected the port at {TURBOQUANT_PINNED_COMMIT}."
            )
        # Class-level (descriptor) presence, not instance-level: on the real port,
        # VOnlyTurboQuantCache.state is a property that RAISES on a fresh/empty instance (it
        # dereferences an inner plain KVCache's `.keys.shape` while `keys` is still None before
        # the first update_and_fetch). hasattr(instance, attr) swallows that AttributeError as
        # "attribute absent" and misreports the genuine port as missing `state` (task-6 F1).
        # hasattr(cls, attr) reads the property descriptor off the class without invoking its
        # getter, so it survives a raising property. Behavioral verification of the FILLED
        # cache still happens below, via the 8-token probe (which fills before reading state).
        missing = [a for a in _TURBOQUANT_REQUIRED if not hasattr(VOnlyTurboQuantCache, a)]
        if missing:
            raise MethodUnavailableError(
                f"VOnlyTurboQuantCache is missing {missing}; expected the port at "
                f"{TURBOQUANT_PINNED_COMMIT} ({TURBOQUANT_INSTALL_HINT})."
            )
        inst = VOnlyTurboQuantCache()
        if hasattr(inst, "bits"):
            raise MethodUnavailableError(
                "VOnlyTurboQuantCache exposes `bits`, which would route it through mlx-lm's "
                "quantized attention instead of standard SDPA; the measured path would no longer "
                f"be the quantizer alone. Expected the port at {TURBOQUANT_PINNED_COMMIT}."
            )
        return VOnlyTurboQuantCache  # type: ignore[no-any-return]

    def _new(self, cls: type) -> object:
        return cls(bits=self.v_bits, seed=self.seed)

    def probe_capability(self, empty_cache: list[object]) -> None:
        """Every layer must be a plain mlx-lm ``KVCache``; then run and EVALUATE the V contract.

        Mirrors :meth:`TurboQuantKVMethod.probe_capability`'s 8-token contract, but the port's V
        dequant working buffer lives one level down at ``probe._v_tq._v_deq_buf`` (the inner
        ``TurboQuantKVCache`` instance the V-only wrapper delegates V storage to).
        """
        from mlx_lm.models.cache import KVCache

        for layer in empty_cache:
            if type(layer) is not KVCache:
                raise CacheNotQuantizableError(
                    f"cache layer {type(layer).__name__} is not a plain KVCache; TurboQuant-MLX "
                    "V-only only replaces plain per-layer caches (sliding-window / MLA / mixed "
                    "models are not supported)."
                )
        cls = self._cache_cls()
        probe = self._new(cls)
        try:
            # fp16, not the fp32 mx.zeros default: the real port's inner plain KVCache stores
            # K (and the unused duplicate V) in whatever dtype it's fed, and bytes_per_token's
            # fp16 assumption (a real forward's activation dtype) only holds if the contract
            # probe itself feeds fp16 -- an fp32 probe silently doubles that portion of the
            # stored bytes and desyncs from the analytic formula below.
            fp16_zeros = mx.zeros((1, 1, 8, 64), dtype=mx.float16)
            probe.update_and_fetch(fp16_zeros, fp16_zeros)  # type: ignore[attr-defined]
            mx.eval(probe.state)  # type: ignore[attr-defined]
        except Exception as exc:
            raise MethodUnavailableError(
                f"VOnlyTurboQuantCache failed to execute its kernels: {exc}; expected the port at "
                f"{TURBOQUANT_PINNED_COMMIT}."
            ) from exc
        expected = self.bytes_per_token(n_layers=1, n_kv_heads=1, head_dim=64) * 8
        got_offset = getattr(probe, "offset", None)
        got_bytes = stored_state_bytes([probe])
        if got_offset != 8 or got_bytes != expected:
            raise MethodUnavailableError(
                f"VOnlyTurboQuantCache behaviour differs from the pinned port (offset {got_offset} "
                f"!= 8 or stored bytes {got_bytes} != {expected}); expected {TURBOQUANT_PINNED_COMMIT}."
            )
        v_tq = getattr(probe, "_v_tq", None)
        if v_tq is None or getattr(v_tq, "_v_deq_buf", None) is None:
            raise MethodUnavailableError(
                "VOnlyTurboQuantCache has no populated _v_tq._v_deq_buf after update_and_fetch; "
                f"expected the port at {TURBOQUANT_PINNED_COMMIT}."
            )
        probe.trim(0)  # type: ignore[attr-defined]
        if getattr(v_tq, "_v_deq_buf", None) is not None:
            raise MethodUnavailableError(
                "VOnlyTurboQuantCache.trim(0) did not release the V dequantized working buffer; "
                f"expected the port at {TURBOQUANT_PINNED_COMMIT}."
            )

    def make_cache(self, *, n_layers: int) -> list[object]:
        """Fresh V-only caches; buffered mode always (``no_v_buffer`` is never passed)."""
        cls = self._cache_cls()
        return [self._new(cls) for _ in range(n_layers)]

    def convert_prefix(self, fp_cache: list[object]) -> list[object]:
        """Replay each layer's stored full-precision prefix through a fresh V-only cache, then trim.

        ``KVCache.state`` is sliced to ``offset`` (never the step-padded buffer). ``trim(0)`` drops
        the V dequantized working buffer the port retains after ``update_and_fetch``. One batched
        eval for all layers, as :meth:`TurboQuantKVMethod.convert_prefix` does.
        """
        cls = self._cache_cls()
        out: list[object] = []
        for layer in fp_cache:
            k, v = layer.state  # type: ignore[attr-defined]
            new = self._new(cls)
            new.update_and_fetch(k, v)  # type: ignore[attr-defined]
            new.trim(0)  # type: ignore[attr-defined]
            out.append(new)
        mx.eval([c.state for c in out])  # type: ignore[attr-defined]
        return out

    def guard(self) -> AbstractContextManager[None]:
        """No known crash to translate."""
        return nullcontext()

    def bytes_per_token(self, *, n_layers: int, n_kv_heads: int, head_dim: int) -> int:
        """fp16 K + fp16 V (the port's unused duplicate) + packed V + fp32 norms, every layer.

        Source-verified against the pinned port (``v_only_cache.py``): ``update_and_fetch`` feeds
        BOTH tensors to the inner plain ``KVCache`` used for K storage, so ``state`` is four
        arrays, not two — this is what makes the method's stored bytes exceed a plain fp16 cache
        at low V bit-widths (see :meth:`report_warnings`).
        """
        packed = _packed_dim(head_dim, self.v_bits) * 4 + 4
        return n_layers * n_kv_heads * head_dim * 2 * 2 + packed * n_kv_heads * n_layers

    def working_set_bytes(
        self, *, window: int, n_layers: int, n_kv_heads: int, head_dim: int, dtype_bytes: int
    ) -> int:
        """Just the V half of :meth:`TurboQuantKVMethod.working_set_bytes` — K stays plain fp16.

        Only V's prefill transient plus its one retained ``_v_deq_buf`` working copy count;
        same step-256-padded buffer shape as the full-cache port, halved.
        """
        padded_window = _step_padded_window(window)
        return 2 * n_layers * n_kv_heads * padded_window * head_dim * dtype_bytes

    def measured_bytes(self, cache: list[object]) -> int:
        """Trimmed stored bytes (see :func:`stored_state_bytes`)."""
        return stored_state_bytes(cache)

    def provenance(self) -> dict[str, str]:
        """Package / module versions, installed vs pinned commit, and the port's V seed.

        No ``k_seed``: K stays fp16 and is never quantized. ``v_seed`` is ``seed + 1`` — the
        port's inner ``TurboQuantKVCache`` seeds its V quantizer at ``seed + 1`` unconditionally,
        the same offset :attr:`TurboQuantKVMethod.provenance` records.
        """
        try:
            import turboquant_mlx
        except ImportError as exc:
            raise MethodUnavailableError(
                f"turboquant_mlx is not installed. Install the pinned port: {TURBOQUANT_INSTALL_HINT}"
            ) from exc

        return {
            "package": "turboquant-mlx",
            "dist_version": _turboquant_dist_version(),
            "module_version": str(getattr(turboquant_mlx, "__version__", "unknown")),
            "commit": _installed_commit(),
            "pinned_commit": TURBOQUANT_PINNED_COMMIT,
            "v_seed": str(self.seed + 1),
            "no_v_buffer": "false",
        }

    def report_warnings(self) -> list[str]:
        """The numerics-path note, the duplicate-storage note, plus a pin-mismatch warning."""
        notes = [
            "turboquant-vonly: K stays fp16 and rides standard SDPA; only V is "
            "quantized-and-dequantized on fetch, so drift measures the V quantizer round-trip "
            "only, at the port's default seed.",
            "turboquant-vonly: the pinned port stores an unused fp16 copy of V in its inner "
            "KVCache, so stored bytes EXCEED a plain fp16 cache — the V-only value at this "
            "commit is V-compression quality, not memory.",
        ]
        commit = _installed_commit()
        if commit != TURBOQUANT_PINNED_COMMIT:
            notes.append(
                f"turboquant-vonly: installed commit {commit} is not the pinned "
                f"{TURBOQUANT_PINNED_COMMIT}; numbers may not reproduce the committed sample."
            )
        return notes


@dataclass(frozen=True, slots=True)
class AffineKVMethod:
    """Per-tensor asymmetric affine quantization, measured quantizer-only.

    K and V each carry their own bit-width (shared group size, default 64 — the layout of
    the only known runtime, the ``arozanov/mlx-lm`` ``feature/turboquant-kv-cache`` fork's
    ``MixedQuantKVCache``). No shipped mlx-lm executes this config: the probe dequantizes
    on fetch and rides standard SDPA, so the number is quantizer fidelity for a
    hypothetical deployment, and every report says so.
    """

    k_bits: int
    v_bits: int
    group_size: int = 64

    def __post_init__(self) -> None:
        """Reject bit widths outside mx.quantize's set and a non-positive group size."""
        for side, bits in (("k_bits", self.k_bits), ("v_bits", self.v_bits)):
            if bits not in _STOCK_BITS:
                raise ValueError(f"unsupported {side}={bits}; MLX affine supports 2/3/4/6/8.")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")

    @property
    def name(self) -> str:
        """``'affine'``."""
        return "affine"

    @property
    def label(self) -> str:
        """``'affine:k:v'``, plus ``':group'`` only when the group size is not the 64 default."""
        base = f"affine:{self.k_bits}:{self.v_bits}"
        return base if self.group_size == 64 else f"{base}:{self.group_size}"

    @property
    def params(self) -> dict[str, int]:
        """``{"k_bits", "v_bits", "group_size"}``."""
        return {"k_bits": self.k_bits, "v_bits": self.v_bits, "group_size": self.group_size}

    def check(self, *, head_dim: int | None, model_type: str) -> list[str]:
        """Pure pre-flight: the head_dim % group_size gate plain mx.quantize actually has."""
        if head_dim is None:
            return [
                f"head_dim/group_size compatibility unverified for '{model_type}'; "
                "relying on MLX to surface a mismatch at first use."
            ]
        if head_dim % self.group_size != 0:
            raise CacheNotQuantizableError(
                f"group_size={self.group_size} does not divide the model's KV "
                f"head_dim={head_dim}; choose a group size that divides {head_dim}."
            )
        return []

    def probe_capability(self, empty_cache: list[object]) -> None:
        """Require plain per-layer KVCache — the same shape every other method replaces."""
        from mlx_lm.models.cache import KVCache

        for layer in empty_cache:
            if type(layer) is not KVCache:
                raise CacheNotQuantizableError(
                    f"cache layer {type(layer).__name__} is not a plain KVCache; the affine "
                    "method only replaces plain per-layer caches."
                )

    def make_cache(self, *, n_layers: int) -> list[object]:
        """Fresh per-layer affine caches."""
        return [
            _AffineCache(k_bits=self.k_bits, v_bits=self.v_bits, group_size=self.group_size)
            for _ in range(n_layers)
        ]

    def convert_prefix(self, fp_cache: list[object]) -> list[object]:
        """Quantize each layer's stored fp prefix into a fresh affine cache; one batched eval."""
        out: list[object] = []
        for layer in fp_cache:
            k, v = layer.state  # type: ignore[attr-defined]
            new = _AffineCache(k_bits=self.k_bits, v_bits=self.v_bits, group_size=self.group_size)
            new.update_and_fetch(k, v)
            out.append(new)
        mx.eval([c.state for c in out])  # type: ignore[attr-defined]
        return out

    def guard(self) -> AbstractContextManager[None]:
        """No known crash to translate."""
        return nullcontext()

    def bytes_per_token(self, *, n_layers: int, n_kv_heads: int, head_dim: int) -> int:
        """Per side: elements x (bits/8 + 4/group); sum the sides, round once."""
        el_side = n_layers * n_kv_heads * head_dim
        k_side = el_side * (self.k_bits / 8 + 4 / self.group_size)
        v_side = el_side * (self.v_bits / 8 + 4 / self.group_size)
        return round(k_side + v_side)

    def working_set_bytes(
        self, *, window: int, n_layers: int, n_kv_heads: int, head_dim: int, dtype_bytes: int
    ) -> int:
        """K+V dequantized-on-fetch transients at the exact window — ``mx.dequantize`` pads nothing."""
        return 2 * n_layers * n_kv_heads * window * head_dim * dtype_bytes

    def measured_bytes(self, cache: list[object]) -> int:
        """Trimmed stored bytes (see :func:`stored_state_bytes`)."""
        return stored_state_bytes(cache)

    def provenance(self) -> dict[str, str]:
        """The quantizer package plus the only known (fork) runtime for this layout."""
        return {
            "package": "mlx",
            "version": importlib.metadata.version("mlx"),
            "k_bits": str(self.k_bits),
            "v_bits": str(self.v_bits),
            "group_size": str(self.group_size),
            "known_runtime": (
                "arozanov/mlx-lm@feature/turboquant-kv-cache (MixedQuantKVCache, "
                "mlx-lm 0.31.3) — not upstream mlx-lm"
            ),
        }

    def report_warnings(self) -> list[str]:
        """The hypothetical-deployment note every affine report must carry."""
        return [
            "affine: per-tensor asymmetric bits have no shipped runtime — upstream mlx-lm's "
            "QuantizedKVCache is symmetric; the only known implementation is an idle mlx-lm "
            "fork. The probe dequantizes on fetch and rides standard SDPA, so this drift is "
            "the quantizer round-trip only, for a hypothetical deployment."
        ]


def _positive_ints(parts: list[str], *, spec: str, expected: str, example: str) -> list[int]:
    if not parts or not all(p.isascii() and p.isdigit() and int(p) > 0 for p in parts):
        raise CompareConfigError(f"--configs entry {spec!r} must be {expected} (e.g. {example}).")
    return [int(p) for p in parts]


def parse_method_spec(spec: str) -> KVCacheMethod:
    """Parse ``'4:64'`` | ``'stock:4:64'`` | ``'turboquant:bits[:seed]'`` | ``'affine:k:v[:group]'``.

    A bare ``bits:group_size`` is stock (backward-compatible). Raises CompareConfigError
    (a ValueError) on anything malformed. The parser enumerates the shipped methods by
    hand; ``METHODS`` is the CLI's choices list, not a registry (a plugin registry is
    future work).
    """
    parts = spec.split(":")
    if parts and parts[0].isdigit():
        parts = ["stock", *parts]
    name, *args = parts
    if name == "stock":
        if len(args) != 2:
            raise CompareConfigError(
                f"--configs entry {spec!r} must be 'bits:group_size' (e.g. 4:64)."
            )
        bits, gs = _positive_ints(args, spec=spec, expected="'bits:group_size'", example="4:64")
        return StockKVMethod(bits=bits, group_size=gs)
    if name == "turboquant":
        if len(args) not in (1, 2):
            raise CompareConfigError(
                f"--configs entry {spec!r} must be 'turboquant:bits' or 'turboquant:bits:seed'."
            )
        nums = _positive_ints(
            args, spec=spec, expected="'turboquant:bits[:seed]'", example="turboquant:4"
        )
        seed = nums[1] if len(nums) == 2 else TURBOQUANT_DEFAULT_SEED
        try:
            return TurboQuantKVMethod(bits=nums[0], seed=seed)
        except ValueError as exc:
            raise CompareConfigError(f"--configs entry {spec!r}: {exc}") from exc
    if name == "turboquant-vonly":
        if len(args) not in (1, 2):
            raise CompareConfigError(
                f"--configs entry {spec!r} must be 'turboquant-vonly:v_bits' or "
                "'turboquant-vonly:v_bits:seed'."
            )
        nums = _positive_ints(
            args,
            spec=spec,
            expected="'turboquant-vonly:v_bits[:seed]'",
            example="turboquant-vonly:3",
        )
        seed = nums[1] if len(nums) == 2 else TURBOQUANT_DEFAULT_SEED
        try:
            return TurboQuantVOnlyKVMethod(v_bits=nums[0], seed=seed)
        except ValueError as exc:
            raise CompareConfigError(f"--configs entry {spec!r}: {exc}") from exc
    if name == "affine":
        if len(args) not in (2, 3):
            raise CompareConfigError(
                f"--configs entry {spec!r} must be 'affine:k_bits:v_bits[:group_size]' "
                "(e.g. affine:8:4)."
            )
        nums = _positive_ints(
            args, spec=spec, expected="'affine:k_bits:v_bits[:group_size]'", example="affine:8:4"
        )
        gs = nums[2] if len(nums) == 3 else 64
        try:
            return AffineKVMethod(k_bits=nums[0], v_bits=nums[1], group_size=gs)
        except ValueError as exc:
            raise CompareConfigError(f"--configs entry {spec!r}: {exc}") from exc
    raise CompareConfigError(
        f"--configs entry {spec!r}: unknown method {name!r}; known: {sorted(METHODS)}."
    )


METHODS: dict[str, type] = {
    "stock": StockKVMethod,
    "turboquant": TurboQuantKVMethod,
    "affine": AffineKVMethod,
    "turboquant-vonly": TurboQuantVOnlyKVMethod,
}
