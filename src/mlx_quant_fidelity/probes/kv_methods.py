"""KV-cache method seam: the probe measures any per-layer cache through this Protocol.

Stock mlx-lm ``QuantizedKVCache`` is the reference implementation; third-party caches
(TurboQuant-MLX first) plug in beside it. Everything method-specific lives here so
``probes/kv.py`` stays method-agnostic (enforced by a ruff TID251 rule).
"""

import importlib.metadata
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from mlx_quant_fidelity.costs import kv_bytes_per_token
from mlx_quant_fidelity.errors import (
    CacheNotQuantizableError,
    CompareConfigError,
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
    ``provenance``, ``report_warnings``) run without a model. Impure members build or
    inspect real cache objects.
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
        """``to_quantized`` on each stored full-precision layer (mirrors mlx-lm's boundary)."""
        return [
            c.to_quantized(group_size=self.group_size, bits=self.bits)  # type: ignore[attr-defined]
            for c in fp_cache
        ]

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

    def measured_bytes(self, cache: list[object]) -> int:
        """Trimmed stored bytes (see :func:`stored_state_bytes`)."""
        return stored_state_bytes(cache)

    def provenance(self) -> dict[str, str]:
        """``{"package": "mlx-lm", "version": <installed>}``."""
        return {"package": "mlx-lm", "version": importlib.metadata.version("mlx-lm")}

    def report_warnings(self) -> list[str]:
        """Stock carries no method note (its attention-path note is already in the docs)."""
        return []


def _positive_ints(parts: list[str], *, spec: str, expected: str) -> list[int]:
    if not parts or not all(p.isdigit() and int(p) > 0 for p in parts):
        raise CompareConfigError(f"--configs entry {spec!r} must be {expected} (e.g. 4:64).")
    return [int(p) for p in parts]


def parse_method_spec(spec: str) -> KVCacheMethod:
    """Parse ``'4:64'`` | ``'stock:4:64'`` (Task 3 adds ``'turboquant:bits[:seed]'``).

    A bare ``bits:group_size`` is stock (backward-compatible). Raises CompareConfigError
    (a ValueError) on anything malformed. The parser enumerates the shipped methods by
    hand; ``METHODS`` is the CLI's choices list, not a registry (a registry is 0022's job).
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
        bits, gs = _positive_ints(args, spec=spec, expected="'bits:group_size'")
        return StockKVMethod(bits=bits, group_size=gs)
    raise CompareConfigError(
        f"--configs entry {spec!r}: unknown method {name!r}; known: {sorted(METHODS)}."
    )


METHODS: dict[str, type] = {"stock": StockKVMethod}
