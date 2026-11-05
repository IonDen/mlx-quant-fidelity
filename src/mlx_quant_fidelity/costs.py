"""Pure cost-denominator functions for `compare` (the Pareto x-axis) and the KV-layout registry."""

from collections.abc import Callable
from dataclasses import dataclass

# mlx-lm 0.31.3 attentions that cache concat(k_nope, k_pe) at qk_nope + qk_rope and values at
# v_head_dim, per head (deepseek_v2.py:224-232, youtu_llm.py:117-125).
_EXPANDED_KV_MODEL_TYPES = frozenset({"deepseek_v2", "youtu_llm"})
_MLA_FIELDS = ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "kv_lora_rank")


def kv_bytes_per_token(
    *, n_layers: int, n_kv_heads: int, head_dim: int, bits: int, group_size: int
) -> int:
    """KV-cache bytes per token for a quantized cache, incl. per-group scale+bias overhead.

    Per element: ``bits/8`` packed + a fp16 scale (2 B) and fp16 bias (2 B) amortized over
    ``group_size``. Counts K and V across every layer. The group overhead is load-bearing — a
    smaller group_size buys accuracy but costs bytes, and the Pareto must see it. See
    docs/ranking-principles.md (the cost axis).

    The ``4 / group_size`` term (scale + bias = 2 B + 2 B = 4 B per group) assumes a
    **bf16/fp16 KV cache**, which is the only path validated on Apple Silicon (MLX's
    ``QuantizedKVCache`` stores scales/biases in the input dtype, always bf16/fp16 here).
    An fp32 KV cache would instead carry 8 B of overhead per group
    (``8 / group_size`` bytes/element), but that path is not currently supported.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    elements_per_token = 2 * n_layers * n_kv_heads * head_dim  # K and V
    bytes_per_element = bits / 8 + 4 / group_size  # 4 B = fp16 scale + fp16 bias per group
    return round(elements_per_token * bytes_per_element)


@dataclass(frozen=True, slots=True, kw_only=True)
class KVLayout:
    """The per-head K and V widths a model's cache stores, as far as its config can say.

    ``modelled=False``: the config carries MLA fields this module cannot read as cache widths.
    ``k_dim``/``v_dim`` are None on such a layout and on a model with no MLA fields at all.
    """

    k_dim: int | None
    v_dim: int | None
    modelled: bool


def _positive_int(value: object) -> int | None:
    """``value`` if it is a positive non-bool int, else None."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def kv_layout(model_type: object, field: Callable[[str], object]) -> KVLayout:
    """Resolve the cached widths from ``model_type`` and a config or ``ModelArgs`` reader.

    Keyed on ``model_type``, because the field names do not say what the cache stores: models
    with these same fields cache a shared latent, or take V from another source.
    """
    if isinstance(model_type, str) and model_type in _EXPANDED_KV_MODEL_TYPES:
        k_nope = _positive_int(field("qk_nope_head_dim"))
        k_rope = _positive_int(field("qk_rope_head_dim"))
        v_dim = _positive_int(field("v_head_dim"))
        if k_nope is None or k_rope is None or v_dim is None:
            return KVLayout(k_dim=None, v_dim=None, modelled=False)
        return KVLayout(k_dim=k_nope + k_rope, v_dim=v_dim, modelled=True)
    has_mla_field = any(field(name) is not None for name in _MLA_FIELDS)
    return KVLayout(k_dim=None, v_dim=None, modelled=not has_mla_field)
