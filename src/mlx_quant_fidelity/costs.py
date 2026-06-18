"""Pure cost-denominator functions for `compare` (the memory-normalized Pareto x-axis)."""


def kv_bytes_per_token(
    *, n_layers: int, n_kv_heads: int, head_dim: int, bits: int, group_size: int
) -> int:
    """KV-cache bytes per token for a quantized cache, incl. per-group scale+bias overhead.

    Per element: ``bits/8`` packed + a fp16 scale (2 B) and fp16 bias (2 B) amortized over
    ``group_size``. Counts K and V across every layer. The group overhead is load-bearing — a
    smaller group_size buys accuracy but costs bytes, and the Pareto must see it. See
    docs/ranking-principles.md (the cost axis).
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    elements_per_token = 2 * n_layers * n_kv_heads * head_dim  # K and V
    bytes_per_element = bits / 8 + 4 / group_size  # 4 B = fp16 scale + fp16 bias per group
    return round(elements_per_token * bytes_per_element)
