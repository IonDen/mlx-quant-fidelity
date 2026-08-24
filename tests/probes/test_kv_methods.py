"""Tests for the KV-cache method seam (no MLX model, no network, no third-party port)."""

import mlx.core as mx
import pytest

from mlx_quant_fidelity.errors import (
    CacheNotQuantizableError,
    CompareConfigError,
)
from mlx_quant_fidelity.probes.kv_methods import (
    METHODS,
    StockKVMethod,
    parse_method_spec,
    stored_state_bytes,
)

# --- parse_method_spec -------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("4:64", StockKVMethod(bits=4, group_size=64)),
        ("stock:8:32", StockKVMethod(bits=8, group_size=32)),
    ],
)
def test_parse_stock_specs(spec, expected):
    assert parse_method_spec(spec) == expected


@pytest.mark.parametrize(
    "spec",
    ["", "4", "stock:4", "4:0", "0:64", "4:64:1", "four:64", "nosuch:4", "stock:4:64:9"],
)
def test_parse_rejects_malformed(spec):
    with pytest.raises(CompareConfigError):
        parse_method_spec(spec)


def test_methods_map_names_stock():
    assert METHODS["stock"] is StockKVMethod


# --- StockKVMethod: pure surface ------------------------------------------------


def test_stock_name_label_and_params():
    m = StockKVMethod(bits=4, group_size=64)
    assert m.name == "stock"
    assert m.label == "4:64"
    assert m.params == {"bits": 4, "group_size": 64}
    assert m == StockKVMethod(4, 64)  # name is not an init field


def test_stock_check_rejects_unsupported_bits():
    with pytest.raises(CacheNotQuantizableError, match="unsupported kv_bits=5"):
        StockKVMethod(bits=5, group_size=64).check(head_dim=64, model_type="llama")


def test_stock_check_bits_zero_is_a_clean_error_not_zero_division():
    with pytest.raises(CacheNotQuantizableError, match="unsupported kv_bits=0"):
        StockKVMethod(bits=0, group_size=64).check(head_dim=128, model_type="llama")


def test_stock_check_rejects_non_divisor_group_size():
    with pytest.raises(CacheNotQuantizableError, match="does not divide"):
        StockKVMethod(bits=4, group_size=48).check(head_dim=64, model_type="llama")


def test_stock_check_warns_when_head_dim_unknown():
    warnings = StockKVMethod(bits=4, group_size=64).check(head_dim=None, model_type="weird")
    assert len(warnings) == 1
    assert "unverified for 'weird'" in warnings[0]


def test_stock_check_rejects_packed_width_mismatch():
    # bits=6 at head_dim=128: 128 // (32 // 6) = 25 != 128 * 6 // 32 = 24 (mlx-lm truncation bug)
    with pytest.raises(CacheNotQuantizableError, match="packed-width"):
        StockKVMethod(bits=6, group_size=64).check(head_dim=128, model_type="llama")


def test_stock_check_unsupported_bits_wins_over_packed_width():
    with pytest.raises(CacheNotQuantizableError, match="unsupported kv_bits=5"):
        StockKVMethod(bits=5, group_size=64).check(head_dim=128, model_type="llama")


def test_stock_check_clean_returns_no_warnings():
    assert StockKVMethod(bits=4, group_size=64).check(head_dim=64, model_type="llama") == []


def test_stock_bytes_per_token_matches_cost_formula():
    # Llama-3.2-1B geometry, hand-computed: 2*16*8*64 elements * (4/8 + 4/64) = 9216
    m = StockKVMethod(bits=4, group_size=64)
    assert m.bytes_per_token(n_layers=16, n_kv_heads=8, head_dim=64) == 9216
    # 8:64 -> 16384 * (1 + 4/64) = 17408
    assert (
        StockKVMethod(bits=8, group_size=64).bytes_per_token(n_layers=16, n_kv_heads=8, head_dim=64)
        == 17408
    )


def test_stock_provenance_names_mlx_lm():
    prov = StockKVMethod(bits=4, group_size=64).provenance()
    assert prov["package"] == "mlx-lm"
    assert prov["version"]


def test_stock_report_warnings_is_empty():
    assert StockKVMethod(bits=4, group_size=64).report_warnings() == []


# --- StockKVMethod: probe_capability (ported from the removed _cache_is_quantizable tests) ---


class _NoQuantCache:
    pass


class _RaisingQuantCache:
    def to_quantized(self, **_kwargs):
        raise NotImplementedError("sliding window")


class _ValueErrorCache:
    def to_quantized(self, **_kwargs):
        raise ValueError("bad group")


class _OkCache:
    def to_quantized(self, **_kwargs):
        return self


def test_probe_capability_accepts_non_raising_to_quantized():
    StockKVMethod(bits=4, group_size=64).probe_capability([_OkCache(), _OkCache()])


def test_probe_capability_flags_missing_to_quantized():
    with pytest.raises(CacheNotQuantizableError, match="_NoQuantCache has no to_quantized"):
        StockKVMethod(bits=4, group_size=64).probe_capability([_OkCache(), _NoQuantCache()])


def test_probe_capability_flags_not_implemented():
    with pytest.raises(CacheNotQuantizableError, match="NYI: sliding window"):
        StockKVMethod(bits=4, group_size=64).probe_capability([_RaisingQuantCache()])


def test_probe_capability_flags_value_error():
    with pytest.raises(CacheNotQuantizableError, match="cannot quantize at group_size=64, bits=4"):
        StockKVMethod(bits=4, group_size=64).probe_capability([_ValueErrorCache()])


# --- stored_state_bytes -----------------------------------------------------------


class _StateCache:
    def __init__(self, state):
        self.state = state


def test_stored_state_bytes_walks_nested_state_and_skips_non_arrays():
    k = mx.zeros((1, 2, 3, 8), dtype=mx.uint32)  # 192 B
    s = mx.zeros((1, 2, 3, 1), dtype=mx.float16)  # 12 B
    stock_like = _StateCache(((k, s, s), (k, s, s)))  # 2 * (192 + 24) = 432
    turbo_like = _StateCache([k, s, k, s])  # 408
    empty = _StateCache(())
    none_leaf = _StateCache((None, None))
    assert stored_state_bytes([stock_like]) == 432
    assert stored_state_bytes([turbo_like]) == 408
    assert stored_state_bytes([stock_like, turbo_like]) == 840
    assert stored_state_bytes([empty]) == 0
    assert stored_state_bytes([none_leaf]) == 0
