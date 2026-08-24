"""Tests for the KV-cache method seam (no MLX model, no network, no third-party port)."""

import json
import sys

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache, RotatingKVCache
from tests.probes.fake_turboquant import install_fake_port

from mlx_quant_fidelity.errors import (
    CacheNotQuantizableError,
    CompareConfigError,
    MethodUnavailableError,
)
from mlx_quant_fidelity.probes.kv_methods import (
    METHODS,
    TURBOQUANT_PINNED_COMMIT,
    StockKVMethod,
    TurboQuantKVMethod,
    parse_method_spec,
    stored_state_bytes,
    turboquant_bytes_per_token,
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
    [
        "",
        "4",
        "stock:4",
        "4:0",
        "0:64",
        "4:64:1",
        "four:64",
        "nosuch:4",
        "stock:4:64:9",
        "²:64",  # unicode superscript digit: isdigit() but not isascii()
        "turboquant:²",
    ],
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


# --- TurboQuantKVMethod: pure ----------------------------------------------------


def test_parse_turboquant_specs():
    assert parse_method_spec("turboquant:4") == TurboQuantKVMethod(bits=4, seed=42)
    assert parse_method_spec("turboquant:4:42") == TurboQuantKVMethod(bits=4, seed=42)
    assert parse_method_spec("turboquant:3:7") == TurboQuantKVMethod(bits=3, seed=7)
    assert METHODS["turboquant"] is TurboQuantKVMethod


@pytest.mark.parametrize(
    "spec", ["turboquant", "turboquant:8", "turboquant:4:0", "turboquant:4:7:1"]
)
def test_parse_turboquant_rejects(spec):
    with pytest.raises(CompareConfigError):
        parse_method_spec(spec)


def test_turboquant_name_label_and_params():
    assert TurboQuantKVMethod(bits=4).name == "turboquant"
    assert TurboQuantKVMethod(bits=4).label == "turboquant:4"
    assert TurboQuantKVMethod(bits=4, seed=7).label == "turboquant:4:7"
    assert TurboQuantKVMethod(bits=3).params == {"bits": 3, "seed": 42}


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_turboquant_check_accepts_power_of_two_up_to_256(head_dim):
    assert TurboQuantKVMethod(bits=4).check(head_dim=head_dim, model_type="llama") == []


@pytest.mark.parametrize("head_dim", [None, 0, 48, 80, 96, 160, 512])
def test_turboquant_check_raises_on_unsupported_head_dim(head_dim):
    with pytest.raises(CacheNotQuantizableError, match="TurboQuant-MLX kernels"):
        TurboQuantKVMethod(bits=4).check(head_dim=head_dim, model_type="phi3")


def test_turboquant_bytes_per_token_gates_geometry_too():
    with pytest.raises(CacheNotQuantizableError, match="TurboQuant-MLX kernels"):
        TurboQuantKVMethod(bits=4).bytes_per_token(n_layers=1, n_kv_heads=1, head_dim=96)


def test_turboquant_constructor_rejects_bits_outside_2_3_4():
    with pytest.raises(ValueError, match="bits"):
        TurboQuantKVMethod(bits=8)


@pytest.mark.parametrize(
    ("bits", "head_dim", "expected"),
    [
        # per (token, head): packed_dim*4 + 4 B fp32 norm; K and V; 16 layers x 8 heads
        (4, 64, 9216),  # (8*4+4)*2*8*16
        (3, 64, 8192),  # ceil(64/10)=7 -> (7*4+4)*2*128
        (2, 64, 5120),  # ceil(64/16)=4 -> (4*4+4)*2*128
        (4, 128, 2 * 8 * 16 * (16 * 4 + 4)),
    ],
)
def test_turboquant_bytes_per_token_hand_values(bits, head_dim, expected):
    assert (
        turboquant_bytes_per_token(n_layers=16, n_kv_heads=8, head_dim=head_dim, bits=bits)
        == expected
    )
    assert (
        TurboQuantKVMethod(bits=bits).bytes_per_token(n_layers=16, n_kv_heads=8, head_dim=head_dim)
        == expected
    )


def test_turboquant_report_warnings_name_the_numerics_path(monkeypatch):
    monkeypatch.setattr("mlx_quant_fidelity.probes.kv_methods._turboquant_direct_url", lambda: None)
    notes = TurboQuantKVMethod(bits=4).report_warnings()
    assert any("standard SDPA" in n and "uniform-bit" in n and "2.3" in n for n in notes)
    assert any("installed commit unknown is not the pinned" in n for n in notes)


# --- TurboQuantKVMethod: adapter wiring through the fake port -----------------------


def test_missing_port_raises_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "turboquant_mlx", None)  # forces ImportError
    with pytest.raises(MethodUnavailableError, match=TURBOQUANT_PINNED_COMMIT):
        TurboQuantKVMethod(bits=4).probe_capability([KVCache()])


def test_squatter_package_is_named(monkeypatch):
    install_fake_port(monkeypatch, no_cache_module=True)
    with pytest.raises(MethodUnavailableError, match="squatter"):
        TurboQuantKVMethod(bits=4).probe_capability([KVCache()])


def test_missing_version_is_named(monkeypatch):
    install_fake_port(monkeypatch, no_version=True)
    with pytest.raises(MethodUnavailableError, match="__version__"):
        TurboQuantKVMethod(bits=4).probe_capability([KVCache()])


def test_missing_attribute_is_named(monkeypatch):
    install_fake_port(monkeypatch, missing=("trim",))
    with pytest.raises(MethodUnavailableError, match="trim"):
        TurboQuantKVMethod(bits=4).probe_capability([KVCache()])


def test_bits_attribute_is_rejected_to_keep_standard_sdpa(monkeypatch):
    install_fake_port(monkeypatch, with_bits_attr=True)
    with pytest.raises(MethodUnavailableError, match="exposes `bits`"):
        TurboQuantKVMethod(bits=4).probe_capability([KVCache()])


def test_behavioural_contract_catches_wrong_offset(monkeypatch):
    install_fake_port(monkeypatch, wrong_behaviour=True)
    with pytest.raises(MethodUnavailableError, match="behaviour differs"):
        TurboQuantKVMethod(bits=4).probe_capability([KVCache()])


def test_behavioural_contract_catches_sticky_trim(monkeypatch):
    install_fake_port(monkeypatch, bad_trim=True)
    with pytest.raises(MethodUnavailableError, match="did not release"):
        TurboQuantKVMethod(bits=4).probe_capability([KVCache()])


def test_probe_capability_rejects_non_plain_kvcache(monkeypatch):
    install_fake_port(monkeypatch)
    with pytest.raises(CacheNotQuantizableError, match="RotatingKVCache"):
        TurboQuantKVMethod(bits=4).probe_capability([KVCache(), RotatingKVCache(max_size=8)])


def test_probe_capability_passes_on_plain_kvcache(monkeypatch):
    install_fake_port(monkeypatch)
    TurboQuantKVMethod(bits=4).probe_capability([KVCache(), KVCache()])


def test_make_cache_passes_pinned_knobs(monkeypatch):
    install_fake_port(monkeypatch)
    caches = TurboQuantKVMethod(bits=3, seed=7).make_cache(n_layers=2)
    assert len(caches) == 2
    c = caches[0]
    assert (c.quant_bits, c.seed, c.fused, c.v_only, c.sparse_v_threshold) == (
        3,
        7,
        False,
        False,
        None,
    )


def test_convert_prefix_replays_state_then_trims(monkeypatch):
    install_fake_port(monkeypatch)
    fp = KVCache()
    fp.update_and_fetch(
        mx.ones((1, 2, 5, 64)), mx.ones((1, 2, 5, 64))
    )  # offset 5 inside a 256 step
    (new,) = TurboQuantKVMethod(bits=4).convert_prefix([fp])
    assert new.offset == 5  # replayed exactly the stored prefix, not the padded buffer
    assert new.trim_calls == [0]  # dequant buffers dropped before segment 2
    assert new._k_deq_buf is None


def test_measured_bytes_uses_state_not_nbytes(monkeypatch):
    cls = install_fake_port(monkeypatch)
    c = cls(bits=4)
    c.update_and_fetch(mx.zeros((1, 8, 511, 64)), mx.zeros((1, 8, 511, 64)))
    # one layer, 8 heads, 511 tokens: (8*4+4) B * 2 * 8 * 511
    assert TurboQuantKVMethod(bits=4).measured_bytes([c]) == 36 * 2 * 8 * 511


def test_provenance_reads_commit_from_direct_url(monkeypatch):
    install_fake_port(monkeypatch)
    monkeypatch.setattr(
        "mlx_quant_fidelity.probes.kv_methods._turboquant_dist_version", lambda: "0.3.0"
    )
    monkeypatch.setattr(
        "mlx_quant_fidelity.probes.kv_methods._turboquant_direct_url",
        lambda: json.dumps({"vcs_info": {"commit_id": "abc"}}),
    )
    prov = TurboQuantKVMethod(bits=4, seed=7).provenance()
    assert prov["package"] == "turboquant-mlx"
    assert prov["dist_version"] == "0.3.0"
    assert prov["module_version"] == "0.3.0"
    assert prov["commit"] == "abc"
    assert prov["pinned_commit"] == TURBOQUANT_PINNED_COMMIT
    assert (prov["k_seed"], prov["v_seed"], prov["fused"], prov["v_only"]) == (
        "7",
        "8",
        "false",
        "false",
    )
    assert any(
        "abc" in w and TURBOQUANT_PINNED_COMMIT in w
        for w in TurboQuantKVMethod(bits=4).report_warnings()
    )


def test_provenance_unknown_when_metadata_absent(monkeypatch):
    install_fake_port(monkeypatch)
    monkeypatch.setattr(
        "mlx_quant_fidelity.probes.kv_methods._turboquant_dist_version", lambda: "unknown"
    )
    monkeypatch.setattr("mlx_quant_fidelity.probes.kv_methods._turboquant_direct_url", lambda: None)
    prov = TurboQuantKVMethod(bits=4).provenance()
    assert prov["commit"] == "unknown"
    assert prov["dist_version"] == "unknown"


def test_no_pin_warning_when_installed_commit_matches(monkeypatch):
    monkeypatch.setattr(
        "mlx_quant_fidelity.probes.kv_methods._turboquant_direct_url",
        lambda: json.dumps({"vcs_info": {"commit_id": TURBOQUANT_PINNED_COMMIT}}),
    )
    assert not any("is not the pinned" in w for w in TurboQuantKVMethod(bits=4).report_warnings())
