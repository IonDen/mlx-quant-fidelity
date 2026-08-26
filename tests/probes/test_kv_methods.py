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
    AffineKVMethod,
    StockKVMethod,
    TurboQuantKVMethod,
    TurboQuantVOnlyKVMethod,
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


def test_positive_ints_example_matches_the_method():
    with pytest.raises(CompareConfigError, match=r"e\.g\. 4:64"):
        parse_method_spec("4:x")
    with pytest.raises(CompareConfigError, match=r"e\.g\. turboquant:4"):
        parse_method_spec("turboquant:x")


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


def test_stock_control_method_returns_affine_at_same_bits_both_sides():
    """Reds if control_method returns the wrong class, the wrong bits, or a mismatched group_size."""
    control = StockKVMethod(bits=4, group_size=32).control_method()
    assert isinstance(control, AffineKVMethod)
    assert control.k_bits == 4
    assert control.v_bits == 4
    assert control.group_size == 32


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


@pytest.mark.parametrize("seed", [0, -1])
def test_turboquant_constructor_rejects_non_positive_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        TurboQuantKVMethod(bits=4, seed=seed)


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


def test_provenance_missing_port_is_package_rooted(monkeypatch):
    monkeypatch.setitem(sys.modules, "turboquant_mlx", None)  # forces ImportError
    with pytest.raises(MethodUnavailableError, match=TURBOQUANT_PINNED_COMMIT):
        TurboQuantKVMethod(bits=4).provenance()


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


def test_behavioural_contract_requires_populated_buffers(monkeypatch):
    install_fake_port(monkeypatch, no_deq_buffers=True)
    with pytest.raises(MethodUnavailableError, match="no populated"):
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


# --- AffineKVMethod ----------------------------------------------------------------


def test_affine_label_and_params():
    """Reds if the label elides group at the wrong default or params lose a key."""
    m = AffineKVMethod(k_bits=8, v_bits=4)
    assert m.name == "affine"
    assert m.label == "affine:8:4"
    assert m.params == {"k_bits": 8, "v_bits": 4, "group_size": 64}
    assert AffineKVMethod(k_bits=8, v_bits=4, group_size=32).label == "affine:8:4:32"


def test_affine_bytes_per_token_hand_values():
    """Reds on any drift in the per-side cost formula.

    1L/1H/64D g64: k8 side 64*1.0625 = 68; v4 side 64*0.5625 = 36 -> 104.
    Llama-3.2-1B (16L/8H/64D): 8704 + 4608 = 13312.
    """
    m = AffineKVMethod(k_bits=8, v_bits=4)
    assert m.bytes_per_token(n_layers=1, n_kv_heads=1, head_dim=64) == 104
    assert m.bytes_per_token(n_layers=16, n_kv_heads=8, head_dim=64) == 13312


def test_affine_equal_bits_cost_equals_stock():
    """Reds if affine:b:b:g and stock b:g ever cost differently (same layout)."""
    from mlx_quant_fidelity.probes.kv_methods import StockKVMethod

    aff = AffineKVMethod(k_bits=4, v_bits=4)
    stock = StockKVMethod(bits=4, group_size=64)
    for geo in (
        {"n_layers": 1, "n_kv_heads": 1, "head_dim": 64},
        {"n_layers": 16, "n_kv_heads": 8, "head_dim": 64},
    ):
        assert aff.bytes_per_token(**geo) == stock.bytes_per_token(**geo)


def test_affine_check_gates_divisibility_and_bits():
    """Reds if the divisibility gate or per-side bit validation is dropped, or bits 6 banned."""
    from mlx_quant_fidelity.errors import CacheNotQuantizableError

    with pytest.raises(CacheNotQuantizableError, match="does not divide"):
        AffineKVMethod(k_bits=8, v_bits=4, group_size=48).check(head_dim=64, model_type="llama")
    with pytest.raises(ValueError, match="k_bits"):
        AffineKVMethod(k_bits=5, v_bits=4)
    with pytest.raises(ValueError, match="v_bits"):
        AffineKVMethod(k_bits=8, v_bits=7)
    assert AffineKVMethod(k_bits=6, v_bits=6).check(head_dim=128, model_type="llama") == []


def test_affine_warns_no_shipped_runtime():
    """Reds if the hypothetical-deployment warning is dropped from the report."""
    notes = AffineKVMethod(k_bits=8, v_bits=4).report_warnings()
    assert any("no shipped runtime" in n for n in notes)


def test_affine_provenance_names_the_fork():
    """Reds if provenance loses the quantizer package or the known-runtime pointer."""
    prov = AffineKVMethod(k_bits=8, v_bits=4).provenance()
    assert prov["package"] == "mlx"
    assert "arozanov/mlx-lm" in prov["known_runtime"]


def test_affine_measured_bytes_matches_analytic():
    """Reds if stored state and the analytic formula disagree (layout or rounding drift)."""
    m = AffineKVMethod(k_bits=8, v_bits=4)
    cache = m.make_cache(n_layers=1)
    k = mx.zeros((1, 1, 8, 64), dtype=mx.float16)
    cache[0].update_and_fetch(k, k)
    mx.eval(cache[0].state)
    assert m.measured_bytes(cache) == m.bytes_per_token(n_layers=1, n_kv_heads=1, head_dim=64) * 8


def test_parse_method_spec_affine():
    """Reds if the parse arm accepts wrong arity or loses the group default."""
    from mlx_quant_fidelity.errors import CompareConfigError
    from mlx_quant_fidelity.probes.kv_methods import parse_method_spec

    m = parse_method_spec("affine:8:4")
    assert isinstance(m, AffineKVMethod)
    assert (m.k_bits, m.v_bits, m.group_size) == (8, 4, 64)
    assert parse_method_spec("affine:8:2:32").group_size == 32
    with pytest.raises(CompareConfigError):
        parse_method_spec("affine:8")
    with pytest.raises(CompareConfigError):
        parse_method_spec("affine:8:4:64:1")


# --- TurboQuantVOnlyKVMethod: pure -------------------------------------------------
# The seven tests below are Appendix C of the task-6 brief, verbatim.


def test_vonly_label_params_and_validation():
    """Reds if the id/label/params drift or the port's bit/seed constraints are dropped."""
    m = TurboQuantVOnlyKVMethod(v_bits=3)
    assert m.name == "turboquant-vonly"
    assert m.label == "turboquant-vonly:3"
    assert TurboQuantVOnlyKVMethod(v_bits=3, seed=7).label == "turboquant-vonly:3:7"
    assert m.params == {"v_bits": 3, "seed": 42}
    with pytest.raises(ValueError):  # noqa: PT011 -- Appendix C verbatim
        TurboQuantVOnlyKVMethod(v_bits=8)
    with pytest.raises(ValueError):  # noqa: PT011 -- Appendix C verbatim
        TurboQuantVOnlyKVMethod(v_bits=3, seed=0)


def test_vonly_bytes_per_token_counts_the_duplicate_fp16_v():
    """Reds if the cost model stops counting the port's stored-but-unused fp16 V.

    1L/1H/64D: fp16 K 128 + fp16 V 128 + (packed_dim(64,4)=8 -> 36) = 292 (v4); 288 (v3).
    Llama-3.2-1B (16L/8H/64D): 292*128 = 37376 (v4) — ABOVE plain fp16's 32768; 36864 (v3).
    """
    assert (
        TurboQuantVOnlyKVMethod(v_bits=4).bytes_per_token(n_layers=1, n_kv_heads=1, head_dim=64)
        == 292
    )
    assert (
        TurboQuantVOnlyKVMethod(v_bits=3).bytes_per_token(n_layers=1, n_kv_heads=1, head_dim=64)
        == 288
    )
    assert (
        TurboQuantVOnlyKVMethod(v_bits=4).bytes_per_token(n_layers=16, n_kv_heads=8, head_dim=64)
        == 37376
    )


def test_vonly_provenance_seed_plus_one():
    """Reds if provenance records the constructor seed instead of the port's seed+1."""
    prov = TurboQuantVOnlyKVMethod(v_bits=3, seed=41).provenance()
    assert prov["v_seed"] == "42"
    assert "k_seed" not in prov


def test_vonly_check_gates_head_dim():
    """Reds if the V path's power-of-two/<=256 kernel gate is dropped."""
    from mlx_quant_fidelity.errors import CacheNotQuantizableError

    with pytest.raises(CacheNotQuantizableError, match="power-of-two"):
        TurboQuantVOnlyKVMethod(v_bits=3).check(head_dim=96, model_type="x")


def test_vonly_warns_duplicate_storage():
    """Reds if the duplicate-fp16-V storage warning is dropped from the report."""
    notes = " ".join(TurboQuantVOnlyKVMethod(v_bits=3).report_warnings())
    assert "fp16 copy of V" in notes and "quality, not memory" in notes  # noqa: PT018


def test_parse_method_spec_vonly():
    """Reds if the parse arm loses the seed default or the two-arg form."""
    from mlx_quant_fidelity.probes.kv_methods import parse_method_spec

    m = parse_method_spec("turboquant-vonly:3")
    assert isinstance(m, TurboQuantVOnlyKVMethod) and (m.v_bits, m.seed) == (3, 42)  # noqa: PT018
    assert parse_method_spec("turboquant-vonly:4:7").seed == 7


def test_uniform_turboquant_warning_names_vonly_and_the_adaptive_trap():
    """Reds if the uniform method's scope note is not updated for 0.7.0."""
    from mlx_quant_fidelity.probes.kv_methods import TurboQuantKVMethod

    notes = " ".join(TurboQuantKVMethod(bits=4).report_warnings())
    assert "silently ignores" in notes and "turboquant-vonly" in notes  # noqa: PT018


# --- TurboQuantVOnlyKVMethod: additional pure coverage ------------------------------


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_vonly_check_accepts_power_of_two_up_to_256(head_dim):
    assert TurboQuantVOnlyKVMethod(v_bits=4).check(head_dim=head_dim, model_type="llama") == []


@pytest.mark.parametrize("seed", [0, -1])
def test_vonly_constructor_rejects_non_positive_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        TurboQuantVOnlyKVMethod(v_bits=3, seed=seed)


def test_vonly_constructor_rejects_bits_outside_2_3_4():
    with pytest.raises(ValueError, match="bits"):
        TurboQuantVOnlyKVMethod(v_bits=8)


@pytest.mark.parametrize(
    "spec",
    ["turboquant-vonly", "turboquant-vonly:8", "turboquant-vonly:4:0", "turboquant-vonly:4:7:1"],
)
def test_parse_vonly_rejects_malformed(spec):
    with pytest.raises(CompareConfigError):
        parse_method_spec(spec)


def test_methods_map_does_not_yet_include_vonly():
    """Registration is a later task; the parse arm must not depend on METHODS containing it."""
    assert "turboquant-vonly" not in METHODS


# --- TurboQuantVOnlyKVMethod: adapter wiring through the extended fake port ---------


def test_vonly_missing_port_raises_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "turboquant_mlx", None)  # forces ImportError
    with pytest.raises(MethodUnavailableError, match=TURBOQUANT_PINNED_COMMIT):
        TurboQuantVOnlyKVMethod(v_bits=4).probe_capability([KVCache()])


def test_vonly_provenance_missing_port_is_package_rooted(monkeypatch):
    monkeypatch.setitem(sys.modules, "turboquant_mlx", None)  # forces ImportError
    with pytest.raises(MethodUnavailableError, match=TURBOQUANT_PINNED_COMMIT):
        TurboQuantVOnlyKVMethod(v_bits=4).provenance()


def test_vonly_squatter_package_is_named(monkeypatch):
    install_fake_port(monkeypatch, no_vonly_module=True)
    with pytest.raises(MethodUnavailableError, match="squatter"):
        TurboQuantVOnlyKVMethod(v_bits=4).probe_capability([KVCache()])


def test_vonly_missing_version_is_named(monkeypatch):
    install_fake_port(monkeypatch, no_version=True)
    with pytest.raises(MethodUnavailableError, match="__version__"):
        TurboQuantVOnlyKVMethod(v_bits=4).probe_capability([KVCache()])


def test_vonly_missing_attribute_is_named(monkeypatch):
    install_fake_port(monkeypatch, missing=("trim",))
    with pytest.raises(MethodUnavailableError, match="trim"):
        TurboQuantVOnlyKVMethod(v_bits=4).probe_capability([KVCache()])


def test_vonly_bits_attribute_is_rejected_to_keep_standard_sdpa(monkeypatch):
    install_fake_port(monkeypatch, with_bits_attr=True)
    with pytest.raises(MethodUnavailableError, match="exposes `bits`"):
        TurboQuantVOnlyKVMethod(v_bits=4).probe_capability([KVCache()])


def test_vonly_behavioural_contract_catches_wrong_offset(monkeypatch):
    install_fake_port(monkeypatch, wrong_behaviour=True)
    with pytest.raises(MethodUnavailableError, match="behaviour differs"):
        TurboQuantVOnlyKVMethod(v_bits=4).probe_capability([KVCache()])


def test_vonly_behavioural_contract_catches_sticky_trim(monkeypatch):
    install_fake_port(monkeypatch, bad_trim=True)
    with pytest.raises(MethodUnavailableError, match="did not release"):
        TurboQuantVOnlyKVMethod(v_bits=4).probe_capability([KVCache()])


def test_vonly_behavioural_contract_requires_populated_buffer(monkeypatch):
    install_fake_port(monkeypatch, no_deq_buffers=True)
    with pytest.raises(MethodUnavailableError, match="no populated"):
        TurboQuantVOnlyKVMethod(v_bits=4).probe_capability([KVCache()])


def test_vonly_probe_capability_rejects_non_plain_kvcache(monkeypatch):
    install_fake_port(monkeypatch)
    with pytest.raises(CacheNotQuantizableError, match="RotatingKVCache"):
        TurboQuantVOnlyKVMethod(v_bits=4).probe_capability([KVCache(), RotatingKVCache(max_size=8)])


def test_vonly_probe_capability_passes_on_plain_kvcache(monkeypatch):
    install_fake_port(monkeypatch)
    TurboQuantVOnlyKVMethod(v_bits=4).probe_capability([KVCache(), KVCache()])


def test_vonly_cache_cls_survives_a_state_property_that_raises_when_empty(monkeypatch):
    """Reds if the required-attrs guard checks hasattr(instance, attr) instead of the class.

    The real port's VOnlyTurboQuantCache.state is a property that raises AttributeError on a
    fresh/empty instance (it dereferences an inner KVCache's .keys.shape while keys is still
    None). hasattr(instance, "state") silently swallows that AttributeError and misreports the
    genuine port as missing the attribute (task-6 F1: this is exactly what happened against the
    real pinned port). The fake VOnly cache mimics this raising-when-empty shape by default.
    """
    install_fake_port(monkeypatch)
    import turboquant_mlx.v_only_cache as vonly_mod

    fresh = vonly_mod.VOnlyTurboQuantCache()
    with pytest.raises(AttributeError):
        _ = fresh.state  # sanity: the fake really mimics the real port's bug
    TurboQuantVOnlyKVMethod(v_bits=4).probe_capability([KVCache()])  # must not misreport "missing"


def test_vonly_make_cache_passes_pinned_knobs(monkeypatch):
    install_fake_port(monkeypatch)
    caches = TurboQuantVOnlyKVMethod(v_bits=3, seed=7).make_cache(n_layers=2)
    assert len(caches) == 2
    c = caches[0]
    assert (c.v_bits, c.seed, c.no_v_buffer) == (3, 7, False)


def test_vonly_convert_prefix_replays_state_then_trims(monkeypatch):
    install_fake_port(monkeypatch)
    fp = KVCache()
    fp.update_and_fetch(
        mx.ones((1, 2, 5, 64)), mx.ones((1, 2, 5, 64))
    )  # offset 5 inside a 256 step
    (new,) = TurboQuantVOnlyKVMethod(v_bits=4).convert_prefix([fp])
    assert new.offset == 5  # replayed exactly the stored prefix, not the padded buffer
    assert new.trim_calls == [0]  # dequant buffer dropped before segment 2
    assert new._v_tq._v_deq_buf is None


def test_vonly_measured_bytes_uses_state_not_nbytes(monkeypatch):
    install_fake_port(monkeypatch)
    import turboquant_mlx.v_only_cache as vonly_mod

    c = vonly_mod.VOnlyTurboQuantCache(bits=4)
    fp16 = mx.zeros((1, 8, 511, 64), dtype=mx.float16)  # a real forward's activation dtype
    c.update_and_fetch(fp16, fp16)
    # one layer, 8 heads, 511 tokens: fp16 K + fp16 V + (8*4+4) B packed V+norms, per head/token
    expected = (64 * 2 * 2 + (8 * 4 + 4)) * 8 * 511
    assert TurboQuantVOnlyKVMethod(v_bits=4).measured_bytes([c]) == expected


def test_vonly_provenance_reads_commit_from_direct_url(monkeypatch):
    install_fake_port(monkeypatch)
    monkeypatch.setattr(
        "mlx_quant_fidelity.probes.kv_methods._turboquant_dist_version", lambda: "0.3.0"
    )
    monkeypatch.setattr(
        "mlx_quant_fidelity.probes.kv_methods._turboquant_direct_url",
        lambda: json.dumps({"vcs_info": {"commit_id": "abc"}}),
    )
    prov = TurboQuantVOnlyKVMethod(v_bits=4, seed=7).provenance()
    assert prov["package"] == "turboquant-mlx"
    assert prov["dist_version"] == "0.3.0"
    assert prov["module_version"] == "0.3.0"
    assert prov["commit"] == "abc"
    assert prov["pinned_commit"] == TURBOQUANT_PINNED_COMMIT
    assert prov["v_seed"] == "8"
    assert "k_seed" not in prov
    assert any(
        "abc" in w and TURBOQUANT_PINNED_COMMIT in w
        for w in TurboQuantVOnlyKVMethod(v_bits=4).report_warnings()
    )


def test_vonly_provenance_unknown_when_metadata_absent(monkeypatch):
    install_fake_port(monkeypatch)
    monkeypatch.setattr(
        "mlx_quant_fidelity.probes.kv_methods._turboquant_dist_version", lambda: "unknown"
    )
    monkeypatch.setattr("mlx_quant_fidelity.probes.kv_methods._turboquant_direct_url", lambda: None)
    prov = TurboQuantVOnlyKVMethod(v_bits=4).provenance()
    assert prov["commit"] == "unknown"
    assert prov["dist_version"] == "unknown"


def test_vonly_no_pin_warning_when_installed_commit_matches(monkeypatch):
    monkeypatch.setattr(
        "mlx_quant_fidelity.probes.kv_methods._turboquant_direct_url",
        lambda: json.dumps({"vcs_info": {"commit_id": TURBOQUANT_PINNED_COMMIT}}),
    )
    assert not any(
        "is not the pinned" in w for w in TurboQuantVOnlyKVMethod(v_bits=4).report_warnings()
    )


# --- working_set_bytes --------------------------------------------------------------


def test_working_set_bytes_hand_values():
    """Reds on any drift in a per-method working-set formula or the step-256 padding rule."""
    geo = dict(window=512, n_layers=1, n_kv_heads=1, head_dim=64, dtype_bytes=2)  # noqa: C408 -- Appendix D verbatim
    assert StockKVMethod(bits=4, group_size=64).working_set_bytes(**geo) == 0
    assert TurboQuantKVMethod(bits=4).working_set_bytes(**geo) == 262144  # 4*512*64*2
    assert TurboQuantVOnlyKVMethod(v_bits=4).working_set_bytes(**geo) == 131072
    assert AffineKVMethod(k_bits=8, v_bits=4).working_set_bytes(**geo) == 131072
    # step-256 padding applies to the port's buffers, not to affine's exact-window transients
    assert (
        TurboQuantKVMethod(bits=4).working_set_bytes(
            window=300, n_layers=1, n_kv_heads=1, head_dim=64, dtype_bytes=2
        )
        == 262144
    )
    assert (
        AffineKVMethod(k_bits=8, v_bits=4).working_set_bytes(
            window=300, n_layers=1, n_kv_heads=1, head_dim=64, dtype_bytes=2
        )
        == 76800
    )
