"""Tests for the KV-cache method seam (no MLX model, no network, no third-party port)."""

import json
import sys

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache, RotatingKVCache
from tests.probes.fake_kv_method import FakeKVMethod
from tests.probes.fake_turboquant import install_fake_port

from mlx_quant_fidelity.costs import KVLayout
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
    _gate_turboquant_layout,
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
        StockKVMethod(bits=4, group_size=128).check(head_dim=64, model_type="llama")


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


def test_probe_capability_allows_partial_coverage():
    # RED if probe_capability still refuses a model with >=1 quantizable layer (0053):
    # a mixed cache (one quantizable, one not) must NOT raise — it is a partial measurement.
    StockKVMethod(bits=4, group_size=64).probe_capability([_OkCache(), _NoQuantCache()])


def test_probe_capability_raises_when_zero_layers_quantizable():
    # RED if a fully-unsupported cache stops refusing: with no quantizable layer there is
    # nothing to measure, so it must still refuse.
    with pytest.raises(CacheNotQuantizableError, match="0 of 2"):
        StockKVMethod(bits=4, group_size=64).probe_capability(
            [_NoQuantCache(), _RaisingQuantCache()]
        )


def test_probe_capability_flags_not_implemented():
    # RED if the zero-quantizable refusal drops the per-layer reason (single sliding-window layer).
    with pytest.raises(CacheNotQuantizableError, match="NYI: sliding window"):
        StockKVMethod(bits=4, group_size=64).probe_capability([_RaisingQuantCache()])


def test_probe_capability_flags_value_error():
    # RED if the zero-quantizable refusal drops the group/bits reason (single bad-geometry layer).
    with pytest.raises(CacheNotQuantizableError, match="cannot quantize at group_size=64, bits=4"):
        StockKVMethod(bits=4, group_size=64).probe_capability([_ValueErrorCache()])


def test_layer_coverage_partitions_mixed_layers():
    # RED until layer_coverage exists: it must report which layer indices are quantizable and
    # tally the skipped layer types (the data the partial report and its note are built from).
    cov = StockKVMethod(bits=4, group_size=64).layer_coverage(
        [_OkCache(), _NoQuantCache(), _RaisingQuantCache(), _OkCache()]
    )
    assert cov.total == 4
    assert cov.quantized_indices == (0, 3)
    assert cov.skipped_types == {"_NoQuantCache": 1, "_RaisingQuantCache": 1}
    assert cov.is_partial is True


def test_layer_coverage_all_quantizable_is_not_partial():
    # RED if a fully-quantizable model is ever flagged partial (would wrongly stamp the note /
    # change the committed headline).
    cov = StockKVMethod(bits=4, group_size=64).layer_coverage([_OkCache(), _OkCache()])
    assert cov.total == 2
    assert cov.quantized_indices == (0, 1)
    assert cov.skipped_types == {}
    assert cov.is_partial is False


class _QuantizableCache:
    """A layer whose to_quantized returns a distinct object (so we can tell it was replaced)."""

    def __init__(self):
        self.result = object()

    def to_quantized(self, **_kwargs):
        return self.result


def test_make_partial_cache_quantizes_only_supported_layers():
    # RED until make_partial_cache exists: supported layers are replaced by their quantized form;
    # skipped layers (e.g. sliding-window) pass through unchanged so the mixed forward still runs.
    sm = StockKVMethod(bits=4, group_size=64)
    q0, sliding, q2 = _QuantizableCache(), _RaisingQuantCache(), _QuantizableCache()
    fp = [q0, sliding, q2]
    cov = sm.layer_coverage(fp)
    assert cov.quantized_indices == (0, 2)
    out = sm.make_partial_cache(fp, cov)
    assert out[0] is q0.result  # quantized
    assert out[1] is sliding  # untouched
    assert out[2] is q2.result  # quantized


def test_layer_coverage_note_states_partial_and_hypothetical():
    # RED until note() exists: the report caveat must state the coverage fraction, name the
    # skipped layer type(s), and that this is NOT a config mlx-lm ships (the honesty catch).
    cov = StockKVMethod(bits=4, group_size=64).layer_coverage(
        [_OkCache(), _RaisingQuantCache(), _RaisingQuantCache()]
    )
    note = cov.note()
    assert "1 of 3" in note
    assert "2 _RaisingQuantCache" in note
    assert "hypothetical" in note
    assert "not a shipping configuration" in note


def test_layer_coverage_note_lists_multiple_skip_types_in_stable_order():
    # RED if note() loses its sorted() over skipped_types: dict-iteration order would make the
    # rendered caveat non-deterministic across runs.
    cov = StockKVMethod(bits=4, group_size=64).layer_coverage(
        [_OkCache(), _RaisingQuantCache(), _NoQuantCache()]
    )
    note = cov.note()
    assert "1 _NoQuantCache" in note
    assert "1 _RaisingQuantCache" in note
    assert note.index("_NoQuantCache") < note.index("_RaisingQuantCache")  # sorted, deterministic


def test_layer_coverage_note_for_skip_only_layers_says_mlx_lm_ships_the_mix():
    # RED until note() branches on the skip reason: a layer with NO to_quantized (state-space,
    # chunked) is what mlx-lm's own gate skips silently, so the mixed cache IS what mlx-lm ships
    # once conversion starts. Calling it "not a shipping configuration" there is wrong.
    cov = StockKVMethod(bits=4, group_size=64).layer_coverage([_OkCache(), _NoQuantCache()])
    note = cov.note()
    assert "1 of 2" in note
    assert "what mlx-lm's own generate path produces" in note
    assert "not a shipping configuration" not in note


def test_layer_coverage_note_raising_layer_wins_over_skip_only_layer():
    # RED if the branch is any-skip-only instead of any-raising: one sliding-window layer makes
    # mlx-lm crash at conversion, so the mix is unreachable there even with a state-space layer.
    cov = StockKVMethod(bits=4, group_size=64).layer_coverage(
        [_OkCache(), _NoQuantCache(), _RaisingQuantCache()]
    )
    assert "not a shipping configuration" in cov.note()


def test_layer_coverage_skip_reason_for_missing_to_quantized_names_the_right_layer_types():
    # RED while the message cites "sliding-window / MLA": sliding-window DOES define to_quantized
    # (it raises), and MLA stores its latent in a plain KVCache. The no-method case is
    # state-space / chunked caches.
    cov = StockKVMethod(bits=4, group_size=64).layer_coverage([_OkCache(), _NoQuantCache()])
    ((_name, reason),) = cov.skip_reasons
    assert "state-space" in reason
    assert "sliding-window" not in reason
    assert "MLA" not in reason


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
        lambda: json.dumps(
            {"url": "https://github.com/arozanov/turboquant-mlx", "vcs_info": {"commit_id": "abc"}}
        ),
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
    monkeypatch.setattr(
        "mlx_quant_fidelity.probes.kv_methods._turboquant_direct_url",
        lambda: json.dumps({"url": "https://github.com/arozanov/turboquant-mlx", "vcs_info": {}}),
    )
    prov = TurboQuantKVMethod(bits=4).provenance()
    assert prov["commit"] == "unknown"
    assert prov["dist_version"] == "unknown"


def test_provenance_refuses_when_install_record_is_absent(monkeypatch):
    """Bug: provenance() imported the port without the install-source check, so a squatter
    distribution ran its code just to be described."""
    install_fake_port(monkeypatch)
    monkeypatch.setattr("mlx_quant_fidelity.probes.kv_methods._turboquant_installed", lambda: True)
    monkeypatch.setattr("mlx_quant_fidelity.probes.kv_methods._turboquant_direct_url", lambda: None)
    with pytest.raises(MethodUnavailableError, match="squatter"):
        TurboQuantKVMethod(bits=4).provenance()


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
        AffineKVMethod(k_bits=8, v_bits=4, group_size=128).check(head_dim=64, model_type="llama")
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


def test_vonly_provenance_seed_plus_one(monkeypatch):
    """Reds if provenance records the constructor seed instead of the port's seed+1."""
    install_fake_port(monkeypatch)
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


def test_methods_map_includes_affine_and_vonly():
    """0.7.0 task 10 registers both in METHODS (supersedes the prior 'registration is a
    later task' placeholder, which asserted turboquant-vonly was absent).
    """
    assert METHODS["affine"] is AffineKVMethod
    assert METHODS["turboquant-vonly"] is TurboQuantVOnlyKVMethod


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
        lambda: json.dumps(
            {"url": "https://github.com/arozanov/turboquant-mlx", "vcs_info": {"commit_id": "abc"}}
        ),
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
    monkeypatch.setattr(
        "mlx_quant_fidelity.probes.kv_methods._turboquant_direct_url",
        lambda: json.dumps({"url": "https://github.com/arozanov/turboquant-mlx", "vcs_info": {}}),
    )
    prov = TurboQuantVOnlyKVMethod(v_bits=4).provenance()
    assert prov["commit"] == "unknown"
    assert prov["dist_version"] == "unknown"


def test_vonly_provenance_refuses_when_install_record_is_absent(monkeypatch):
    """Bug: provenance() imported the port without the install-source check, so a squatter
    distribution ran its code just to be described."""
    install_fake_port(monkeypatch)
    monkeypatch.setattr("mlx_quant_fidelity.probes.kv_methods._turboquant_installed", lambda: True)
    monkeypatch.setattr("mlx_quant_fidelity.probes.kv_methods._turboquant_direct_url", lambda: None)
    with pytest.raises(MethodUnavailableError, match="squatter"):
        TurboQuantVOnlyKVMethod(v_bits=4).provenance()


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


# --- validation before load, availability before load, package-rooted errors -----------------


@pytest.mark.parametrize("group_size", [16, 0, -64])
def test_stock_method_rejects_unsupported_group_size(group_size):
    """Reds if StockKVMethod accepts a group size mx.quantize does not support (16, 0, -64)."""
    with pytest.raises(CompareConfigError, match="group_size"):
        StockKVMethod(bits=4, group_size=group_size)


@pytest.mark.parametrize("group_size", [16, 0, -64])
def test_affine_rejects_unsupported_group_size(group_size):
    """Reds if AffineKVMethod keeps the old `> 0` check and lets 16 through."""
    with pytest.raises(CompareConfigError, match="group_size"):
        AffineKVMethod(k_bits=8, v_bits=4, group_size=group_size)


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_supported_group_sizes_construct(group_size):
    assert StockKVMethod(bits=4, group_size=group_size).group_size == group_size
    assert AffineKVMethod(k_bits=8, v_bits=4, group_size=group_size).group_size == group_size


def test_stock_validate_before_load_rejects_unsupported_bits():
    """Reds if bits=5 survives to the model load (check() needs a loaded model's head_dim)."""
    with pytest.raises(CacheNotQuantizableError, match="kv_bits=5"):
        StockKVMethod(bits=5, group_size=64).validate_before_load()
    StockKVMethod(bits=4, group_size=64).validate_before_load()  # supported: no raise


def test_method_constructor_errors_are_package_rooted():
    """Reds if any constructor raises a bare ValueError instead of CompareConfigError."""
    bad = [
        lambda: StockKVMethod(bits=4, group_size=16),
        lambda: AffineKVMethod(k_bits=9, v_bits=4),
        lambda: AffineKVMethod(k_bits=8, v_bits=4, group_size=0),
        lambda: TurboQuantKVMethod(bits=5),
        lambda: TurboQuantKVMethod(bits=4, seed=0),
        lambda: TurboQuantVOnlyKVMethod(v_bits=5),
        lambda: TurboQuantVOnlyKVMethod(v_bits=4, seed=0),
    ]
    for build in bad:
        with pytest.raises(CompareConfigError):
            build()


def test_parse_method_spec_errors_start_with_the_spec_not_a_flag():
    """Reds if the parser hard-codes `--configs entry`, wrong for a `--kv-method` caller."""
    for spec in ("affine:9:4", "turboquant:9", "nosuch:1", "stock:4", "affine:8"):
        with pytest.raises(CompareConfigError) as excinfo:
            parse_method_spec(spec)
        assert str(excinfo.value).startswith(f"method spec {spec!r}"), str(excinfo.value)
        assert "--configs" not in str(excinfo.value)


def test_ensure_available_is_a_no_op_for_stock_and_affine():
    StockKVMethod(bits=4, group_size=64).ensure_available()
    AffineKVMethod(k_bits=8, v_bits=4).ensure_available()


@pytest.mark.parametrize("method", [TurboQuantKVMethod(bits=4), TurboQuantVOnlyKVMethod(v_bits=4)])
def test_turboquant_ensure_available_raises_when_port_missing(monkeypatch, method):
    """Reds if availability is only discovered after the model load."""
    monkeypatch.setitem(sys.modules, "turboquant_mlx", None)  # forces ImportError
    with pytest.raises(MethodUnavailableError, match=TURBOQUANT_PINNED_COMMIT):
        method.ensure_available()


@pytest.mark.parametrize("method", [TurboQuantKVMethod(bits=4), TurboQuantVOnlyKVMethod(v_bits=4)])
def test_turboquant_ensure_available_passes_with_the_pinned_fake_port(monkeypatch, method):
    install_fake_port(monkeypatch)
    monkeypatch.setattr(
        "mlx_quant_fidelity.probes.kv_methods._turboquant_direct_url",
        lambda: json.dumps(
            {
                "url": "https://github.com/arozanov/turboquant-mlx",
                "vcs_info": {"commit_id": TURBOQUANT_PINNED_COMMIT},
            }
        ),
    )
    method.ensure_available()


@pytest.mark.parametrize(
    "direct_url",
    [
        None,
        json.dumps({"url": "file:///tmp/x", "dir_info": {}}),
        json.dumps({"url": "https://github.com/someone/turboquant-mlx", "vcs_info": {}}),
    ],
)
def test_squatter_refused_without_import(monkeypatch, direct_url):
    """Reds if a non-arozanov distribution is imported (executing its code) before refusal."""
    monkeypatch.delitem(sys.modules, "turboquant_mlx", raising=False)
    monkeypatch.setattr(
        "mlx_quant_fidelity.probes.kv_methods._turboquant_direct_url", lambda: direct_url
    )
    real_import = __import__

    def guarded(name, *a, **k):
        if name.split(".")[0] == "turboquant_mlx":
            raise AssertionError("turboquant_mlx was imported before the provenance refusal")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", guarded)
    with pytest.raises(MethodUnavailableError, match=TURBOQUANT_PINNED_COMMIT):
        TurboQuantKVMethod(bits=4).ensure_available()
    assert "turboquant_mlx" not in sys.modules


def test_control_lane_protocol_is_structural():
    """Reds if ControlLaneMethod stops matching by `control_method` (stock) or matches others."""
    from mlx_quant_fidelity.probes.kv_methods import ControlLaneMethod

    assert isinstance(StockKVMethod(bits=4, group_size=64), ControlLaneMethod)
    assert not isinstance(AffineKVMethod(k_bits=8, v_bits=4), ControlLaneMethod)
    assert not isinstance(TurboQuantKVMethod(bits=4), ControlLaneMethod)


def test_packed_width_mismatch_lives_in_kv_methods_and_kv_reexports_it():
    # bug caught: the helper drifting back into the method-agnostic probe (import cycle dodge)
    from mlx_quant_fidelity.probes import kv, kv_methods

    assert kv_methods.packed_width_mismatch(128, 6) is True
    assert kv.packed_width_mismatch is kv_methods.packed_width_mismatch


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/arozanov/turboquant-mlx",
        "https://github.com/arozanov/turboquant-mlx.git",
        "https://github.com/arozanov/turboquant-mlx.git/",
        "https://GitHub.com/Arozanov/TurboQuant-MLX.git/",
    ],
)
def test_canonical_source_url_accepts_the_spellings_of_the_pinned_repo(url):
    """Bug: a trailing `.git`, a trailing slash or different case makes the genuine install
    look like a squatter and refuses the method."""
    from mlx_quant_fidelity.probes.kv_methods import _TURBOQUANT_SOURCE_URL, _canonical_source_url

    assert _canonical_source_url(url) == _TURBOQUANT_SOURCE_URL


@pytest.mark.parametrize(
    "record",
    [
        # right repo, but an archive/wheel install: nothing pins the commit
        {"url": "https://github.com/arozanov/turboquant-mlx", "archive_info": {"hash": "x"}},
        # a fork under another owner
        {"url": "https://github.com/other/turboquant-mlx", "vcs_info": {"vcs": "git"}},
        # a name-alike path under the right owner
        {"url": "https://github.com/arozanov/turboquant-mlx-fork", "vcs_info": {"vcs": "git"}},
    ],
)
def test_import_gate_refuses_installs_that_are_not_the_pinned_git_repo(monkeypatch, record):
    """Bug: the gate accepts the right URL without a vcs_info record (an archive install), or a
    near-match URL, and imports code that is not the pinned commit."""
    from mlx_quant_fidelity.probes.kv_methods import _import_turboquant

    monkeypatch.setattr("mlx_quant_fidelity.probes.kv_methods._turboquant_installed", lambda: True)
    monkeypatch.setattr(
        "mlx_quant_fidelity.probes.kv_methods._turboquant_direct_url",
        lambda: json.dumps(record),
    )
    monkeypatch.setitem(sys.modules, "turboquant_mlx", None)
    with pytest.raises(MethodUnavailableError, match="squatter"):
        _import_turboquant()


def test_absent_distribution_says_not_installed_not_squatter(monkeypatch):
    """Bug: a machine with no turboquant-mlx at all is told it has an unrelated squatter."""
    monkeypatch.setattr("mlx_quant_fidelity.probes.kv_methods._turboquant_installed", lambda: False)
    monkeypatch.setattr("mlx_quant_fidelity.probes.kv_methods._turboquant_direct_url", lambda: None)
    with pytest.raises(MethodUnavailableError) as exc:
        TurboQuantKVMethod(bits=4).probe_capability([KVCache()])
    assert "is not installed" in str(exc.value)
    assert TURBOQUANT_PINNED_COMMIT in str(exc.value)
    assert "squatter" not in str(exc.value)


def test_present_distribution_without_git_record_is_named_a_wrong_install(monkeypatch):
    """Bug: a present distribution with no git install record is reported as merely absent,
    hiding that the wrong package is installed."""
    monkeypatch.setattr("mlx_quant_fidelity.probes.kv_methods._turboquant_installed", lambda: True)
    monkeypatch.setattr("mlx_quant_fidelity.probes.kv_methods._turboquant_direct_url", lambda: None)
    with pytest.raises(MethodUnavailableError, match="squatter"):
        TurboQuantKVMethod(bits=4).probe_capability([KVCache()])


# --- the module-origin check (a shadowing turboquant_mlx is refused before import) ---------


def _write_package(root, sentinel):
    """A `turboquant_mlx/` package under ``root`` whose import writes ``sentinel``."""
    pkg = root / "turboquant_mlx"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        f"import pathlib\npathlib.Path({str(sentinel)!r}).write_text('ran')\n"
    )
    return pkg / "__init__.py"


def _pinned_distribution(site, sentinel, *, record=True):
    """A real on-disk `turboquant-mlx` install under ``site``: package files, a RECORD that lists
    them, and a direct_url.json that names the pinned arozanov git commit."""
    import importlib.metadata

    _write_package(site, sentinel)
    info = site / "turboquant_mlx-0.3.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: turboquant-mlx\nVersion: 0.3.0\n")
    if record:
        (info / "RECORD").write_text(
            "turboquant_mlx/__init__.py,,\nturboquant_mlx-0.3.0.dist-info/RECORD,,\n"
        )
    (info / "direct_url.json").write_text(
        json.dumps(
            {
                "url": "https://github.com/arozanov/turboquant-mlx",
                "vcs_info": {"vcs": "git", "commit_id": TURBOQUANT_PINNED_COMMIT},
            }
        )
    )
    return importlib.metadata.PathDistribution(info)


def _serve_distribution(monkeypatch, dist):
    """Route the stdlib metadata lookup for turboquant-mlx to ``dist``; forget any cached port."""
    import importlib.metadata

    real = importlib.metadata.distribution

    def _distribution(distribution_name):
        if distribution_name.lower().replace("_", "-") == "turboquant-mlx":
            return dist
        return real(distribution_name)

    monkeypatch.setattr(importlib.metadata, "distribution", _distribution)
    # setitem then delitem: the undo restores whatever was there (the real port, --hide-port's
    # None, or nothing), so a module imported from tmp_path never leaks into later tests.
    monkeypatch.setitem(sys.modules, "turboquant_mlx", None)
    monkeypatch.delitem(sys.modules, "turboquant_mlx")


def test_shadowing_package_is_refused_without_running_it(monkeypatch, tmp_path):
    """Bug: the gate trusts direct_url.json alone, so a `turboquant_mlx/` directory earlier on
    sys.path than the pinned install is imported, and its code runs."""
    from mlx_quant_fidelity.probes.kv_methods import _import_turboquant

    _serve_distribution(monkeypatch, _pinned_distribution(tmp_path / "site", tmp_path / "real-ran"))
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    shadow_sentinel = tmp_path / "shadow-ran"
    _write_package(tmp_path / "shadow", shadow_sentinel)
    monkeypatch.syspath_prepend(str(tmp_path / "shadow"))

    with pytest.raises(MethodUnavailableError, match="not a file of the installed"):
        _import_turboquant()
    assert not shadow_sentinel.exists()
    assert not (tmp_path / "real-ran").exists()
    assert "turboquant_mlx" not in sys.modules


def test_module_from_the_pinned_distribution_still_imports(monkeypatch, tmp_path):
    """Bug: the origin check compares the module path against the RECORD's relative names (or
    an unresolved path), so the genuine pinned install is refused as a shadow."""
    from mlx_quant_fidelity.probes.kv_methods import _import_turboquant

    sentinel = tmp_path / "real-ran"
    _serve_distribution(monkeypatch, _pinned_distribution(tmp_path / "site", sentinel))
    monkeypatch.syspath_prepend(str(tmp_path / "site"))

    module = _import_turboquant()
    assert sentinel.read_text() == "ran"
    assert module.__file__ == str(tmp_path / "site" / "turboquant_mlx" / "__init__.py")


def test_valid_git_record_with_no_importable_module_says_not_installed(monkeypatch, tmp_path):
    """Bug: a pinned git install record whose module cannot be found (deleted package files, or
    a masked import) falls through to the shadowing check and is reported as a module that
    "resolves to None" shadowing the port, instead of "not installed"."""
    from mlx_quant_fidelity.probes.kv_methods import _import_turboquant

    sentinel = tmp_path / "real-ran"
    _serve_distribution(monkeypatch, _pinned_distribution(tmp_path / "site", sentinel))
    # site is NOT on sys.path, and a None entry makes find_spec report the module unfindable
    # even where the real port is installed (the mechanism --hide-port uses).
    monkeypatch.setitem(sys.modules, "turboquant_mlx", None)

    with pytest.raises(MethodUnavailableError) as excinfo:
        _import_turboquant()
    msg = str(excinfo.value)
    assert msg.startswith("turboquant_mlx is not installed.")
    assert TURBOQUANT_PINNED_COMMIT in msg
    assert "resolves to" not in msg
    assert not sentinel.exists()


def test_distribution_without_a_file_list_is_refused_without_running_it(monkeypatch, tmp_path):
    """Bug: a distribution that lists no files (no RECORD) makes the origin check vacuous, so
    any `turboquant_mlx` on sys.path is imported."""
    from mlx_quant_fidelity.probes.kv_methods import _import_turboquant

    sentinel = tmp_path / "real-ran"
    dist = _pinned_distribution(tmp_path / "site", sentinel, record=False)
    _serve_distribution(monkeypatch, dist)
    monkeypatch.syspath_prepend(str(tmp_path / "site"))

    with pytest.raises(MethodUnavailableError, match="not a file of the installed"):
        _import_turboquant()
    assert not sentinel.exists()


# --- _gate_turboquant_layout: the width the cache really holds ------------------------------

_V2_LAYOUT = KVLayout(k_dim=192, v_dim=128, modelled=True)
_UNMODELLED = KVLayout(k_dim=None, v_dim=None, modelled=False)
_NO_MLA = KVLayout(k_dim=None, v_dim=None, modelled=True)


def test_layout_gate_refuses_turboquant_on_the_key_width():
    """Bug: uniform TurboQuant reaches the kernels with K 192, where it returns wrong values."""
    with pytest.raises(CacheNotQuantizableError) as excinfo:
        _gate_turboquant_layout(TurboQuantKVMethod(bits=4), _V2_LAYOUT, model_type="deepseek_v2")
    msg = str(excinfo.value)
    assert "key width=192" in msg
    assert "deepseek_v2" in msg
    assert "port's kernels" in msg


@pytest.mark.parametrize(
    ("layout", "named"),
    [
        (KVLayout(k_dim=128, v_dim=96, modelled=True), "96"),  # V is not a power of two
        (KVLayout(k_dim=128, v_dim=512, modelled=True), "512"),  # V overruns the arrays
        (KVLayout(k_dim=512, v_dim=128, modelled=True), "512"),  # K overruns the arrays
    ],
)
def test_layout_gate_turboquant_checks_both_widths(layout, named):
    """Bug: only one side is gated while both go through the kernels."""
    with pytest.raises(CacheNotQuantizableError, match=named):
        _gate_turboquant_layout(TurboQuantKVMethod(bits=4), layout, model_type="x")


def test_layout_gate_allows_turboquant_when_both_widths_are_fit():
    """Bug: every expanded layout is refused, not only one whose width the kernels cannot take."""
    layout = KVLayout(k_dim=128, v_dim=64, modelled=True)
    _gate_turboquant_layout(TurboQuantKVMethod(bits=4), layout, model_type="deepseek_v2")


def test_layout_gate_allows_vonly_on_value_width():
    """Bug: K is gated on V-only, where it stays in plain fp16 and never meets a kernel."""
    vonly = TurboQuantVOnlyKVMethod(v_bits=4)
    _gate_turboquant_layout(vonly, _V2_LAYOUT, model_type="deepseek_v2")
    k_too_wide = KVLayout(k_dim=512, v_dim=128, modelled=True)
    _gate_turboquant_layout(vonly, k_too_wide, model_type="deepseek_v2")


@pytest.mark.parametrize("v_dim", [96, 512])
def test_layout_gate_refuses_vonly_on_unfit_value_width(v_dim):
    """Bug: V-only trusts hidden//heads, which says nothing about the width V is cached at."""
    layout = KVLayout(k_dim=192, v_dim=v_dim, modelled=True)
    with pytest.raises(CacheNotQuantizableError, match=f"value width={v_dim}"):
        _gate_turboquant_layout(TurboQuantVOnlyKVMethod(v_bits=4), layout, model_type="deepseek_v2")


@pytest.mark.parametrize(
    "method", [TurboQuantKVMethod(bits=4), TurboQuantVOnlyKVMethod(v_bits=4)], ids=["uniform", "v"]
)
def test_layout_gate_refuses_both_methods_on_unmodelled_layout(method):
    """Bug: a cache whose widths nobody can state reaches the kernels."""
    with pytest.raises(CacheNotQuantizableError) as excinfo:
        _gate_turboquant_layout(method, _UNMODELLED, model_type="deepseek_v3")
    msg = str(excinfo.value)
    assert "deepseek_v3" in msg
    assert "port's kernels" in msg


@pytest.mark.parametrize(
    "method", [TurboQuantKVMethod(bits=4), TurboQuantVOnlyKVMethod(v_bits=4)], ids=["uniform", "v"]
)
def test_layout_gate_leaves_models_without_mla_fields_to_the_head_dim_gate(method):
    """Bug: a K == V model is refused here, or None widths fail the power-of-two test."""
    _gate_turboquant_layout(method, _NO_MLA, model_type="llama")


@pytest.mark.parametrize(
    "method",
    [
        StockKVMethod(bits=4, group_size=64),
        AffineKVMethod(k_bits=8, v_bits=4),
        FakeKVMethod(name="turboquant"),  # a third-party method is judged by class, not by name
    ],
    ids=["stock", "affine", "third-party"],
)
@pytest.mark.parametrize("layout", [_V2_LAYOUT, _UNMODELLED, _NO_MLA], ids=["v2", "latent", "none"])
def test_layout_gate_is_a_no_op_for_other_methods(method, layout):
    """Bug: the helper gates by anything wider than the two TurboQuant classes."""
    _gate_turboquant_layout(method, layout, model_type="deepseek_v2")


def test_head_dim_gate_message_is_unchanged():
    """Bug: the K == V refusal text drifts while the gate gains a width label."""
    with pytest.raises(CacheNotQuantizableError) as excinfo:
        TurboQuantKVMethod(bits=4).check(head_dim=96, model_type="phi3")
    assert str(excinfo.value) == (
        "TurboQuant-MLX kernels require a known power-of-two head_dim <= 256; model 'phi3' has "
        "head_dim=96. This is a limitation of the port's kernels, not of the model."
    )


# --- K and V widths counted separately (v_dim) -----------------------------------------------

# DeepSeek-V2-Lite: 27 layers, 16 KV heads, K = 128 + 64 = 192, V = 128 (config.json at a33a8ad).
_V2_LITE_GEO = {"n_layers": 27, "n_kv_heads": 16, "head_dim": 192, "v_dim": 128}


def test_stock_bytes_per_token_forwards_v_dim():
    """Bug: the stock method drops v_dim and costs V at K's width (93312 instead of 77760)."""
    assert StockKVMethod(bits=4, group_size=64).bytes_per_token(**_V2_LITE_GEO) == 77760


def test_affine_bytes_per_token_counts_each_side_at_its_width():
    """Bug: V is costed at K's width. 432 * (192 * 1.0625 + 128 * 0.5625) = 432 * 276 = 119232."""
    assert AffineKVMethod(k_bits=8, v_bits=4).bytes_per_token(**_V2_LITE_GEO) == 119232


def test_affine_bytes_per_token_pairs_k_bits_with_the_key_width():
    """Bug: k_bits is applied to V's width and v_bits to K's (the sides are swapped).
    432 * (192 * 0.5625 + 128 * 1.0625) = 432 * 244 = 105408; swapped it would be 119232."""
    assert AffineKVMethod(k_bits=4, v_bits=8).bytes_per_token(**_V2_LITE_GEO) == 105408


def test_turboquant_bytes_per_token_packs_each_side_at_its_width():
    """Bug: V is packed at K's width. K 128: 16 words * 4 + 4 = 68; V 64: 8 * 4 + 4 = 36."""
    assert (
        TurboQuantKVMethod(bits=4).bytes_per_token(n_layers=1, n_kv_heads=1, head_dim=128, v_dim=64)
        == 104
    )


@pytest.mark.parametrize("v_dim", [96, 512])
def test_turboquant_bytes_per_token_gates_the_value_width(v_dim):
    """Bug: only K reaches the kernel gate, so a V of 96 (not a power of two) or 512 (overruns
    the threadgroup arrays) gets a cost."""
    with pytest.raises(CacheNotQuantizableError, match=f"value width={v_dim}"):
        turboquant_bytes_per_token(n_layers=1, n_kv_heads=1, head_dim=128, v_dim=v_dim, bits=4)
    with pytest.raises(CacheNotQuantizableError, match=f"value width={v_dim}"):
        TurboQuantKVMethod(bits=4).bytes_per_token(
            n_layers=1, n_kv_heads=1, head_dim=128, v_dim=v_dim
        )


def test_vonly_bytes_per_token_counts_fp16_k_and_v_plus_packed_v():
    """Bug: the fp16 duplicate is counted at 2 * K, or V is packed at K's width.
    2 B * (192 + 128) = 640 fp16, plus V 128 at 4 bits: 16 * 4 + 4 = 68; total 708."""
    assert (
        TurboQuantVOnlyKVMethod(v_bits=4).bytes_per_token(
            n_layers=1, n_kv_heads=1, head_dim=192, v_dim=128
        )
        == 708
    )


def test_affine_working_set_counts_k_plus_v():
    """Bug: the dequantized transients are sized at 2 * K. 432 * 1 * (192 + 128) * 2 = 276480."""
    ws = AffineKVMethod(k_bits=8, v_bits=4).working_set_bytes(
        window=1, **_V2_LITE_GEO, dtype_bytes=2
    )
    assert ws == 276480


def test_turboquant_working_set_counts_k_plus_v():
    """Bug: the four port buffers are sized at K's width. 2 * 512 * (128 + 64) * 2 = 393216."""
    ws = TurboQuantKVMethod(bits=4).working_set_bytes(
        window=512, n_layers=1, n_kv_heads=1, head_dim=128, v_dim=64, dtype_bytes=2
    )
    assert ws == 393216


def test_vonly_working_set_counts_only_v():
    """Bug: K leaks into the V-only buffers (K stays plain fp16). 2 * 512 * 64 * 2 = 131072."""
    ws = TurboQuantVOnlyKVMethod(v_bits=4).working_set_bytes(
        window=512, n_layers=1, n_kv_heads=1, head_dim=128, v_dim=64, dtype_bytes=2
    )
    assert ws == 131072


def test_stock_working_set_ignores_v_dim():
    """Bug: stock gains a working set on a K != V model (its attention reads the packed cache)."""
    ws = StockKVMethod(bits=4, group_size=64).working_set_bytes(
        window=512, **_V2_LITE_GEO, dtype_bytes=2
    )
    assert ws == 0


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        # Llama-3.2-1B hand values from the committed sample (16 layers, 8 KV heads, 64 dims).
        (StockKVMethod(bits=4, group_size=64), 9216),
        (AffineKVMethod(k_bits=8, v_bits=4), 13312),
        (TurboQuantKVMethod(bits=4), 9216),
        (TurboQuantVOnlyKVMethod(v_bits=4), 37376),
    ],
    ids=["stock", "affine", "turboquant", "vonly"],
)
def test_explicit_equal_v_dim_keeps_the_symmetric_cost(method, expected):
    """Bug: an explicit v_dim equal to head_dim takes a different arithmetic path than None."""
    geo = {"n_layers": 16, "n_kv_heads": 8, "head_dim": 64}
    assert method.bytes_per_token(**geo, v_dim=64) == expected
    assert method.bytes_per_token(**geo) == expected


# --- check(..., v_dim=): every gate looks at both widths ------------------------------------


def test_stock_check_refuses_a_group_that_does_not_divide_the_key_width():
    """Bug: the divisibility gate tests V only; group 128 divides V 128 but not K 192."""
    with pytest.raises(CacheNotQuantizableError, match="192"):
        StockKVMethod(bits=4, group_size=128).check(
            head_dim=192, v_dim=128, model_type="deepseek_v2"
        )


def test_stock_check_refuses_a_group_that_does_not_divide_the_value_width():
    """Bug: the divisibility gate tests K only; group 128 divides K 128 but not V 192."""
    with pytest.raises(CacheNotQuantizableError, match="192"):
        StockKVMethod(bits=4, group_size=128).check(head_dim=128, v_dim=192, model_type="x")


def test_stock_check_refuses_a_packed_width_mismatch_on_the_key_width():
    """Bug: the packed-width gate tests V only. K 192 at 3 bits: 192 // 10 = 19 vs 18."""
    with pytest.raises(CacheNotQuantizableError, match="packed-width"):
        StockKVMethod(bits=3, group_size=64).check(
            head_dim=192, v_dim=128, model_type="deepseek_v2"
        )


def test_stock_check_refuses_a_packed_width_mismatch_on_the_value_width():
    """Bug: the packed-width gate tests K only. K 64 at 6 bits is clean; V 128: 25 vs 24."""
    with pytest.raises(CacheNotQuantizableError, match="packed-width"):
        StockKVMethod(bits=6, group_size=64).check(head_dim=64, v_dim=128, model_type="x")


def test_stock_check_accepts_deepseek_v2_widths_at_4_64():
    """Bug: any split layout is refused, not only one a width cannot take."""
    assert (
        StockKVMethod(bits=4, group_size=64).check(
            head_dim=192, v_dim=128, model_type="deepseek_v2"
        )
        == []
    )


def test_affine_check_refuses_a_group_that_does_not_divide_the_value_width():
    """Bug: the affine gate tests K only; group 64 divides K 128 but not V 96."""
    with pytest.raises(CacheNotQuantizableError, match="96"):
        AffineKVMethod(k_bits=8, v_bits=4).check(head_dim=128, v_dim=96, model_type="x")


def test_affine_check_accepts_deepseek_v2_widths():
    """Bug: any split layout is refused by the affine gate."""
    assert (
        AffineKVMethod(k_bits=8, v_bits=4).check(head_dim=192, v_dim=128, model_type="deepseek_v2")
        == []
    )


def test_turboquant_check_refuses_the_key_width():
    """Bug: uniform TurboQuant gates V only, so K 192 (not a power of two) reaches the kernels."""
    with pytest.raises(CacheNotQuantizableError, match="192"):
        TurboQuantKVMethod(bits=4).check(head_dim=192, v_dim=128, model_type="deepseek_v2")


def test_turboquant_check_refuses_the_value_width():
    """Bug: uniform TurboQuant gates K only, so V 96 reaches the kernels."""
    with pytest.raises(CacheNotQuantizableError, match="value width=96"):
        TurboQuantKVMethod(bits=4).check(head_dim=128, v_dim=96, model_type="x")


def test_turboquant_check_accepts_two_fit_widths():
    """Bug: any split layout is refused by the uniform gate."""
    assert TurboQuantKVMethod(bits=4).check(head_dim=128, v_dim=64, model_type="x") == []


def test_vonly_check_does_not_gate_the_key_width():
    """Bug: V-only gates K 192, which stays in plain fp16 and never meets a kernel."""
    assert (
        TurboQuantVOnlyKVMethod(v_bits=4).check(head_dim=192, v_dim=128, model_type="deepseek_v2")
        == []
    )


def test_vonly_check_gates_the_value_width():
    """Bug: V-only gates head_dim (K) instead of the value width it really quantizes."""
    with pytest.raises(CacheNotQuantizableError, match="value width=96"):
        TurboQuantVOnlyKVMethod(v_bits=4).check(head_dim=128, v_dim=96, model_type="x")


def test_explicit_equal_v_dim_keeps_the_refusal_text():
    """Bug: an explicit v_dim equal to head_dim switches the K == V refusal to the split wording."""
    with pytest.raises(CacheNotQuantizableError) as excinfo:
        TurboQuantKVMethod(bits=4).check(head_dim=96, v_dim=96, model_type="phi3")
    assert str(excinfo.value) == (
        "TurboQuant-MLX kernels require a known power-of-two head_dim <= 256; model 'phi3' has "
        "head_dim=96. This is a limitation of the port's kernels, not of the model."
    )
