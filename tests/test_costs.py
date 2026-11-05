import pytest

from mlx_quant_fidelity.costs import KVLayout, kv_bytes_per_token, kv_layout


def test_kv_bytes_per_token_hand_value():
    # elements/token = 2*K,V * 2 layers * 4 kv-heads * 64 head_dim = 1024
    # bytes/element  = 4/8 + 4/64 = 0.5 + 0.0625 = 0.5625
    # total          = 1024 * 0.5625 = 576
    assert kv_bytes_per_token(n_layers=2, n_kv_heads=4, head_dim=64, bits=4, group_size=64) == 576


def test_kv_bytes_smaller_group_costs_more():
    big = kv_bytes_per_token(n_layers=2, n_kv_heads=4, head_dim=64, bits=4, group_size=64)
    small = kv_bytes_per_token(
        n_layers=2, n_kv_heads=4, head_dim=64, bits=4, group_size=32
    )  # group=32: 640
    assert small > big  # the group scale/bias overhead must be visible to the Pareto


def test_kv_bytes_higher_bits_costs_more():
    four = kv_bytes_per_token(n_layers=2, n_kv_heads=4, head_dim=64, bits=4, group_size=64)
    eight = kv_bytes_per_token(n_layers=2, n_kv_heads=4, head_dim=64, bits=8, group_size=64)
    assert eight > four


def test_kv_bytes_scales_with_model_shape():
    # linear in n_layers, n_kv_heads, head_dim — a hardcoded factor fails one of these.
    base = kv_bytes_per_token(n_layers=2, n_kv_heads=4, head_dim=64, bits=4, group_size=64)  # 576
    assert (
        kv_bytes_per_token(n_layers=4, n_kv_heads=4, head_dim=64, bits=4, group_size=64) == 2 * base
    )
    assert (
        kv_bytes_per_token(n_layers=2, n_kv_heads=8, head_dim=64, bits=4, group_size=64) == 2 * base
    )
    assert (
        kv_bytes_per_token(n_layers=2, n_kv_heads=4, head_dim=128, bits=4, group_size=64)
        == 2 * base
    )


def test_kv_bytes_rejects_nonpositive_group_size():
    with pytest.raises(ValueError, match="group_size must be positive"):
        kv_bytes_per_token(n_layers=2, n_kv_heads=4, head_dim=64, bits=4, group_size=0)


# --- kv_layout: which widths a model's KV cache stores ---------------------------------------

_V2_FIELDS = {"qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128}
_LATENT_FIELDS = {**_V2_FIELDS, "kv_lora_rank": 512}


@pytest.mark.parametrize("model_type", ["deepseek_v2", "youtu_llm"])
def test_kv_layout_expanded_cache_adds_rope_to_k(model_type):
    """Bug: K drops the rope part, or V is read from the wrong field. Hand values from
    deepseek_v2.py:224-232 and youtu_llm.py:117-125: K is concat(k_nope, k_pe), V is values."""
    fields = {"qk_nope_head_dim": 96, "qk_rope_head_dim": 32, "v_head_dim": 64}
    assert kv_layout(model_type, fields.get) == KVLayout(k_dim=128, v_dim=64, modelled=True)
    assert kv_layout(model_type, _V2_FIELDS.get) == KVLayout(k_dim=192, v_dim=128, modelled=True)


def test_kv_layout_no_mla_fields_keeps_the_old_rule():
    """Bug: a K == V model loses its old rule and is refused or given invented widths."""
    assert kv_layout("llama", {}.get) == KVLayout(k_dim=None, v_dim=None, modelled=True)


@pytest.mark.parametrize(
    "model_type",
    [
        "deepseek_v3",
        "deepseek_v32",
        "longcat_flash",
        "longcat_flash_ngram",
        "glm_moe_dsa",
        "glm4_moe_lite",
        "kimi_vl",
        "kimi_k25",
        "kimi_linear",
        "kimi_k2",  # mlx-lm's MODEL_REMAPPING sends these two to deepseek_v3
        "joyai_llm_flash",
        "minicpm3",
        "mimo_v2_flash",
        "brand_new_mla_model",
    ],
)
def test_kv_layout_same_fields_elsewhere_are_not_modelled(model_type):
    """Bug: the rule keys on the fields alone, so a latent cache gets the expanded widths."""
    assert kv_layout(model_type, _LATENT_FIELDS.get) == KVLayout(
        k_dim=None, v_dim=None, modelled=False
    )


@pytest.mark.parametrize(
    "field", ["qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "kv_lora_rank"]
)
def test_kv_layout_any_single_mla_field_marks_the_layout(field):
    """Bug: one field drops out of the detection. Each is the only one some mlx-lm model is sure
    to carry: kimi_linear's qk fields are optional, mimo_v2_flash has only v_head_dim."""
    assert kv_layout("mystery", {field: 64}.get).modelled is False


@pytest.mark.parametrize("missing", ["qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"])
def test_kv_layout_deepseek_v2_missing_a_field_is_not_modelled(missing):
    """Bug: a partial config guesses a width from the fields it has."""
    fields = {k: v for k, v in _V2_FIELDS.items() if k != missing}
    assert kv_layout("deepseek_v2", fields.get).modelled is False


def test_kv_layout_deepseek_v2_with_no_fields_is_not_modelled():
    """Bug: an expanded model with no widths falls through to the K == V rule."""
    assert kv_layout("deepseek_v2", {}.get).modelled is False


@pytest.mark.parametrize("bad", [0, -64, True, "64", 64.0, None])
def test_kv_layout_field_must_be_a_positive_int(bad):
    """Bug: a bool, a string, zero or a negative is taken as a width (True would add 1 to K)."""
    assert kv_layout("deepseek_v2", {**_V2_FIELDS, "qk_rope_head_dim": bad}.get).modelled is False
    assert kv_layout("deepseek_v2", {**_V2_FIELDS, "v_head_dim": bad}.get).modelled is False


@pytest.mark.parametrize("value", [128.0, "128", True])
def test_kv_layout_any_non_none_mla_value_marks_another_type(value):
    """Bug: only a positive int counts as an MLA field, so a float or string width on a model
    the registry does not model falls through to the hidden//heads gate."""
    assert kv_layout("llama", {"v_head_dim": value}.get) == KVLayout(
        k_dim=None, v_dim=None, modelled=False
    )


@pytest.mark.parametrize("model_type", [None, 7, ["deepseek_v2"]])
def test_kv_layout_non_string_model_type_is_not_the_expanded_rule(model_type):
    """Bug: an unhashable model_type crashes the lookup, or a non-string reaches the rule."""
    assert kv_layout(model_type, _V2_FIELDS.get).modelled is False
    assert kv_layout(model_type, {}.get).modelled is True
