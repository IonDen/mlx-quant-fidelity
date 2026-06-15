"""Offline fake-forward tests for the weight-quant probe (no real model load)."""

from mlx_quant_fidelity.probes.weights import QuantMeta, extract_quant_meta


def test_extract_native_quantization():
    meta = extract_quant_meta({"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}})
    assert meta == QuantMeta(bits=4, group_size=64, mode="affine", per_layer=False)


def test_extract_per_layer_native_quantization():
    cfg = {"quantization": {"group_size": 64, "bits": 4, "model.layers.0.mlp": {"bits": 8}}}
    meta = extract_quant_meta(cfg)
    assert meta is not None
    assert meta.bits == 4
    assert meta.per_layer is True


def test_extract_hf_quantization_config():
    meta = extract_quant_meta({"quantization_config": {"quant_method": "awq", "bits": 4}})
    assert meta is not None
    assert meta.bits == 4
    assert meta.mode == "awq"


def test_extract_nested_text_config():
    cfg = {"text_config": {"quantization": {"group_size": 32, "bits": 8, "mode": "affine"}}}
    meta = extract_quant_meta(cfg)
    assert meta is not None
    assert meta.bits == 8
    assert meta.group_size == 32


def test_extract_returns_none_when_not_quantized():
    assert extract_quant_meta({"model_type": "llama", "vocab_size": 128256}) is None


def test_extract_rejects_bool_bits():
    # bool is a subclass of int but is not a valid bits value -> treated as unknown
    meta = extract_quant_meta({"quantization": {"bits": True, "group_size": 64}})
    assert meta is not None
    assert meta.bits is None


def test_extract_empty_native_block_is_quantized_with_unknown_bits():
    # {"quantization": {}} declares quantization (key present) but carries no params:
    # returns a non-None QuantMeta with bits=None. The Task 8 gate treats key-present as
    # "quantized"; an actually-unquantized model would trip the exact-zero guard downstream.
    meta = extract_quant_meta({"quantization": {}})
    assert meta is not None
    assert meta.bits is None
    assert meta.per_layer is False
