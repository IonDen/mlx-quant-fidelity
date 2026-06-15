"""Offline fake-forward tests for the weight-quant probe (no real model load)."""

import math

import mlx.core as mx

from mlx_quant_fidelity.probes.weights import QuantMeta, _score_weight_chunk, extract_quant_meta


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


class _FakeWeightModel:
    """No-cache forward returning a fixed one-hot-peaked logit tensor (peak index `peak`)."""

    def __init__(self, peak: int):
        self._peak = peak

    def __call__(self, inp):  # inp [1, L]; returns [1, L, 3]
        length = inp.shape[1]
        out = mx.zeros((1, length, 3))
        out[:, :, self._peak] = 5.0
        return out


def test_score_weight_chunk_detects_divergence():
    ids = mx.array([0, 1, 2])  # vocab=3; targets [1, 2]
    kl, flips, ref_nll, quant_nll = _score_weight_chunk(
        _FakeWeightModel(0), _FakeWeightModel(1), ids
    )
    mx.eval(kl, flips, ref_nll, quant_nll)
    assert kl.shape == (2,)  # L-1 teacher-forced positions
    # ref peak [5,0,0] vs quant peak [0,5,0] -> per-position KL(ref||quant) ~ 4.90 nats (hand-computed)
    assert math.isclose(float(kl.mean()), 4.90, abs_tol=0.02)
    assert int(flips.astype(mx.int32).sum()) == 2  # both positions flip
    # A bypass that called ref_model twice would give KLD 0 -> this assertion goes red.


def test_score_weight_chunk_identical_is_zero():
    ids = mx.array([0, 1, 2])
    kl, _flips, _r, _q = _score_weight_chunk(_FakeWeightModel(0), _FakeWeightModel(0), ids)
    mx.eval(kl)
    assert float(kl.mean()) == 0.0
