"""Offline fake-forward tests for the weight-quant probe (no real model load)."""

import math

import mlx.core as mx
import pytest

from mlx_quant_fidelity.corpora.provenance import Corpus, CorpusProvenance
from mlx_quant_fidelity.errors import (
    CorpusError,
    ExactZeroError,
    InsufficientMemoryError,
    ModelMismatchError,
)
from mlx_quant_fidelity.probes.weights import (
    TOKENIZER_ASSUMPTION_WARNING,
    QuantMeta,
    _gate_configs,
    _preflight_memory,
    _resolve_weight_bytes,
    _score_weight_chunk,
    _tokenizer_warnings,
    extract_quant_meta,
    measure_weight_fidelity,
)


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
    # returns a non-None QuantMeta with bits=None. The gate rejects bits=None with
    # ModelMismatchError; see test_gate_rejects_unknown_bits.
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


_Q_NATIVE = {
    "model_type": "llama",
    "vocab_size": 128,
    "quantization": {"bits": 4, "group_size": 64},
}
_REF_FP = {"model_type": "llama", "vocab_size": 128}


def test_gate_passes_and_records_reference_bits_when_reference_quantized():
    ref_q8 = {
        "model_type": "llama",
        "vocab_size": 128,
        "quantization": {"bits": 8, "group_size": 64},
    }
    meta, reference_bits = _gate_configs(quant_config=_Q_NATIVE, reference_config=ref_q8)
    assert meta.bits == 4
    assert reference_bits == 8


def test_gate_passes_for_full_precision_reference():
    meta, reference_bits = _gate_configs(quant_config=_Q_NATIVE, reference_config=_REF_FP)
    assert meta.bits == 4
    assert reference_bits is None


def test_gate_rejects_unquantized_quant_repo():
    with pytest.raises(ModelMismatchError, match="no quantization"):
        _gate_configs(quant_config=_REF_FP, reference_config=_REF_FP)


def test_gate_rejects_model_type_mismatch():
    bad = {"model_type": "qwen2", "vocab_size": 128}
    with pytest.raises(ModelMismatchError, match="model_type"):
        _gate_configs(quant_config=_Q_NATIVE, reference_config=bad)


def test_gate_rejects_vocab_size_mismatch():
    bad = {"model_type": "llama", "vocab_size": 99}
    with pytest.raises(ModelMismatchError, match="vocab_size"):
        _gate_configs(quant_config=_Q_NATIVE, reference_config=bad)


class _Tok:
    def __init__(self, bos, eos):
        self.bos_token_id = bos
        self.eos_token_id = eos


def test_tokenizer_warnings_standing_plus_mismatch():
    same = _tokenizer_warnings(_Tok(1, 2), _Tok(1, 2))
    assert same == [TOKENIZER_ASSUMPTION_WARNING]
    # eos mismatch
    diff_eos = _tokenizer_warnings(_Tok(1, 2), _Tok(1, 9))
    assert len(diff_eos) == 2
    assert any("eos" in w for w in diff_eos)
    # bos mismatch (independent check — a misspelled attr in the comprehension would pass eos but fail here)
    diff_bos = _tokenizer_warnings(_Tok(9, 2), _Tok(1, 2))
    assert len(diff_bos) == 2
    assert any("bos" in w for w in diff_bos)
    # both mismatch
    diff_both = _tokenizer_warnings(_Tok(9, 9), _Tok(1, 2))
    assert len(diff_both) == 3


def test_gate_rejects_both_none_model_type():
    """Two configs that both omit model_type must not silently pass as 'matched'."""
    quant_no_type = {"vocab_size": 128, "quantization": {"bits": 4, "group_size": 64}}
    ref_no_type = {"vocab_size": 128}
    with pytest.raises(ModelMismatchError, match="model_type"):
        _gate_configs(quant_config=quant_no_type, reference_config=ref_no_type)


def test_gate_rejects_both_none_vocab_size():
    """Two configs that both omit vocab_size must not silently pass as 'matched'."""
    quant_no_vocab = {"model_type": "llama", "quantization": {"bits": 4, "group_size": 64}}
    ref_no_vocab = {"model_type": "llama"}
    with pytest.raises(ModelMismatchError, match="vocab_size"):
        _gate_configs(quant_config=quant_no_vocab, reference_config=ref_no_vocab)


def test_gate_rejects_unknown_bits():
    """A quantization block with no 'bits' key must be rejected at the gate, not passed through."""
    quant_no_bits = {"model_type": "llama", "vocab_size": 128, "quantization": {}}
    with pytest.raises(ModelMismatchError, match="bits"):
        _gate_configs(quant_config=quant_no_bits, reference_config=_REF_FP)


def test_resolve_weight_bytes_local_dir(tmp_path):
    # standard HF sharded naming, summed across shards
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x" * 2048)
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"x" * 1024)
    (tmp_path / "tokenizer.json").write_bytes(b"y" * 10)  # not summed (only model*.safetensors)
    assert _resolve_weight_bytes(str(tmp_path), None) == 3072


def test_resolve_weight_bytes_none_when_no_weights(tmp_path):
    assert _resolve_weight_bytes(str(tmp_path), None) is None


def test_preflight_raises_when_over_budget(monkeypatch):
    from mlx_quant_fidelity.probes import weights as w

    monkeypatch.setattr(
        w.mx, "device_info", lambda: {"max_recommended_working_set_size": 10 * 1024**3}
    )
    with pytest.raises(InsufficientMemoryError, match="exceed"):
        _preflight_memory(quant_bytes=6 * 1024**3, reference_bytes=6 * 1024**3)


def test_preflight_silent_when_within_budget(monkeypatch):
    from mlx_quant_fidelity.probes import weights as w

    monkeypatch.setattr(
        w.mx, "device_info", lambda: {"max_recommended_working_set_size": 32 * 1024**3}
    )
    _preflight_memory(quant_bytes=4 * 1024**3, reference_bytes=14 * 1024**3)  # no raise


def test_preflight_skips_when_size_unknown(monkeypatch):
    from mlx_quant_fidelity.probes import weights as w

    monkeypatch.setattr(w.mx, "device_info", lambda: {"max_recommended_working_set_size": 1})
    _preflight_memory(quant_bytes=None, reference_bytes=14 * 1024**3)  # no raise (can't pre-flight)


# ---------------------------------------------------------------------------
# Orchestration tests (measure_weight_fidelity)
# ---------------------------------------------------------------------------


def _corpus(n_chunks: int, chunk_len: int = 4) -> Corpus:
    chunks = tuple(mx.arange(chunk_len) for _ in range(n_chunks))
    prov = CorpusProvenance(
        "x", "test", "org/m-bf16", chunk_len, chunk_len, "none", "drop", "raw", chunk_len * n_chunks
    )
    return Corpus(chunks=chunks, provenance=prov)


def _patch_loads(monkeypatch, ref_peak, quant_peak, *, calls, ref_quantized=False):
    """Patch weights.* so two fake models load with known configs; record call order."""
    import mlx_lm

    from mlx_quant_fidelity.probes import weights as w

    monkeypatch.setattr(w, "install_memory_caps", lambda: calls.append("caps") or (0, 0))
    monkeypatch.setattr(w, "_resolve_weight_bytes", lambda *a, **k: None)  # skip pre-flight
    ref_cfg = {"model_type": "llama", "vocab_size": 3}
    if ref_quantized:
        ref_cfg = {**ref_cfg, "quantization": {"bits": 8, "group_size": 64}}
    cfgs = {
        "ref": ref_cfg,
        "quant": {
            "model_type": "llama",
            "vocab_size": 3,
            "quantization": {"bits": 4, "group_size": 64},
        },
    }
    toks = type("T", (), {"bos_token_id": 1, "eos_token_id": 2})()

    def fake_load(repo, **kw):
        if repo == "ref":
            calls.append("load_ref")
            return _FakeWeightModel(ref_peak), toks, cfgs["ref"]
        calls.append("load_quant")
        return _FakeWeightModel(quant_peak), toks, cfgs["quant"]

    monkeypatch.setattr(mlx_lm, "load", fake_load)


def test_measure_weight_caps_before_both_loads(monkeypatch):
    calls: list[str] = []
    _patch_loads(monkeypatch, ref_peak=0, quant_peak=1, calls=calls)
    report = measure_weight_fidelity("quant", "ref", corpus=_corpus(2))
    assert calls == ["caps", "load_ref", "load_quant"]
    assert report.n_chunks == 2
    assert report.n_positions == 6  # 2 chunks x (4-1) positions, pooled
    assert report.kl.mean > 0.1
    assert report.reference_bits is None


def test_measure_weight_records_quantized_reference_bits(monkeypatch):
    calls: list[str] = []
    _patch_loads(monkeypatch, ref_peak=0, quant_peak=1, calls=calls, ref_quantized=True)
    report = measure_weight_fidelity("quant", "ref", corpus=_corpus(1))
    assert report.reference_bits == 8
    assert any("not full precision" in w for w in report.warnings)


def test_measure_weight_exact_zero_when_identical(monkeypatch):
    calls: list[str] = []
    _patch_loads(monkeypatch, ref_peak=0, quant_peak=0, calls=calls)  # identical models
    with pytest.raises(ExactZeroError, match="identical"):
        measure_weight_fidelity("quant", "ref", corpus=_corpus(1))


def test_measure_weight_rejects_bad_max_chunks():
    with pytest.raises(CorpusError, match="max_chunks"):
        measure_weight_fidelity("quant", "ref", max_chunks=0)


def test_measure_weight_rejects_empty_corpus():
    empty = Corpus(
        chunks=(), provenance=CorpusProvenance("x", "t", "t", 4, 4, "none", "drop", "raw", 0)
    )
    with pytest.raises(CorpusError, match="no chunks"):
        measure_weight_fidelity("quant", "ref", corpus=empty)
