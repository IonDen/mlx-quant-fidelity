"""Offline fake-forward tests for the KV-quant probe (no real model load)."""

import math

import mlx.core as mx
import pytest

from mlx_quant_fidelity.errors import (
    CacheNotQuantizableError,
    CorpusError,
    ExactZeroError,
    QuantFidelityError,
)
from mlx_quant_fidelity.probes.kv import (
    _aggregate_chunks,
    _cache_is_quantizable,
    _check_exact_zero,
    _head_dim_gate,
    _kv_head_dim,
    _score_chunk,
    measure_kv_fidelity,
)

# ---------------------------------------------------------------------------
# Task 4.1 — pure helpers: capability gate, exact-zero guard, aggregation
# ---------------------------------------------------------------------------


class _NoQuantCache:
    pass  # no to_quantized attribute


class _RaisingQuantCache:
    def to_quantized(self, **_kwargs):
        raise NotImplementedError


class _OkCache:
    def to_quantized(self, **_kwargs):
        return self


def test_exact_zero_guard_raises_on_identical():
    # (a) identical => guard RAISES, never a silent pass
    with pytest.raises(ExactZeroError):
        _check_exact_zero(kl_mean=0.0, flip_rate=0.0, context="quantization did not engage")


def test_exact_zero_guard_silent_when_nonzero():
    _check_exact_zero(kl_mean=0.3, flip_rate=0.1, context="quantization did not engage")  # no raise


def test_capability_gate_flags_unsupported():
    # (b) NYI cache (has the attr but raises) => flagged unsupported, naming the type, not a crash
    with pytest.raises(CacheNotQuantizableError, match="_RaisingQuantCache"):
        _cache_is_quantizable([_RaisingQuantCache()], group_size=64, bits=4)
    with pytest.raises(CacheNotQuantizableError, match="_NoQuantCache"):
        _cache_is_quantizable([_NoQuantCache()], group_size=64, bits=4)
    assert _cache_is_quantizable([_OkCache()], group_size=64, bits=4) is True


def test_two_chunk_aggregation_combines_both():
    # (c) >=2 chunks => aggregate over ALL positions, not last-chunk-only
    kls = [mx.array([0.0, 0.0]), mx.array([2.0, 2.0])]
    flips = [mx.array([False, False]), mx.array([True, True])]
    ref_nlls = [mx.array([1.0, 1.0]), mx.array([1.0, 1.0])]
    quant_nlls = [mx.array([1.0, 1.0]), mx.array([2.0, 2.0])]
    agg = _aggregate_chunks(kls, flips, ref_nlls, quant_nlls)
    assert agg.kl.mean == 1.0  # (0+0+2+2)/4, not 2.0 (last chunk only)
    assert agg.flip_rate == 0.5
    assert agg.n_positions == 4
    # perplexity must pool ALL positions, not last-chunk-only (catches a broken aggregator)
    assert math.isclose(agg.perplexity_ref, math.exp(1.0), abs_tol=1e-4)  # all ref nll = 1.0
    assert math.isclose(agg.perplexity_quant, math.exp(1.5), abs_tol=1e-4)  # quant nll mean = 1.5


# ---------------------------------------------------------------------------
# Task 4.2 — _score_chunk: teacher-forced loop, reduce/eval, KLD detection
# ---------------------------------------------------------------------------


class _FakeModel:
    """Returns fixed logits; the 'quantized' cache flag perturbs them deterministically."""

    def __call__(self, ids, cache):
        base = mx.zeros((1, ids.shape[1], 3))
        # standard cache -> peak on token 0; quantized cache (has .bits) -> peak on token 1
        bump = 1 if getattr(cache[0], "bits", None) is not None else 0
        onehot = mx.zeros((1, ids.shape[1], 3))
        onehot[:, :, bump] = 5.0
        return base + onehot


def test_score_chunk_detects_quantized_divergence():
    model = _FakeModel()
    ids = mx.array([0, 1, 2])  # vocab=3; targets become [1, 2] — in-vocab
    ref_cache = [type("C", (), {"bits": None})()]
    quant_cache = [type("C", (), {"bits": 4})()]
    kl, flips, ref_nll, quant_nll = _score_chunk(model, ids, ref_cache, quant_cache)
    mx.eval(kl, flips, ref_nll, quant_nll)
    assert kl.shape == (2,)  # L-1 teacher-forced positions (ids has 3, scores 2)
    # ref peak [5,0,0] vs quant peak [0,5,0] -> per-position KL(ref||quant) = 4.90 nats (hand-computed)
    assert math.isclose(float(kl.mean()), 4.90, abs_tol=0.02)
    assert int(flips.astype(mx.int32).sum()) == ids.shape[0] - 1  # every (L-1) position flipped
    # NLL: ref logits [5,0,0] at targets [1,2] -> NLL = 5.0134 each
    # quant logits [0,5,0] at target 1 -> NLL = 0.0134 (peak ON target); at target 2 -> 5.0134
    assert math.isclose(float(ref_nll.mean()), 5.0134, abs_tol=0.01)
    assert float(quant_nll[0]) < float(ref_nll[0])  # quant peak is ON target 1 -> much lower NLL


# ---------------------------------------------------------------------------
# Task 4.3 — deployment-mode guard (raises BEFORE any model load)
# ---------------------------------------------------------------------------


def test_quantize_start_nonzero_raises_before_model_load():
    with pytest.raises(QuantFidelityError, match="deployment mode"):
        measure_kv_fidelity("any-string", quantize_start=1)


# ---------------------------------------------------------------------------
# Task 4.4 — gate ValueError branch and unsupported-bits check
# ---------------------------------------------------------------------------


class _ValueErrorCache:
    def to_quantized(self, **_kwargs):
        raise ValueError("group_size=64 does not divide head_dim=48")


def test_capability_gate_value_error_branch():
    # ValueError from to_quantized -> CacheNotQuantizableError naming the type
    with pytest.raises(CacheNotQuantizableError, match="_ValueErrorCache"):
        _cache_is_quantizable([_ValueErrorCache()], group_size=64, bits=4)


def test_capability_gate_unsupported_bits():
    # bits=5 is not in (2,3,4,6,8) -> CacheNotQuantizableError naming kv_bits=5
    with pytest.raises(CacheNotQuantizableError, match="kv_bits=5"):
        _cache_is_quantizable([_OkCache()], group_size=64, bits=5)


# ---------------------------------------------------------------------------
# Task 4.5 — memory cap is installed BEFORE the model load (safety ordering)
# ---------------------------------------------------------------------------


def test_measure_installs_caps_before_model_load(monkeypatch):
    # Patch the cap installer and mlx_lm.load to record order; bail at load so no model
    # is fetched. A regression that drops or reorders the cap install goes red here.
    import mlx_lm

    from mlx_quant_fidelity.probes import kv as kv_mod

    calls: list[str] = []

    class _LoadStopError(Exception):
        pass

    monkeypatch.setattr(kv_mod, "install_memory_caps", lambda: calls.append("caps") or (0, 0))

    def _fake_load(*_a, **_k):
        calls.append("load")
        raise _LoadStopError

    monkeypatch.setattr(mlx_lm, "load", _fake_load)

    with pytest.raises(_LoadStopError):
        measure_kv_fidelity("any-model", quantize_start=0)
    assert calls == ["caps", "load"]  # caps installed BEFORE the model load


# ---------------------------------------------------------------------------
# Task 4.6 — empty-corpus / bad max_chunks raise a clean CorpusError before load
# ---------------------------------------------------------------------------


def test_max_chunks_zero_raises_clean_error():
    with pytest.raises(CorpusError, match="max_chunks"):
        measure_kv_fidelity("any-model", max_chunks=0)


def test_max_chunks_negative_raises_clean_error():
    with pytest.raises(CorpusError, match="max_chunks"):
        measure_kv_fidelity("any-model", max_chunks=-1)


def test_empty_corpus_raises_clean_error():
    from mlx_quant_fidelity.corpora.provenance import Corpus, CorpusProvenance

    empty = Corpus(
        chunks=(),
        provenance=CorpusProvenance("x", "test", "t", 512, 512, "none", "drop", "raw", 0),
    )
    with pytest.raises(CorpusError, match="no chunks"):
        measure_kv_fidelity("any-model", corpus=empty)


# ---------------------------------------------------------------------------
# Task 0009 — head_dim gate: group-size divisibility check before scoring
# ---------------------------------------------------------------------------


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _ModelWithArgs:
    def __init__(self, args):
        self.args = args


def test_kv_head_dim_prefers_explicit_then_derives():
    assert _kv_head_dim(_ModelWithArgs(_Args(head_dim=48))) == 48
    assert _kv_head_dim(_ModelWithArgs(_Args(hidden_size=4096, num_attention_heads=32))) == 128
    assert _kv_head_dim(_ModelWithArgs(_Args(head_dim=None))) is None  # nothing derivable


def test_head_dim_gate_raises_on_non_divisor():
    with pytest.raises(CacheNotQuantizableError, match="head_dim=48"):
        _head_dim_gate(head_dim=48, kv_group_size=64, model_type="llama")


def test_head_dim_gate_passes_on_divisor():
    assert _head_dim_gate(head_dim=64, kv_group_size=64, model_type="llama") is None


def test_head_dim_gate_warns_when_unknown():
    warning = _head_dim_gate(head_dim=None, kv_group_size=64, model_type="mystery")
    assert warning is not None
    assert "mystery" in warning


def test_measure_kv_rejects_incompatible_group_size_before_scoring(monkeypatch):
    """An incompatible kv_group_size raises CacheNotQuantizableError after load but BEFORE
    make_prompt_cache / any scoring (monkeypatched load + make_prompt_cache record order)."""
    import mlx_lm
    import mlx_lm.models.cache as cache_mod

    from mlx_quant_fidelity.corpora.provenance import Corpus, CorpusProvenance
    from mlx_quant_fidelity.probes import kv as kv_mod

    monkeypatch.setattr(kv_mod, "install_memory_caps", lambda: (0, 0))
    stub = _ModelWithArgs(_Args(head_dim=48, model_type="llama"))
    monkeypatch.setattr(mlx_lm, "load", lambda *a, **k: (stub, object()))
    cache_calls: list[str] = []
    monkeypatch.setattr(
        cache_mod, "make_prompt_cache", lambda *a, **k: cache_calls.append("cache") or []
    )
    corpus = Corpus(
        chunks=(mx.array([0, 1, 2, 3]),),
        provenance=CorpusProvenance("x", "test", "t", 4, 4, "none", "drop", "raw", 4),
    )
    with pytest.raises(CacheNotQuantizableError, match="head_dim=48"):
        measure_kv_fidelity("any-model", kv_bits=4, kv_group_size=64, corpus=corpus)
    assert cache_calls == []  # gate fired BEFORE make_prompt_cache / scoring


def test_kv_head_dim_zero_falls_back_to_derived():
    # head_dim=0 is falsy -> falls back to hidden//heads (documented "truthy" contract)
    assert (
        _kv_head_dim(_ModelWithArgs(_Args(head_dim=0, hidden_size=4096, num_attention_heads=32)))
        == 128
    )
