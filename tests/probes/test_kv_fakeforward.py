"""Offline fake-forward tests for the KV-quant probe (no real model load)."""

import math

import mlx.core as mx
import pytest

from mlx_quant_fidelity.corpora.provenance import Corpus, CorpusProvenance
from mlx_quant_fidelity.errors import (
    CacheNotQuantizableError,
    CorpusError,
    ExactZeroError,
)
from mlx_quant_fidelity.probes import kv as kvmod
from mlx_quant_fidelity.probes.kv import (
    _aggregate_chunks,
    _cache_is_quantizable,
    _check_exact_zero,
    _head_dim_gate,
    _kv_head_dim,
    _score_chunk,
    measure_kv_fidelity,
    packed_width_mismatch,
    score_kv_config,
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


def test_score_chunk_identical_paths_is_exactly_zero():
    # full-vs-full through _score_chunk: identical logits on both paths reduce to EXACTLY 0
    # (KL(P||P)=0 for cache-agnostic identical outputs). _FakeKVModel ignores the cache, so the ref
    # and quant paths are identical. Paired with test_score_chunk_detects_quantized_divergence this
    # pins both ends of _score_chunk (the weight probe has the analogous identical-is-zero test).
    model = _FakeKVModel(peak=1)
    ids = mx.array([0, 1, 2])
    ref_cache = [type("C", (), {"bits": None})()]
    quant_cache = [type("C", (), {"bits": 4})()]
    kl, flips, ref_nll, quant_nll = _score_chunk(model, ids, ref_cache, quant_cache)
    mx.eval(kl, flips, ref_nll, quant_nll)
    assert float(kl.mean()) == 0.0
    assert int(flips.astype(mx.int32).sum()) == 0


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
    make_prompt_cache / any scoring (monkeypatched load + make_prompt_cache record order).

    The patch targets ``kv_mod.make_prompt_cache`` — the name that ``score_kv_config``
    actually uses — because kv.py binds the name locally via ``from mlx_lm.models.cache import
    make_prompt_cache``.  Patching ``mlx_lm.models.cache.make_prompt_cache`` would be
    disconnected and leave the assertion vacuously true.
    """
    import mlx_lm

    from mlx_quant_fidelity.corpora.provenance import Corpus, CorpusProvenance
    from mlx_quant_fidelity.probes import kv as kv_mod

    monkeypatch.setattr(kv_mod, "install_memory_caps", lambda: (0, 0))
    stub = _ModelWithArgs(_Args(head_dim=48, model_type="llama"))
    monkeypatch.setattr(mlx_lm, "load", lambda *a, **k: (stub, object()))
    cache_calls: list[str] = []
    monkeypatch.setattr(
        kv_mod, "make_prompt_cache", lambda *a, **k: cache_calls.append("cache") or []
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


# ---------------------------------------------------------------------------
# Task 4 — score_kv_config: extracted helper operating on an already-loaded model
# ---------------------------------------------------------------------------


class _FakeKVModel:
    """Forward that ignores the cache and returns a fixed peak per position (peak index `peak`).

    Returns a raw tensor [1, L, vocab] (matching real mlx-lm model behaviour): _score_chunk
    indexes [0] on the tensor to squeeze the batch dim to [L, vocab].
    """

    def __init__(self, peak):
        self._peak = peak
        self.args = type("A", (), {"model_type": "llama"})()

    def __call__(self, inp, cache=None):  # inp [1, L]; returns [1, L, 3]
        length = inp.shape[1]
        out = mx.zeros((1, length, 3))
        out[:, :, self._peak] = 5.0
        return out


class _FakeDivergentModel:
    """Mirrors _FakeModel's cache-bump logic with proper args for score_kv_config.

    Returns peak on token 0 for ref cache (no .bits) and peak on token 1 for quant cache
    (has .bits), producing divergent logits so KLD > 0 and _check_exact_zero runs live.
    head_dim=64, kv_group_size default 64: 64 % 64 == 0 so no head_dim warning fires.
    """

    def __init__(self, head_dim=64):
        self.args = type(
            "A",
            (),
            {
                "model_type": "llama",
                "head_dim": head_dim,
                "hidden_size": None,
                "num_attention_heads": None,
            },
        )()

    def __call__(self, inp, cache=None):  # inp [1, L]; returns [1, L, 3]
        bump = 1 if (cache is not None and getattr(cache[0], "bits", None) is not None) else 0
        out = mx.zeros((1, inp.shape[1], 3))
        out[:, :, bump] = 5.0
        return out


class _FakeLayerCache:
    def to_quantized(self, group_size, bits):  # present + non-raising -> quantizable
        return self


class _FakeQuantCache:
    """A quantized cache: has .bits so _FakeDivergentModel returns quant (peak-1) logits."""

    bits = 4
    state = property(lambda self: ())

    def to_quantized(self, group_size, bits):
        return self


def _kv_corpus(n_chunks, chunk_len=4):
    chunks = tuple(mx.arange(chunk_len) for _ in range(n_chunks))
    prov = CorpusProvenance(
        "x", "test", "org/m", chunk_len, chunk_len, "none", "drop", "raw", chunk_len * n_chunks
    )
    return Corpus(chunks=chunks, provenance=prov)


def _patch_kv_caches_divergent(monkeypatch, n_layers=2):
    """Patch cache helpers so ref and quant caches are distinguishable by .bits.

    make_prompt_cache returns _FakeLayerCache (no .bits → ref peak 0).
    QuantizedKVCache returns _FakeQuantCache (has .bits=4 → quant peak 1).
    _check_exact_zero is NOT silenced: divergent logits yield KLD > 0.
    """
    monkeypatch.setattr(
        kvmod,
        "make_prompt_cache",
        lambda model: [_FakeLayerCache() for _ in range(n_layers)],
        raising=False,
    )
    monkeypatch.setattr(
        kvmod, "QuantizedKVCache", lambda group_size, bits: _FakeQuantCache(), raising=False
    )


def test_score_kv_config_offline_reaches_loop(monkeypatch):
    """score_kv_config runs the full chunk loop and produces divergent, non-zero KLD.

    Uses _FakeDivergentModel (head_dim=64, model_type=llama) so:
    - _kv_head_dim returns 64; 64 % 64 == 0 → NO head_dim warning (warnings == ())
    - ref cache has no .bits → peak on token 0; quant cache has .bits=4 → peak on token 1
    - KLD > 0 → _check_exact_zero runs live (not silenced)

    A constant-zero _score_chunk would leave kl.mean == 0 and trigger ExactZeroError (red).
    Stubbing _score_chunk to zeros would also go red.  Dropping the max_chunks slice would
    change n_chunks (red for test_score_kv_config_caps_provided_corpus).
    """
    _patch_kv_caches_divergent(monkeypatch)
    report = score_kv_config(_FakeDivergentModel(), _kv_corpus(2), model_id="org/m")
    assert report.n_chunks == 2
    assert report.n_positions == 6  # 2 chunks x (4-1) teacher-forced positions
    assert report.cache_supported is True
    assert report.kl.mean > 0  # divergent caches → KLD must be non-zero
    assert report.warnings == ()  # head_dim=64 divides group_size=64 → no warning
    assert report.quantize_mode == "stress"  # mutation guard: unconditional "deployment" goes RED


def test_score_kv_config_caps_provided_corpus(monkeypatch):
    """max_chunks caps a caller-provided corpus (backlog 0012).

    Uses the same divergent model so _check_exact_zero is not silenced and a broken
    max_chunks slice (e.g. dropped corpus.chunks[:max_chunks]) would change n_chunks (red).
    """
    _patch_kv_caches_divergent(monkeypatch)
    report = score_kv_config(_FakeDivergentModel(), _kv_corpus(5), model_id="org/m", max_chunks=2)
    assert report.n_chunks == 2
    assert report.n_positions == 6


def test_score_kv_config_emits_warning_when_head_dim_unknown(monkeypatch):
    """score_kv_config appends a warning when head_dim is not derivable (covers kv.py line 135).

    Uses _FakeDivergentModel with head_dim=None and no hidden_size/num_attention_heads
    so _kv_head_dim returns None → _head_dim_gate returns a warning string → line 135 runs.
    KLD is still > 0 (divergent caches) so _check_exact_zero does not fire.
    """

    class _UnknownHeadDimModel:
        args = type(
            "A",
            (),
            {
                "model_type": "mystery",
                "head_dim": None,
                "hidden_size": None,
                "num_attention_heads": None,
            },
        )()

        def __call__(self, inp, cache=None):  # inp [1, L]; returns [1, L, 3]
            bump = 1 if (cache is not None and getattr(cache[0], "bits", None) is not None) else 0
            out = mx.zeros((1, inp.shape[1], 3))
            out[:, :, bump] = 5.0
            return out

    _patch_kv_caches_divergent(monkeypatch)
    report = score_kv_config(_UnknownHeadDimModel(), _kv_corpus(1), model_id="org/m")
    assert len(report.warnings) == 1
    assert "mystery" in report.warnings[0]


def test_score_kv_config_raises_exact_zero_when_quant_indistinguishable(monkeypatch):
    """The exact-zero guard is wired into score_kv_config, not just unit-tested in isolation.

    Both caches are _FakeLayerCache (no .bits), so _FakeDivergentModel returns identical logits
    on the ref and quant paths -> KLD 0. score_kv_config MUST raise ExactZeroError rather than
    emit a silent "perfect fidelity" report. Guard mutation: deleting the _check_exact_zero(...)
    call in score_kv_config makes this return a report instead of raising.
    """
    monkeypatch.setattr(
        kvmod,
        "make_prompt_cache",
        lambda model: [_FakeLayerCache(), _FakeLayerCache()],
        raising=False,
    )
    monkeypatch.setattr(
        kvmod, "QuantizedKVCache", lambda group_size, bits: _FakeLayerCache(), raising=False
    )
    with pytest.raises(ExactZeroError):
        score_kv_config(_FakeDivergentModel(), _kv_corpus(1), model_id="org/m")


# ---------------------------------------------------------------------------
# Task 5 — _score_chunk_deployment: split-forward scorer for deployment mode
# ---------------------------------------------------------------------------


class _FakeLayerCachePreQ:
    """A full-precision layer cache (no .bits) whose to_quantized() yields a .bits-bearing cache.

    The existing _FakeLayerCache.to_quantized() returns self (still no .bits), which would make
    the post-boundary switch invisible to _FakeDivergentModel. `.state` is required because
    _score_chunk_deployment evals [c.state for c in quant_cache] before the boundary.
    """

    state = property(lambda self: ())

    def to_quantized(self, group_size, bits):
        return _FakeQuantCache()


def test_score_chunk_deployment_boundary():
    from mlx_quant_fidelity.probes.kv import _score_chunk_deployment

    model = _FakeDivergentModel()  # peak-0 for no-.bits cache, peak-1 for .bits cache
    ids = mx.arange(6)  # L=6 → 5 prediction positions
    n = 2
    ref_cache = [_FakeLayerCachePreQ()]
    quant_cache = [_FakeLayerCachePreQ()]  # full-precision; converted at the boundary
    kl, flips, ref_nll, quant_nll = _score_chunk_deployment(
        model, ids, ref_cache, quant_cache, quantize_start=n, group_size=64, bits=4
    )
    mx.eval(kl, flips, ref_nll, quant_nll)
    assert kl.shape == (5,)  # all L-1 positions returned
    assert float(kl[:n].max()) == 0.0  # prefix exact-0 (offline)
    assert float(kl[n:].min()) > 0.0  # post-boundary drift engaged
    assert int(kl[n:].shape[0]) == len(ids) - 1 - n  # segment-2 length exactly L-1-N == 3


# ---------------------------------------------------------------------------
# Task 6 — score_kv_config deployment mode (quantize_start > 0)
# ---------------------------------------------------------------------------


def _patch_kv_caches_deployment(monkeypatch, n_layers=1):
    # Deployment builds BOTH caches from make_prompt_cache; the quant cache converts at the boundary.
    monkeypatch.setattr(
        kvmod,
        "make_prompt_cache",
        lambda model: [_FakeLayerCachePreQ() for _ in range(n_layers)],
        raising=False,
    )


def test_score_kv_config_deployment_reports_post_boundary(monkeypatch):
    _patch_kv_caches_deployment(monkeypatch)
    report = score_kv_config(
        _FakeDivergentModel(), _kv_corpus(1, 6), model_id="org/m", quantize_start=2
    )
    assert report.quantize_mode == "deployment"
    assert report.quantize_start == 2
    assert report.n_positions == 3  # post-boundary [2:5) only, NOT all 5 (pins the slice)
    assert report.kl.mean > 0


def test_score_kv_config_deployment_skips_short_chunk(monkeypatch):
    # A mixed corpus: one full (len 6) + one short (len 3 < N+2 for N=2). Short is skipped, not fatal.
    _patch_kv_caches_deployment(monkeypatch)
    mixed = Corpus(
        chunks=(mx.arange(6), mx.arange(3)),
        provenance=CorpusProvenance("x", "test", "org/m", 6, 6, "none", "drop", "raw", 9),
    )
    report = score_kv_config(_FakeDivergentModel(), mixed, model_id="org/m", quantize_start=2)
    assert report.n_positions == 3  # only the full chunk contributed (3 post-boundary)
    assert report.n_chunks == 1  # the short chunk was skipped


def test_score_kv_config_deployment_all_short_raises(monkeypatch):
    from mlx_quant_fidelity.errors import QuantizeStartError

    _patch_kv_caches_deployment(monkeypatch)
    short = Corpus(
        chunks=(mx.arange(3),),
        provenance=CorpusProvenance("x", "test", "org/m", 6, 6, "none", "drop", "raw", 3),
    )
    with pytest.raises(QuantizeStartError, match="no position"):
        score_kv_config(_FakeDivergentModel(), short, model_id="org/m", quantize_start=2)


def test_measure_kv_deployment_boundary_validated_before_load(monkeypatch):
    from mlx_quant_fidelity.errors import QuantizeStartError

    monkeypatch.setattr(kvmod, "install_memory_caps", lambda: (0, 0))
    with pytest.raises(QuantizeStartError):
        measure_kv_fidelity("any-model", quantize_start=511)  # == window-1 → zero quantized
    with pytest.raises(QuantizeStartError):
        measure_kv_fidelity("any-model", quantize_start=-1)


# ---------------------------------------------------------------------------
# 0031 — bits=6 packed-width mismatch gate (mlx-lm QuantizedKVCache append crash)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("head_dim", "bits", "expected"),
    [
        (128, 6, True),  # 128 // 5 = 25  vs  128 * 6 // 32 = 24 — the 0031 crash
        (96, 6, True),  # 96 // 5 = 19   vs  96 * 6 // 32 = 18
        (64, 6, False),  # 12 == 12
        (128, 3, False),  # 128 // 10 = 12 vs  128 * 3 // 32 = 12
        (128, 2, False),
        (128, 4, False),
        (128, 8, False),
    ],
)
def test_packed_width_mismatch(head_dim, bits, expected):
    assert packed_width_mismatch(head_dim, bits) is expected


def test_bits6_head_dim_128_rejected_package_rooted():
    # head_dim=128 % kv_group_size=32 == 0 (no head_dim warning); the packed-width gate fires
    # before make_prompt_cache / any forward, so a bare args stub is enough.
    model = _FakeDivergentModel(head_dim=128)
    with pytest.raises(CacheNotQuantizableError, match="mlx-lm"):
        score_kv_config(model, _kv_corpus(1), model_id="fake", kv_bits=6, kv_group_size=32)


def test_bits6_head_dim_64_still_measures(monkeypatch):
    # head_dim=64 agrees at bits=6 (12 == 12) so the gate does not fire — scoring proceeds.
    _patch_kv_caches_divergent(monkeypatch)
    report = score_kv_config(
        _FakeDivergentModel(head_dim=64),
        _kv_corpus(2),
        model_id="fake",
        kv_bits=6,
        kv_group_size=32,
    )
    assert report.kv_bits == 6


def test_bits5_head_dim_128_reports_unsupported_bits_not_packing_bug():
    # Regression: the packed-width gate must not shadow the unsupported-bits check.
    # packed_width_mismatch(128, 5) is True (128 // 6 = 21 != 128 * 5 // 32 = 20), but the
    # message users see must be the "unsupported kv_bits" one, not the mlx-lm packing message.
    model = _FakeDivergentModel(head_dim=128)
    with pytest.raises(CacheNotQuantizableError, match="unsupported kv_bits=5"):
        score_kv_config(model, _kv_corpus(1), model_id="fake", kv_bits=5, kv_group_size=32)


def test_bits0_head_dim_128_raises_clean_error_not_zerodivision():
    # Regression: packed_width_mismatch's `32 // bits` must never run on an unvalidated bits=0.
    model = _FakeDivergentModel(head_dim=128)
    with pytest.raises(CacheNotQuantizableError, match="unsupported kv_bits=0"):
        score_kv_config(model, _kv_corpus(1), model_id="fake", kv_bits=0, kv_group_size=32)


class _FakeBroadcastCrashModel:
    """head_dim=64 (agrees at any gated bits) so the early gate does NOT fire; the forward call
    against a quantized cache raises the raw mlx-lm broadcast_shapes ValueError instead, so this
    exercises the belt catch inside the stress-mode chunk loop.
    """

    def __init__(self):
        self.args = type(
            "A",
            (),
            {
                "model_type": "llama",
                "head_dim": 64,
                "hidden_size": None,
                "num_attention_heads": None,
            },
        )()

    def __call__(self, inp, cache=None):
        if cache is not None and getattr(cache[0], "bits", None) is not None:
            raise ValueError("[broadcast_shapes] Shapes (1,4,25) and (1,4,24) cannot be broadcast.")
        return mx.zeros((1, inp.shape[1], 3))


def test_score_kv_config_wraps_broadcast_shapes_crash(monkeypatch):
    _patch_kv_caches_divergent(monkeypatch)
    with pytest.raises(CacheNotQuantizableError, match="broadcast_shapes"):
        score_kv_config(_FakeBroadcastCrashModel(), _kv_corpus(1), model_id="fake")


class _FakeUnrelatedValueErrorModel(_FakeBroadcastCrashModel):
    """Same shape as _FakeBroadcastCrashModel but the quant-pass ValueError is unrelated to the
    packed-width bug — the belt catch must not swallow it into a misleading CacheNotQuantizableError.
    """

    def __call__(self, inp, cache=None):
        if cache is not None and getattr(cache[0], "bits", None) is not None:
            raise ValueError("some unrelated numerical error")
        return mx.zeros((1, inp.shape[1], 3))


def test_score_kv_config_reraises_unrelated_value_error(monkeypatch):
    _patch_kv_caches_divergent(monkeypatch)
    with pytest.raises(ValueError, match="unrelated numerical error"):
        score_kv_config(_FakeUnrelatedValueErrorModel(), _kv_corpus(1), model_id="fake")
