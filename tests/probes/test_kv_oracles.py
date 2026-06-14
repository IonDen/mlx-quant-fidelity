"""Real-model oracle tests for the KV-quant probe.

These are ``@pytest.mark.slow`` and require a real model load (mlx-community/Llama-3.2-1B-Instruct-4bit).
They are skipped by default; run with ``--run-slow`` on the main thread.
DO NOT run in CI or in a subagent.
"""

import mlx.core as mx
import pytest

from mlx_quant_fidelity.corpora.provenance import Corpus, CorpusProvenance
from mlx_quant_fidelity.metrics import kl_divergence
from mlx_quant_fidelity.probes.kv import _score_chunk, measure_kv_fidelity

MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"  # match the spike's model


def _tiny_corpus(tok, *, chunk_length: int = 64, n_chunks: int = 2) -> Corpus:
    """In-memory corpus so the stress oracle stays slow-only (no --run-network needed)."""
    ids = tok.encode("Teacher forcing keeps both runs on identical tokens. " * 40)
    chunks = tuple(
        mx.array(ids[i * chunk_length : (i + 1) * chunk_length]) for i in range(n_chunks)
    )
    prov = CorpusProvenance(
        "in-memory",
        "test",
        "tok",
        chunk_length,
        chunk_length,
        "none",
        "drop",
        "raw",
        chunk_length * n_chunks,
    )
    return Corpus(chunks=chunks, provenance=prov)


@pytest.mark.slow
def test_full_vs_full_is_exactly_zero():
    """KVCache vs KVCache, in-process, same test -> EXACTLY 0 (equality, not tolerance, no saved fixture)."""
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tok = load(MODEL)
    ids = mx.array(tok.encode("Teacher forcing keeps both runs on identical tokens." * 5)[:128])
    a = model(ids[None], cache=make_prompt_cache(model))[0].astype(mx.float32)
    b = model(ids[None], cache=make_prompt_cache(model))[0].astype(mx.float32)
    kl = kl_divergence(a, b)
    mx.eval(kl)
    assert bool((kl == 0).all())


@pytest.mark.slow
def test_stress_quantization_engages():
    """QuantizedKVCache from token 0 produces non-zero KLD and valid flip_rate."""
    from mlx_lm import load

    _model, tok = load(MODEL)
    del _model
    mx.clear_cache()
    report = measure_kv_fidelity(MODEL, kv_bits=4, quantize_start=0, corpus=_tiny_corpus(tok))
    assert report.kl.mean > 0.01  # spike measured ~0.083 for 4-bit; 0.01 is a safe non-flaky floor
    assert report.peak_memory_bytes > 0


@pytest.mark.slow
def test_corrupted_cache_kld_rises_and_control_is_zero():
    """Proves the quantized cache is ACTUALLY CONSUMED.

    A zeroing cache -> KLD rises; the same corruption on BOTH paths -> exactly 0.
    Threshold from the spike doc (>0.1 nats, spike measured 8.46).
    """
    from mlx_lm import load
    from mlx_lm.models.cache import KVCache, make_prompt_cache

    model, tok = load(MODEL)
    ids = mx.array(tok.encode("Proves the quantized cache is consumed." * 8)[:128])
    n_layers = len(make_prompt_cache(model))

    class _ZeroCache(KVCache):
        def update_and_fetch(self, keys, values):
            k, v = super().update_and_fetch(keys, values)
            return mx.zeros_like(k), mx.zeros_like(v)

    ref = _score_chunk(model, ids, make_prompt_cache(model), make_prompt_cache(model))[0]
    corrupted = _score_chunk(
        model,
        ids,
        make_prompt_cache(model),
        [_ZeroCache() for _ in range(n_layers)],
    )[0]
    both = _score_chunk(
        model,
        ids,
        [_ZeroCache() for _ in range(n_layers)],
        [_ZeroCache() for _ in range(n_layers)],
    )[0]
    mx.eval(ref, corrupted, both)
    assert float(ref.mean()) == 0.0  # full-vs-full control -> exactly 0
    assert (
        float(corrupted.mean()) > 0.1
    )  # corruption consumed -> KLD rises past the spike threshold
    assert float(both.mean()) == 0.0  # same corruption on both -> back to exactly 0
