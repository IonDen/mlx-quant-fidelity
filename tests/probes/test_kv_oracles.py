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


def _tiny_corpus(tok, *, chunk_length: int = 64, n_chunks: int = 2, repeats: int = 40) -> Corpus:
    """In-memory corpus so the stress oracle stays slow-only (no --run-network needed)."""
    ids = tok.encode("Teacher forcing keeps both runs on identical tokens. " * repeats)
    if len(ids) < chunk_length * n_chunks:
        raise ValueError(f"need {chunk_length * n_chunks} tokens, have {len(ids)}; raise repeats")
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


_DEPLOY_BANDS = {
    "stock": (0.3, 3.0),
    "turboquant": (0.3, 3.0),
}  # a-priori; see the module docstring


@pytest.mark.slow
@pytest.mark.parametrize("method_name", ["stock", "turboquant"])
def test_deployment_post_boundary_matches_stress(method_name):
    """Deployment reports post-boundary positions only; per-token drift is the same order as stress.

    Recorded ratios (main-thread run 2026-08-24, M1 Max 32 GB): stock 0.4537 ; turboquant 0.3707 .
    """
    from mlx_lm import load

    from mlx_quant_fidelity.errors import MethodUnavailableError
    from mlx_quant_fidelity.probes.kv import score_kv_config
    from mlx_quant_fidelity.probes.kv_methods import StockKVMethod, TurboQuantKVMethod

    method = (
        StockKVMethod(bits=4, group_size=64)
        if method_name == "stock"
        else TurboQuantKVMethod(bits=4)
    )
    if method_name == "turboquant":
        try:
            method.probe_capability([])
        except MethodUnavailableError as exc:
            pytest.skip(str(exc))
    model, tok = load(MODEL)
    corpus = _tiny_corpus(tok, chunk_length=64, n_chunks=2)
    stress = score_kv_config(model, corpus, model_id=MODEL, method=method, quantize_start=0)
    mx.clear_cache()
    deploy = score_kv_config(model, corpus, model_id=MODEL, method=method, quantize_start=32)
    mx.clear_cache()
    assert deploy.quantize_mode == "deployment"
    assert deploy.n_positions == (64 - 1 - 32) * 2
    assert stress.n_positions == (64 - 1) * 2
    assert deploy.kl.mean > 0
    assert stress.kl.mean > 0
    ratio = deploy.kl.mean / stress.kl.mean
    print(f"{method_name} deploy/stress ratio = {ratio:.4f}")
    lo, hi = _DEPLOY_BANDS[method_name]
    assert lo < ratio < hi


@pytest.mark.slow
@pytest.mark.network
def test_stock_method_reproduces_committed_sample_mean():
    """Re-measure llama-3.2-1b-4bit-kv4.json through the seam; kl.mean must match to 1e-6."""
    import json
    from pathlib import Path

    from mlx_quant_fidelity.probes.kv import measure_kv_fidelity

    sample = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "_artifacts"
            / "samples"
            / "llama-3.2-1b-4bit-kv4.json"
        ).read_text()
    )
    report = measure_kv_fidelity(
        MODEL,
        kv_bits=4,
        kv_group_size=64,
        model_revision=sample["model_revision"],
        max_chunks=sample["n_chunks"],
    )
    assert abs(report.kl.mean - sample["kl"]["mean"]) < 1e-6  # 0.14773540842612012 committed
    assert report.measured_kv_bytes_per_token == 9216


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
