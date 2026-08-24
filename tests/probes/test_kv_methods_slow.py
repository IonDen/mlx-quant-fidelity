"""Real-port oracles for TurboQuantKVMethod (mlx-community/Llama-3.2-1B-Instruct-4bit).

``@pytest.mark.slow``; run with ``--run-slow`` on the main thread. DO NOT run in CI or in a subagent.
Requires the pinned port: see ``TURBOQUANT_INSTALL_HINT``. Skips when it is absent; tests that
need no port run regardless.

Pre-registered hypotheses (fixed 2026-08-23 from the spike, before any implementation run):
  t4.kl.mean < s4.kl.mean and t4.kl.p99 < s4.kl.p99; t3.kl.mean > s4.kl.mean (spike: 0.0255 / 0.0476 / 0.1773).
Recorded values (main-thread run, date, machine): s4 ____  t4 ____  t3 ____  seed-43 t4 ____ .
"""

import mlx.core as mx
import pytest
from tests.probes.test_kv_oracles import MODEL, _tiny_corpus

from mlx_quant_fidelity.errors import ExactZeroError, MethodUnavailableError
from mlx_quant_fidelity.probes.kv import score_kv_config
from mlx_quant_fidelity.probes.kv_methods import (
    StockKVMethod,
    TurboQuantKVMethod,
    stored_state_bytes,
    turboquant_bytes_per_token,
)

pytestmark = pytest.mark.slow


def _port_or_skip():
    try:
        TurboQuantKVMethod(bits=4).probe_capability([])
    except MethodUnavailableError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def loaded():
    from mlx_lm import load

    return load(MODEL)


@pytest.fixture(scope="module")
def loaded_with_port(loaded):
    _port_or_skip()  # after the load, so a missing port skips only the tests that need it
    return loaded


def test_ordering_oracle_turbo4_beats_stock4_and_turbo3_loses(loaded_with_port):
    """The launch claim, asserted on the same corpus in one test."""
    model, tok = loaded_with_port
    corpus = _tiny_corpus(tok, chunk_length=64, n_chunks=4)
    s4 = score_kv_config(model, corpus, model_id=MODEL, method=StockKVMethod(bits=4, group_size=64))
    mx.clear_cache()
    t4 = score_kv_config(model, corpus, model_id=MODEL, method=TurboQuantKVMethod(bits=4))
    mx.clear_cache()
    t3 = score_kv_config(model, corpus, model_id=MODEL, method=TurboQuantKVMethod(bits=3))
    mx.clear_cache()
    print(
        f"s4 mean={s4.kl.mean:.4f} p99={s4.kl.p99:.4f}  t4 mean={t4.kl.mean:.4f} p99={t4.kl.p99:.4f}  t3 mean={t3.kl.mean:.4f}"
    )
    assert 0 < t4.kl.mean < s4.kl.mean
    assert t4.kl.p99 < s4.kl.p99
    assert t3.kl.mean > s4.kl.mean
    assert t4.kv_method == "turboquant"
    assert t4.kv_group_size is None
    assert any("standard SDPA" in w for w in t4.warnings)


def test_stock_analytic_equals_measured_at_offset_511(loaded):
    """D5 proof for stock, no port needed: the probe feeds 511 tokens, not a step-256 multiple."""
    model, tok = loaded
    corpus = _tiny_corpus(tok, chunk_length=512, n_chunks=1, repeats=200)
    for method, expected in (
        (StockKVMethod(bits=4, group_size=64), 9216),
        (StockKVMethod(bits=8, group_size=64), 17408),
    ):
        rep = score_kv_config(model, corpus, model_id=MODEL, method=method)
        mx.clear_cache()
        assert rep.measured_kv_bytes_per_token == expected, method.label
        assert not any("differs from the analytic" in w for w in rep.warnings)


def test_turboquant_analytic_equals_measured_at_offset_511(loaded_with_port):
    model, tok = loaded_with_port
    corpus = _tiny_corpus(tok, chunk_length=512, n_chunks=1, repeats=200)
    for bits, expected in ((4, 9216), (3, 8192), (2, 5120)):
        rep = score_kv_config(model, corpus, model_id=MODEL, method=TurboQuantKVMethod(bits=bits))
        mx.clear_cache()
        assert rep.measured_kv_bytes_per_token == expected, bits
        assert not any("differs from the analytic" in w for w in rep.warnings)


def test_turboquant_analytic_equals_measured_in_deployment_mode(loaded_with_port):
    model, tok = loaded_with_port
    corpus = _tiny_corpus(tok, chunk_length=512, n_chunks=1, repeats=200)
    rep = score_kv_config(
        model, corpus, model_id=MODEL, method=TurboQuantKVMethod(bits=4), quantize_start=100
    )
    assert rep.measured_kv_bytes_per_token == 9216


def test_vendored_packing_table_matches_port():
    _port_or_skip()
    from turboquant_mlx.packing import packed_dim

    for bits in (2, 3, 4):
        for head_dim in (64, 128, 256):
            ours = turboquant_bytes_per_token(
                n_layers=1, n_kv_heads=1, head_dim=head_dim, bits=bits
            )
            assert ours == (packed_dim(head_dim, bits) * 4 + 4) * 2


def test_numeric_sanity_roundtrip_cosine_4bit():
    """Distinguishes 'bad quantizer' from 'illegal geometry' at bits=4, head_dim 64.

    Pre-registered floor 0.95 (spike relative error ~0.09 -> cosine ~0.995). 3-bit ~0.983 and
    2-bit ~0.941 would NOT clear it — this test is 4-bit only by design. fp16 input, the dtype a
    real forward feeds.
    """
    _port_or_skip()
    from turboquant_mlx.cache import TurboQuantKVCache

    mx.random.seed(0)
    x = mx.random.normal((1, 8, 64, 64)).astype(mx.float16)
    c = TurboQuantKVCache(bits=4, seed=42, fused=False)
    k, _ = c.update_and_fetch(x, x)
    xf, kf = x.astype(mx.float32), k.astype(mx.float32)
    cos = (xf * kf).sum() / (mx.sqrt((xf * xf).sum()) * mx.sqrt((kf * kf).sum()))
    assert float(cos) > 0.95


def test_two_seed_spread_is_small_relative_to_the_gap(loaded_with_port):
    model, tok = loaded_with_port
    corpus = _tiny_corpus(tok, chunk_length=64, n_chunks=4)
    s4 = score_kv_config(
        model, corpus, model_id=MODEL, method=StockKVMethod(bits=4, group_size=64)
    ).kl.mean
    a = score_kv_config(
        model, corpus, model_id=MODEL, method=TurboQuantKVMethod(bits=4, seed=42)
    ).kl.mean
    b = score_kv_config(
        model, corpus, model_id=MODEL, method=TurboQuantKVMethod(bits=4, seed=43)
    ).kl.mean
    mx.clear_cache()
    print(f"seed 42 -> {a:.4f}, seed 43 -> {b:.4f}, stock {s4:.4f}")
    assert max(a, b) < s4, "precondition: both seeds beat stock (the ordering oracle)"
    assert abs(a - b) < 0.5 * (s4 - max(a, b))  # the seed moves the number less than half the gap


def test_corrupted_adapter_trips_exact_zero(loaded_with_port):
    from mlx_lm.models.cache import KVCache

    model, tok = loaded_with_port

    class _Bypass(TurboQuantKVMethod):
        def make_cache(self, *, n_layers):
            return [KVCache() for _ in range(n_layers)]

    with pytest.raises(ExactZeroError):
        score_kv_config(model, _tiny_corpus(tok), model_id=MODEL, method=_Bypass(bits=4))


def test_convert_prefix_drops_dequant_buffers_and_keeps_offset():
    """The trim(0) guarantee, observed at the boundary — segment 2 re-creates the buffers later."""
    _port_or_skip()
    from mlx_lm.models.cache import KVCache

    fp = KVCache()
    fp.update_and_fetch(
        mx.zeros((1, 8, 300, 64), mx.float16), mx.zeros((1, 8, 300, 64), mx.float16)
    )
    (new,) = TurboQuantKVMethod(bits=4).convert_prefix([fp])
    assert new.offset == 300
    assert new._k_deq_buf is None
    assert new._v_deq_buf is None
    assert stored_state_bytes([new]) == 36 * 2 * 8 * 300


def test_deployment_three_chunks_peak_under_cap(loaded_with_port):
    """Three 512-token deployment chunks: peak stays under the installed wired cap (no buffer leak)."""
    from mlx_quant_fidelity._memory_caps import compute_safe_caps_gb

    wired_gb, _ = compute_safe_caps_gb()
    if wired_gb == 0:
        pytest.skip("device reports no working-set size")
    model, tok = loaded_with_port
    corpus = _tiny_corpus(tok, chunk_length=512, n_chunks=3, repeats=200)
    mx.reset_peak_memory()
    rep = score_kv_config(
        model, corpus, model_id=MODEL, method=TurboQuantKVMethod(bits=4), quantize_start=100
    )
    assert rep.n_chunks == 3
    assert rep.peak_memory_bytes < wired_gb * 1024**3
