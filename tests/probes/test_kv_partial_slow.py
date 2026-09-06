"""Real-model confirmation for per-layer partial KV coverage (0053).

``@pytest.mark.slow``; run with ``--run-slow`` on the main thread (needs a model download). Skips
if the model is not cached. Uses ``mlx-community/gemma-3-1b-it-4bit`` — a genuine sliding/full
hybrid (``gemma3_text``, ``sliding_window_pattern=6`` over 26 layers), so ``make_prompt_cache``
returns a mix of ``KVCache`` (quantizable) and ``RotatingKVCache`` (skipped). Confirms the probe
measures the quantizable layers, reports the partial coverage, and carries the hypothetical-partial
note — the end-to-end behaviour the offline fake-forward test cannot prove.

Recorded (main-thread run 2026-09-06, M1 Max 32 GB, mlx 0.31.2, mlx-lm 0.31.3, chunk_length=64,
3 chunks): quantized 4/26 layers (22 RotatingKVCache skipped), KL mean=0.0553, peak=1.37 GB.
"""

import pytest
from tests.probes.test_kv_oracles import _tiny_corpus

from mlx_quant_fidelity.probes.kv import score_kv_config
from mlx_quant_fidelity.probes.kv_methods import StockKVMethod

pytestmark = pytest.mark.slow

HYBRID_MODEL = "mlx-community/gemma-3-1b-it-4bit"


@pytest.fixture(scope="module")
def loaded_hybrid():
    from mlx_lm import load

    return load(HYBRID_MODEL)


def test_partial_coverage_on_real_hybrid(loaded_hybrid):
    # Confirms 0053 on a real sliding/full hybrid: the full-attention layers quantize, the sliding
    # layers stay full-precision, and the probe returns a partial report with non-zero drift.
    # RED would be: the mixed [QuantizedKVCache, RotatingKVCache, ...] forward crashes, or the
    # report is not flagged partial, or drift is exactly zero (the quantized layers never engaged).
    model, tok = loaded_hybrid
    corpus = _tiny_corpus(tok, chunk_length=64, n_chunks=3)
    report = score_kv_config(
        model, corpus, model_id=HYBRID_MODEL, method=StockKVMethod(bits=4, group_size=64)
    )
    print(
        f"partial={report.kv_partial} "
        f"quantized={report.kv_layers_quantized}/{report.kv_layers_total} "
        f"skipped={report.kv_layers_skipped} kl_mean={report.kl.mean:.4f} "
        f"peak={report.peak_memory_bytes / 1e9:.2f}GB"
    )
    assert report.kv_partial is True
    assert report.cache_supported is True
    # gemma3_text: 26 layers, 1 full-attention every 6th => 4 quantizable, 22 sliding skipped.
    assert report.kv_layers_total == 26
    assert report.kv_layers_quantized == 4
    assert report.kv_layers_skipped == {"RotatingKVCache": 22}
    assert report.kl.mean > 0  # the quantized full-attention layers engaged
    assert any("hypothetical partial" in w for w in report.warnings)
