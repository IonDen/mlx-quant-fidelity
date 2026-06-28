import json

from mlx_quant_fidelity.corpora.provenance import CorpusProvenance
from mlx_quant_fidelity.metrics import ScalarSummary
from mlx_quant_fidelity.report import FidelityReport, render_json, render_markdown


def _report() -> FidelityReport:
    return FidelityReport(
        model_id="m",
        model_revision="abc123",
        kv_bits=4,
        kv_group_size=64,
        quantize_start=0,
        quantize_mode="stress",
        kl=ScalarSummary(0.02, 0.01, 0.2, 1.5),
        flip_rate=0.03,
        perplexity_ref=10.0,
        perplexity_quant=10.4,
        perplexity_delta=0.4,
        n_positions=1000,
        n_chunks=2,
        corpus=CorpusProvenance(
            "wikitext-2-raw", "test", "tok", 512, 512, "none", "drop", "raw", 1024
        ),
        mlx_version="0.21",
        mlx_lm_version="0.31.3",
        peak_memory_bytes=123,
        cache_supported=True,
        verdict="marginal",
        warnings=("bundles quantized-SDPA numerics",),
    )


def test_render_json_is_stable_and_complete():
    data = json.loads(render_json(_report()))
    assert data["kl"]["p99"] == 0.2
    assert data["verdict"] == "marginal"
    assert data["corpus"]["chunk_length"] == 512
    assert data["model_revision"] == "abc123"


def test_render_markdown_qualifies_by_corpus_and_length():
    md = render_markdown(_report())
    assert "wikitext-2-raw" in md  # never a bare fidelity number
    assert "512" in md
    assert "stress" in md
    assert "0.0300" in md  # flip rate is rendered (corpus-qualified, not a bare score)
    assert "bundles quantized-SDPA numerics" in md  # warnings are rendered
    assert "marginal" in md  # the verdict badge is the primary human-facing result
