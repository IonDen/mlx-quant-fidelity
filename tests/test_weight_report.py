import dataclasses
import json

from mlx_quant_fidelity.corpora.provenance import CorpusProvenance
from mlx_quant_fidelity.metrics import ScalarSummary
from mlx_quant_fidelity.report import WeightFidelityReport, render_json, render_weight_markdown


def _report(*, reference_bits=None, warnings=("tok assumption",)) -> WeightFidelityReport:
    return WeightFidelityReport(
        quant_model_id="org/m-4bit",
        quant_revision=None,
        reference_model_id="org/m-bf16",
        reference_revision=None,
        quant_bits=4,
        quant_group_size=64,
        quant_mode="affine",
        per_layer=False,
        reference_bits=reference_bits,
        kl=ScalarSummary(0.06, 0.03, 0.4, 2.0),
        flip_rate=0.03,
        perplexity_ref=10.0,
        perplexity_quant=10.6,
        perplexity_delta=0.6,
        n_positions=1000,
        n_chunks=2,
        corpus=CorpusProvenance(
            "wikitext-2-raw", "test", "org/m-bf16", 512, 512, "none", "drop", "raw", 1024
        ),
        mlx_version="0.21",
        mlx_lm_version="0.31.3",
        peak_memory_bytes=18_000_000_000,
        quant_model_bytes=4_000_000_000,
        reference_model_bytes=14_000_000_000,
        verdict="marginal",
        warnings=warnings,
    )


def test_render_json_includes_weight_fields_and_warnings_list():
    data = json.loads(render_json(_report(reference_bits=8)))
    assert data["quant_bits"] == 4
    assert data["reference_bits"] == 8
    assert data["quant_model_bytes"] == 4_000_000_000
    assert isinstance(data["warnings"], list)  # tuple serializes as JSON list


def test_render_weight_markdown_qualifies_and_banners_quantized_reference():
    md = render_weight_markdown(_report(reference_bits=8))
    assert "wikitext-2-raw" in md  # corpus qualifier, never a bare score
    assert "512" in md
    assert "org/m-4bit" in md
    assert "org/m-bf16" in md
    assert "not full precision" in md  # reference-precision banner when reference_bits is set
    assert "tok assumption" in md  # warnings rendered


def test_render_weight_markdown_no_banner_for_full_precision_reference():
    md = render_weight_markdown(_report(reference_bits=None))
    assert "not full precision" not in md


def test_render_weight_markdown_handles_unknown_bits():
    md = render_weight_markdown(dataclasses.replace(_report(), quant_bits=None))
    assert "None-bit" not in md
    assert "unknown-bit" in md
