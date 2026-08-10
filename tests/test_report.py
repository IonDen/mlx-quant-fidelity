import dataclasses
import json

import pytest

from mlx_quant_fidelity.corpora.provenance import CorpusProvenance
from mlx_quant_fidelity.errors import ReportSchemaError
from mlx_quant_fidelity.metrics import DepthBucketSummary, ScalarSummary
from mlx_quant_fidelity.report import (
    FidelityReport,
    fidelity_report_from_dict,
    render_json,
    render_markdown,
    weight_report_from_dict,
)


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


def test_fidelity_from_dict_missing_kl_raises_report_schema_error():
    with pytest.raises(ReportSchemaError):
        fidelity_report_from_dict({"model_id": "m"})  # no "kl"/"corpus"


def test_fidelity_from_dict_malformed_kl_subdict_raises_report_schema_error():
    with pytest.raises(ReportSchemaError):
        fidelity_report_from_dict({"kl": {"mean": 0.1}, "corpus": {}})  # kl dict missing fields


def test_weight_from_dict_missing_kl_raises_report_schema_error():
    with pytest.raises(ReportSchemaError):
        weight_report_from_dict({"quant_model_id": "m"})


def test_render_markdown_deployment_states_post_boundary():
    from tests.test_cli import _fake_report

    rep = dataclasses.replace(_fake_report(), quantize_start=5, quantize_mode="deployment")
    md = render_markdown(rep)
    assert "post-boundary" in md or "excludes the first" in md  # the NEW exclusion statement


def test_fidelity_report_defaults_device_none():
    report = _report()
    assert report.device is None


def test_markdown_includes_device_when_set():
    report = dataclasses.replace(_report(), device="Apple M1 Max, 32 GB")
    assert "Apple M1 Max, 32 GB" in render_markdown(report)


def test_from_dict_accepts_missing_device():
    d = dataclasses.asdict(_report())
    del d["device"]
    assert fidelity_report_from_dict(d).device is None


def test_markdown_renders_depth_table():
    report = dataclasses.replace(
        _report(),
        kl_by_depth=(
            DepthBucketSummary(0, 2, 0.01, 0.02, 4),
            DepthBucketSummary(2, 4, 0.03, 0.05, 4),
        ),
    )
    md = render_markdown(report)
    assert "Drift by position depth" in md
    assert "| 0-1 |" in md


def test_from_dict_roundtrips_depth_buckets():
    buckets = (
        DepthBucketSummary(0, 2, 0.01, 0.02, 4),
        DepthBucketSummary(2, 4, 0.03, 0.05, 4),
    )
    report = dataclasses.replace(_report(), kl_by_depth=buckets)
    d = dataclasses.asdict(report)
    rehydrated = fidelity_report_from_dict(d)
    assert rehydrated.kl_by_depth == buckets


def test_from_dict_accepts_missing_depth():
    d = dataclasses.asdict(_report())
    d.pop("kl_by_depth", None)
    assert fidelity_report_from_dict(d).kl_by_depth is None


def test_from_dict_rejects_non_list_depth():
    d = dataclasses.asdict(_report())
    d["kl_by_depth"] = "boom"
    with pytest.raises(ReportSchemaError):
        fidelity_report_from_dict(d)
