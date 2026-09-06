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


def _mk_report(**overrides: object) -> FidelityReport:
    """`_report()` with field overrides applied via `dataclasses.replace` (a test helper)."""
    return dataclasses.replace(_report(), **overrides)  # type: ignore[arg-type]


def test_fidelity_report_defaults_not_partial():
    # RED until the partial-coverage fields exist: a normal (full-coverage) report is not partial
    # and carries no layer-coverage counts, so its committed headline stays unchanged.
    r = _report()
    assert r.kv_partial is False
    assert r.kv_layers_total is None
    assert r.kv_layers_quantized is None
    assert r.kv_layers_skipped is None


def test_from_dict_roundtrips_partial_fields():
    # RED until the fields exist + round-trip: a partial report survives asdict -> from_dict.
    partial = _mk_report(
        kv_partial=True,
        kv_layers_total=28,
        kv_layers_quantized=20,
        kv_layers_skipped={"RotatingKVCache": 8},
    )
    back = fidelity_report_from_dict(json.loads(render_json(partial)))
    assert back.kv_partial is True
    assert back.kv_layers_total == 28
    assert back.kv_layers_quantized == 20
    assert back.kv_layers_skipped == {"RotatingKVCache": 8}


def test_from_dict_defaults_partial_fields_for_legacy_dict():
    # backward-compat: a pre-0.9.0 dict (no partial keys) rehydrates as full-coverage.
    d = dataclasses.asdict(_report())
    for k in ("kv_partial", "kv_layers_total", "kv_layers_quantized", "kv_layers_skipped"):
        d.pop(k, None)
    back = fidelity_report_from_dict(d)
    assert back.kv_partial is False
    assert back.kv_layers_total is None


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("kv_layers_skipped", "boom"),
        ("kv_layers_total", "not-an-int"),
        ("kv_layers_quantized", []),
        ("kv_partial", "yes"),
    ],
)
def test_from_dict_rejects_malformed_partial_fields(field, bad):
    # RED until fidelity_report_from_dict type-guards the partial fields: a corrupted persisted
    # partial must be isolated as a schema error, not rehydrated into a broken report (mirrors
    # the control_kl isolation the surrounding compare machinery relies on).
    d = dataclasses.asdict(_report())
    d[field] = bad
    with pytest.raises(ReportSchemaError):
        fidelity_report_from_dict(d)


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


def test_from_dict_defaults_method_fields_for_legacy_json():
    d = json.loads(render_json(_report()))
    for key in (
        "kv_method",
        "kv_method_params",
        "kv_method_provenance",
        "measured_kv_bytes_per_token",
    ):
        d.pop(key, None)
    r = fidelity_report_from_dict(d)
    assert r.kv_method == "stock"
    assert r.kv_method_params == {}
    assert r.kv_method_provenance == {}
    assert r.measured_kv_bytes_per_token is None


def test_markdown_header_unchanged_for_stock_and_tagged_for_other_methods():
    base = _report()
    assert render_markdown(base).splitlines()[0] == (
        f"# KV-fidelity: `{base.model_id}` @ {base.kv_bits}-bit (group {base.kv_group_size})"
    )
    other = dataclasses.replace(base, kv_method="turboquant", kv_group_size=None)
    assert render_markdown(other).splitlines()[0].endswith("(group —) via turboquant")


def test_json_gains_exactly_four_keys_over_legacy():
    # Compare JSON to JSON (asdict keeps `warnings` as a tuple; JSON makes it a list).
    legacy = json.loads(render_json(_report()))
    for key in (
        "kv_method",
        "kv_method_params",
        "kv_method_provenance",
        "measured_kv_bytes_per_token",
    ):
        legacy.pop(key, None)
    now = json.loads(render_json(fidelity_report_from_dict(legacy)))
    assert set(now) - set(legacy) == {
        "kv_method",
        "kv_method_params",
        "kv_method_provenance",
        "measured_kv_bytes_per_token",
    }
    assert all(now[k] == legacy[k] for k in legacy)


def test_control_block_renders_and_roundtrips():
    """Reds if the control table is dropped or round-trip yields a dict, not ScalarSummary."""
    r = _mk_report(
        control_kl=ScalarSummary(mean=0.05, median=0.04, p99=0.1, max=0.2), control_flip_rate=0.02
    )
    md = render_markdown(r)
    assert "quantizer-only" in md
    assert "0.0500" in md
    back = fidelity_report_from_dict(json.loads(render_json(r)))
    assert isinstance(back.control_kl, ScalarSummary)
    assert back.control_kl == r.control_kl


def test_control_table_renders_dash_for_missing_flip_rate():
    """Reds if a control_kl-without-flip-rate report crashes the renderer instead of degrading."""
    r = _mk_report(control_kl=ScalarSummary(mean=0.05, median=0.04, p99=0.1, max=0.2))
    md = render_markdown(r)
    assert "quantizer-only" in md
    assert "—" in md


def test_no_control_no_new_markdown_lines():
    """Reds if a default stock report's markdown gains any new line (byte-identity guard)."""
    plain = render_markdown(_mk_report())
    assert "quantizer-only" not in plain
    assert "footing" not in plain


def test_method_bits_text_variants():
    """Reds if any of the four bits-text branches regresses (incl. the 'None-bit' bug)."""
    from mlx_quant_fidelity.report import method_bits_text

    assert method_bits_text(_mk_report()) == "4-bit"
    assert (
        method_bits_text(
            _mk_report(
                kv_bits=None,
                kv_method="affine",
                kv_method_params={"k_bits": 8, "v_bits": 4, "group_size": 64},
            )
        )
        == "k8v4-bit"
    )
    assert (
        method_bits_text(
            _mk_report(
                kv_bits=None,
                kv_method="turboquant-vonly",
                kv_method_params={"v_bits": 3, "seed": 42},
            )
        )
        == "v3-bit"
    )


def test_footing_line_renders_for_adapter_methods():
    """Reds if adapter reports lose the footing line or render 'None-bit' titles."""
    r = _mk_report(
        kv_bits=None,
        kv_group_size=None,
        drift_footing="quantizer_only",
        kv_method="affine",
        kv_method_params={"k_bits": 8, "v_bits": 4, "group_size": 64},
    )
    md = render_markdown(r)
    assert "k8v4-bit" in md
    assert "None-bit" not in md
    assert "quantizer_only" in md


def test_rehydrate_legacy_dict_defaults_new_fields():
    """Reds if legacy dicts stop defaulting cleanly, or a turboquant dict gets 'bundled'."""
    d = json.loads(render_json(_mk_report()))
    for key in ("drift_footing", "control_kl", "control_flip_rate", "working_set_bytes_per_token"):
        d.pop(key)
    back = fidelity_report_from_dict(d)
    assert back.drift_footing == "bundled"
    assert back.control_kl is None
    d["kv_method"] = "turboquant"
    assert fidelity_report_from_dict(d).drift_footing == "quantizer_only"


def test_control_kl_non_dict_raises_report_schema_error():
    """Reds if a malformed (non-dict, non-null) persisted control_kl is silently accepted."""
    d = json.loads(render_json(_mk_report()))
    d["control_kl"] = "boom"
    with pytest.raises(ReportSchemaError):
        fidelity_report_from_dict(d)
