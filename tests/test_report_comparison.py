"""Rehydrating a ComparisonReport from its JSON, and the standing method-not-recorded note."""

import json

import pytest
from tests.factories import make_fid_report, make_provenance, make_weight_report

from mlx_quant_fidelity.errors import ReportSchemaError
from mlx_quant_fidelity.metrics import DepthBucketSummary, ScalarSummary
from mlx_quant_fidelity.ranking import RankPoint
from mlx_quant_fidelity.report import (
    METHOD_NOT_RECORDED_WARNING,
    ComparisonReport,
    ComparisonTargetResult,
    comparison_report_from_dict,
    render_comparison_json,
    render_comparison_markdown,
    weight_report_from_dict,
)

# The 0.8.0 to 0.10.x wording, verbatim as committed samples and saved partials carry it.
OLD_SENTENCE = (
    "Quantization method is not recorded in the repo config — mlx-lm's DWQ, AWQ and dynamic "
    "quantizers write a quantization block that records only geometry (bits, group sizes, "
    "per-module overrides), never the recipe. This report describes the geometry of the loaded "
    "model, not the method that produced it."
)
NEW_SENTENCE = (
    "The recipe that produced this quantization is not recorded in a form this tool can check: "
    "mlx-lm's DWQ, AWQ and dynamic quantizers write a quantization block that records only "
    "geometry (bits, group sizes, per-module overrides), never the recipe. This report describes "
    "the geometry of the loaded model, not the method that produced it."
)


def _mixed_report() -> ComparisonReport:
    """A KV row, a weight row, a skipped row and a failed row, each field off its default."""
    kv = make_fid_report(
        model_id="org/kv-model",
        kv_method="turboquant",
        kv_method_params={"k_bits": 4, "v_bits": 4},
        kv_method_provenance={"package": "turboquant-mlx", "commit": "6e928d7"},
        kl_by_depth=(DepthBucketSummary(0, 256, 0.01, 0.1, 50),),
        control_kl=ScalarSummary(0.011, 0.005, 0.12, 0.9),
        control_flip_rate=0.004,
        drift_footing="quantizer_only",
        working_set_bytes_per_token=2048,
        warnings=("kv note",),
    )
    weight = make_weight_report(
        quant_model_id="org/m-4bit",
        quant_geometry=((4, 64, 168), (6, 64, 1)),
        quant_n_full_precision=2,
        quant_bits_per_weight=4.531,
        quant_precision="mixed",
        corpus=make_provenance(dataset_revision="b08601e0"),
        warnings=("tok", NEW_SENTENCE),
    )
    return ComparisonReport(
        mode="weight",
        reference="org/m-bf16",
        model=None,
        corpus=make_provenance(dataset_revision="b08601e0"),
        quantize_start=None,
        quantize_mode=None,
        budget="KL <= 0.1",
        results=(
            ComparisonTargetResult(
                "org/kv-model",
                "ok",
                kv,
                RankPoint("org/kv-model", 0.011, 1000),
                None,
                None,
                None,
                ranked_kl=0.011,
                ranked_verdict="good",
                ranked_footing="quantizer_only",
            ),
            ComparisonTargetResult(
                "org/m-4bit", "ok", weight, None, "cost unavailable", None, None
            ),
            ComparisonTargetResult.skipped("org/m-2bit", "over the KV budget"),
            ComparisonTargetResult.failed("org/m-3bit", "LoadError", "no weights"),
        ),
        frontier=("org/kv-model",),
        dominated=(("org/m-5bit", "org/kv-model"),),
        budget_pick="org/kv-model",
        mlx_version="0.31.2",
        mlx_lm_version="0.31.3",
    )


def test_comparison_report_round_trips_through_its_json():
    """Reds if rehydration drops a field, leaves `dominated` as lists, or rebuilds a row's
    report as the wrong type (a KV row read as a weight report, or the reverse)."""
    original = _mixed_report()
    rehydrated = comparison_report_from_dict(json.loads(render_comparison_json(original)))
    assert rehydrated == original


def test_rehydrated_report_renders_the_same_markdown():
    """Reds if a rehydrated comparison renders differently from the one it was saved from."""
    original = _mixed_report()
    rehydrated = comparison_report_from_dict(json.loads(render_comparison_json(original)))
    assert render_comparison_markdown(rehydrated) == render_comparison_markdown(original)


def _saved() -> dict[str, object]:
    return json.loads(render_comparison_json(_mixed_report()))


def _row(d: dict[str, object], i: int) -> dict[str, object]:
    return d["results"][i]  # type: ignore[index,no-any-return]


def _break_row_not_a_dict(d):
    d["results"][2] = ["org/m-2bit", "skipped"]


def _break_report_a_list(d):
    _row(d, 1)["report"] = [1, 2]


def _break_report_without_kl(d):
    del _row(d, 1)["report"]["kl"]


def _break_report_of_neither_kind(d):
    report = _row(d, 0)["report"]
    del report["model_id"]


def _break_point_without_cost(d):
    del _row(d, 0)["point"]["cost_bytes"]


def _break_unknown_status(d):
    _row(d, 3)["status"] = "maybe"


def _break_row_without_label(d):
    del _row(d, 3)["label"]


def _break_results_not_a_list(d):
    d["results"] = {"org/m-4bit": {}}


def _break_dominated_pair(d):
    d["dominated"] = [["org/m-5bit"]]


@pytest.mark.parametrize(
    "corrupt",
    [
        _break_row_not_a_dict,
        _break_report_a_list,
        _break_report_without_kl,
        _break_report_of_neither_kind,
        _break_point_without_cost,
        _break_unknown_status,
        _break_row_without_label,
        _break_results_not_a_list,
        _break_dominated_pair,
    ],
    ids=lambda f: f.__name__.removeprefix("_break_"),
)
def test_malformed_comparison_raises_report_schema_error(corrupt):
    """Reds if a corrupt saved comparison escapes as a bare KeyError/TypeError/AttributeError
    (or, worse, rehydrates) instead of the package's ReportSchemaError."""
    d = _saved()
    corrupt(d)
    with pytest.raises(ReportSchemaError):
        comparison_report_from_dict(d)


def test_method_note_says_the_recipe_is_not_recorded():
    """Reds if the note goes back to claiming the method is absent from the repo config, which
    is false for an imported AWQ repo whose `quantization_config.quant_method` survives."""
    assert METHOD_NOT_RECORDED_WARNING == NEW_SENTENCE
    assert "repo config" not in METHOD_NOT_RECORDED_WARNING


def test_resumed_comparison_shows_the_method_note_once_in_its_new_wording():
    """Bug: a resumed comparison shows both the old and new method-not-recorded sentences.

    A saved partial written before the rewording carries the old sentence; the resume path
    rehydrates it through `weight_report_from_dict`, and the renderer then re-adds the standing
    note because the new sentence is missing. Exactly one copy, the new one, must come out.
    """
    saved = json.loads(
        json.dumps(
            {
                **json.loads(render_comparison_json(_mixed_report()))["results"][1]["report"],
                "warnings": ["tok", OLD_SENTENCE],
            }
        )
    )
    a = weight_report_from_dict(saved)
    b = weight_report_from_dict({**saved, "quant_model_id": "org/m-8bit"})
    assert a.warnings == ("tok", NEW_SENTENCE)
    report = ComparisonReport(
        mode="weight",
        reference="org/m-bf16",
        model=None,
        corpus=None,
        quantize_start=None,
        quantize_mode=None,
        budget=None,
        results=(
            ComparisonTargetResult(
                "org/m-4bit", "ok", a, RankPoint("org/m-4bit", 0.06, 1), None, None, None
            ),
            ComparisonTargetResult(
                "org/m-8bit", "ok", b, RankPoint("org/m-8bit", 0.06, 2), None, None, None
            ),
        ),
        frontier=("org/m-4bit",),
        dominated=(("org/m-8bit", "org/m-4bit"),),
        budget_pick=None,
        mlx_version="0.31.2",
        mlx_lm_version="0.31.3",
    )
    md = render_comparison_markdown(report)
    assert md.count(f"> Note: {NEW_SENTENCE}") == 1
    assert OLD_SENTENCE not in md
    assert OLD_SENTENCE not in render_comparison_json(report)
