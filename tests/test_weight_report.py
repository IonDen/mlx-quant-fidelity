import dataclasses
import json

import pytest

from mlx_quant_fidelity.corpora.provenance import CorpusProvenance
from mlx_quant_fidelity.metrics import ScalarSummary
from mlx_quant_fidelity.report import (
    WeightFidelityReport,
    render_json,
    render_weight_markdown,
    weight_bits_text,
    weight_report_from_dict,
)


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
    assert "marginal" in md  # the verdict badge is the primary human-facing result


def test_render_weight_markdown_no_banner_for_full_precision_reference():
    md = render_weight_markdown(_report(reference_bits=None))
    assert "not full precision" not in md


def test_render_weight_markdown_handles_unknown_bits():
    md = render_weight_markdown(dataclasses.replace(_report(), quant_bits=None))
    assert "None-bit" not in md
    assert "unknown-bit" in md


def test_weight_fidelity_report_defaults_device_none():
    report = _report()
    assert report.device is None


def test_weight_markdown_includes_device_when_set():
    report = dataclasses.replace(_report(), device="Apple M1 Max, 32 GB")
    assert "Apple M1 Max, 32 GB" in render_weight_markdown(report)


def test_weight_from_dict_accepts_missing_device():
    d = dataclasses.asdict(_report())
    del d["device"]
    assert weight_report_from_dict(d).device is None


def _measured(**overrides) -> WeightFidelityReport:
    base = _report()
    fields = {
        "quant_geometry": ((4, 64, 197),),
        "quant_n_full_precision": 0,
        "quant_bits_per_weight": 4.501,
        "quant_precision": "uniform",
    }
    fields.update(overrides)
    return dataclasses.replace(base, **fields)


def test_legacy_report_defaults_new_fields_to_none_and_keeps_the_old_headline():
    """Reds if a legacy report (no bits/weight) picks up the new headline."""
    r = _report()
    assert (
        r.quant_geometry,
        r.quant_n_full_precision,
        r.quant_bits_per_weight,
        r.quant_precision,
    ) == (None, None, None, None)
    assert render_weight_markdown(r).splitlines()[0] == (
        "# Weight-fidelity: `org/m-4bit` @ 4-bit (group 64) vs `org/m-bf16`"
    )


def test_uniform_headline_carries_group_and_bits_per_weight():
    """Reds if nominal and measured are swapped or bits/weight is not two decimals."""
    assert render_weight_markdown(_measured()).splitlines()[0] == (
        "# Weight-fidelity: `org/m-4bit` @ 4-bit (group 64, 4.50 bits/weight) vs `org/m-bf16`"
    )


def test_uniform_with_two_group_sizes_prints_both():
    """Reds if the group-size set collapses to the first (or the nominal) group size."""
    r = _measured(quant_geometry=((4, 32, 1), (4, 64, 196)), quant_bits_per_weight=4.632)
    assert "@ 4-bit (group 32/64, 4.63 bits/weight) vs" in render_weight_markdown(r).splitlines()[0]


def test_mixed_headline_lists_modules_per_bit_width():
    """Reds if a mixed report renders as uniform or the per-bits counts are lost."""
    r = _measured(
        quant_geometry=((4, 64, 53), (5, 64, 144)),
        quant_bits_per_weight=5.02,
        quant_precision="mixed",
    )
    assert (
        "@ mixed 4/5-bit (group 64, 5.02 bits/weight; 53 modules at 4-bit, 144 at 5-bit) vs"
        in (render_weight_markdown(r).splitlines()[0])
    )


def test_headline_reports_quantizable_modules_left_at_full_precision():
    """Reds on a singular/plural slip or if the count is dropped."""
    r = _measured(quant_n_full_precision=1)
    assert (
        "; 1 quantizable module left at full precision) vs"
        in render_weight_markdown(r).splitlines()[0]
    )
    r2 = _measured(quant_n_full_precision=2)
    assert (
        "; 2 quantizable modules left at full precision) vs"
        in render_weight_markdown(r2).splitlines()[0]
    )


def test_headline_with_no_quantized_modules_says_so():
    """Unreachable through the gate today (the config declares quantization) but reachable if a
    config lies; reds if an empty geometry renders as a bare uniform headline."""
    r = _measured(quant_geometry=())
    assert (
        "@ 4-bit (group 64, 4.50 bits/weight; 0 quantized modules) vs"
        in render_weight_markdown(r).splitlines()[0]
    )


def test_headline_with_no_group_size_at_all_says_unknown():
    """Reds if an empty group-size set renders as a dangling `group , 4.50 bits/weight` —
    a measured report whose geometry is empty and whose config declared no group size."""
    r = _measured(quant_geometry=(), quant_group_size=None)
    assert (
        "@ 4-bit (group ?, 4.50 bits/weight; 0 quantized modules) vs"
        in render_weight_markdown(r).splitlines()[0]
    )


def test_mixed_headline_uses_the_singular_noun_for_a_single_module():
    """Reds on the plural slip: one module at the lowest width must read `1 module at 4-bit`,
    the same singular/plural rule the full-precision clause already applies."""
    r = _measured(
        quant_geometry=((4, 64, 1), (5, 64, 196)),
        quant_bits_per_weight=5.0,
        quant_precision="mixed",
    )
    assert (
        "@ mixed 4/5-bit (group 64, 5.00 bits/weight; 1 module at 4-bit, 196 at 5-bit) vs"
        in render_weight_markdown(r).splitlines()[0]
    )


def test_legacy_report_with_geometry_uses_the_nominal_everywhere():
    """One legacy predicate: a report is legacy iff it has no bits/weight. Reds if the badge
    keys off geometry instead — a foreign partial carrying geometry but no bits/weight would
    get a `mixed 4/5-bit` badge beside a legacy `4-bit (group 64)` headline."""
    r = dataclasses.replace(_report(), quant_geometry=((4, 64, 3), (5, 64, 1)))
    assert weight_bits_text(r) == "4-bit"
    assert render_weight_markdown(r).splitlines()[0] == (
        "# Weight-fidelity: `org/m-4bit` @ 4-bit (group 64) vs `org/m-bf16`"
    )


def test_weight_from_dict_round_trips_geometry_as_tuples():
    d = json.loads(render_json(_measured(quant_geometry=((4, 32, 1), (4, 64, 196)))))
    back = weight_report_from_dict(d)
    assert back.quant_geometry == ((4, 32, 1), (4, 64, 196))
    assert isinstance(back.quant_geometry[0], tuple)


def test_weight_from_dict_accepts_the_asdict_tuple_form():
    """`dataclasses.asdict` keeps tuples; reds if the validator accepts lists only."""
    original = _measured(quant_geometry=((4, 32, 1), (4, 64, 196)))
    assert weight_report_from_dict(dataclasses.asdict(original)) == original


def test_weight_from_dict_legacy_dict_yields_none_fields():
    d = json.loads(render_json(_report()))
    for key in (
        "quant_geometry",
        "quant_n_full_precision",
        "quant_bits_per_weight",
        "quant_precision",
    ):
        d.pop(key, None)
    back = weight_report_from_dict(d)
    assert back.quant_geometry is None
    assert back.quant_bits_per_weight is None


@pytest.mark.parametrize(
    "bad", [[[4, 64]], [["4", 64, 1]], [[True, 64, 1]], "4-64-1", [[4, 64, 1, 9]]]
)
def test_weight_from_dict_rejects_malformed_geometry(bad):
    """Reds if geometry is passed through unvalidated (a frozen dataclass checks nothing)."""
    from mlx_quant_fidelity.errors import ReportSchemaError

    d = json.loads(render_json(_measured()))
    d["quant_geometry"] = bad
    with pytest.raises(ReportSchemaError, match="quant_geometry"):
        weight_report_from_dict(d)
