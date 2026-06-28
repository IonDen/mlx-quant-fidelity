import dataclasses
import json

from mlx_quant_fidelity.corpora.provenance import CorpusProvenance
from mlx_quant_fidelity.metrics import ScalarSummary
from mlx_quant_fidelity.ranking import RankPoint
from mlx_quant_fidelity.report import (
    ComparisonReport,
    ComparisonTargetResult,
    FidelityReport,
    WeightFidelityReport,
    _human_bytes,
    fidelity_report_from_dict,
    render_comparison_json,
    render_comparison_markdown,
    weight_report_from_dict,
)


def _wreport(label: str, kl_mean: float, cost: int) -> WeightFidelityReport:
    return WeightFidelityReport(
        quant_model_id=label,
        quant_revision=None,
        reference_model_id="ref",
        reference_revision=None,
        quant_bits=4,
        quant_group_size=64,
        quant_mode="affine",
        per_layer=False,
        reference_bits=None,
        kl=ScalarSummary(kl_mean, kl_mean, kl_mean, kl_mean),
        flip_rate=0.02,
        perplexity_ref=10.0,
        perplexity_quant=10.1,
        perplexity_delta=0.1,
        n_positions=10,
        n_chunks=2,
        corpus=CorpusProvenance(
            "wikitext-2-raw", "test", "ref", 512, 512, "none", "drop", "raw", 10
        ),
        mlx_version="0.21",
        mlx_lm_version="0.31.3",
        peak_memory_bytes=1,
        quant_model_bytes=cost,
        reference_model_bytes=8000,
        verdict="good",
        warnings=(),
    )


def _freport() -> FidelityReport:
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


def _report() -> ComparisonReport:
    ok = ComparisonTargetResult(
        label="q4",
        status="ok",
        report=_wreport("q4", 0.09, 4200),
        point=RankPoint("q4", 0.09, 4200),
        excluded_reason=None,
        error_type=None,
        message=None,
    )
    unrankable = ComparisonTargetResult(
        label="q4-nobytes",
        status="ok",
        report=_wreport("q4-nobytes", 0.08, 0),
        point=None,
        excluded_reason="cost unavailable",
        error_type=None,
        message=None,
    )
    failed = ComparisonTargetResult(
        label="q2",
        status="failed",
        report=None,
        point=None,
        excluded_reason=None,
        error_type="ModelMismatchError",
        message="vocab_size mismatch",
    )
    return ComparisonReport(
        mode="weight",
        reference="ref",
        model=None,
        corpus=_wreport("q4", 0.09, 4200).corpus,
        quantize_start=None,
        quantize_mode=None,
        budget="--min-tier good",
        results=(ok, unrankable, failed),
        frontier=("q4",),
        dominated=(),
        budget_pick="q4",
        mlx_version="0.21",
        mlx_lm_version="0.31.3",
    )


def test_comparison_json_round_trips_status() -> None:
    out = json.loads(render_comparison_json(_report()))
    statuses = {r["label"]: r["status"] for r in out["results"]}
    assert statuses == {"q4": "ok", "q4-nobytes": "ok", "q2": "failed"}
    assert out["frontier"] == ["q4"]
    assert out["budget_pick"] == "q4"
    assert out["dominated"] == []  # fixture has dominated=()


def test_comparison_json_serializes_nested_report() -> None:
    # The status round-trip test only reads labels+status; a transform that dropped the nested
    # result["report"] payload would leave it green. Pin the nested report (and that a failed
    # target legitimately carries none).
    out = json.loads(render_comparison_json(_report()))
    q4 = next(r for r in out["results"] if r["label"] == "q4")
    assert q4["report"]["kl"]["mean"] == 0.09
    assert q4["report"]["verdict"] == "good"
    q2 = next(r for r in out["results"] if r["label"] == "q2")
    assert q2["report"] is None  # the failed target has no report


def test_comparison_markdown_renders_distinct_kl_mean_and_p99_columns() -> None:
    # _wreport sets all four KL fields equal, so no other test can tell the "KL mean" column from
    # the "KL p99" column. Pin them with distinct values: swapping the two columns (or printing
    # mean in both) in render_comparison_markdown goes red here.
    rep = dataclasses.replace(
        _wreport("q4", 0.05, 4200),
        kl=ScalarSummary(mean=0.0500, median=0.0300, p99=0.3000, max=0.8000),
    )
    result = ComparisonTargetResult(
        label="q4",
        status="ok",
        report=rep,
        point=RankPoint("q4", 0.05, 4200),
        excluded_reason=None,
        error_type=None,
        message=None,
    )
    report = ComparisonReport(
        mode="weight",
        reference="ref",
        model=None,
        corpus=None,
        quantize_start=None,
        quantize_mode=None,
        budget=None,
        results=(result,),
        frontier=("q4",),
        dominated=(),
        budget_pick=None,
        mlx_version="0.21",
        mlx_lm_version="0.31.3",
    )
    md = render_comparison_markdown(report)
    row = next(line for line in md.splitlines() if "| `q4` |" in line)
    # columns: target | cost | KL mean | KL p99 | flip | verdict | frontier
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    assert cells[2] == "0.0500"  # KL mean column
    assert cells[3] == "0.3000"  # KL p99 column — distinct from mean, pins the column order


def test_comparison_json_dominated_shape() -> None:
    """asdict serializes tuple-of-tuples as list-of-lists; dominated pair is preserved."""
    cheaper = ComparisonTargetResult(
        label="q4",
        status="ok",
        report=_wreport("q4", 0.05, 3000),
        point=RankPoint("q4", 0.05, 3000),
        excluded_reason=None,
        error_type=None,
        message=None,
    )
    pricier = ComparisonTargetResult(
        label="q8",
        status="ok",
        report=_wreport("q8", 0.01, 7000),
        point=RankPoint("q8", 0.01, 7000),
        excluded_reason=None,
        error_type=None,
        message=None,
    )
    report = ComparisonReport(
        mode="weight",
        reference="ref",
        model=None,
        corpus=None,
        quantize_start=None,
        quantize_mode=None,
        budget=None,
        results=(cheaper, pricier),
        frontier=("q4",),
        dominated=(("q8", "q4"),),
        budget_pick=None,
        mlx_version="0.21",
        mlx_lm_version="0.31.3",
    )
    out = json.loads(render_comparison_json(report))
    assert out["dominated"] == [["q8", "q4"]]


def test_comparison_markdown_shows_all_three_row_kinds() -> None:
    md = render_comparison_markdown(_report())
    assert "| `q4` |" in md  # uniquely targets the ranked table row body
    assert "✓" in md  # q4 is on the frontier in this fixture
    assert "vocab_size mismatch" in md  # failed row's message, not just error_type
    assert "cost unavailable" in md
    assert "ModelMismatchError" in md
    assert "Recommended" in md  # the budget-pick line


def test_weight_report_from_dict_round_trip() -> None:
    original = _wreport("q4", 0.09, 4200)
    rebuilt = weight_report_from_dict(dataclasses.asdict(original))
    assert rebuilt == original


def test_fidelity_report_from_dict_round_trip() -> None:
    original = _freport()
    rebuilt = fidelity_report_from_dict(dataclasses.asdict(original))
    assert rebuilt == original


def test_comparison_markdown_no_budget_pick_shows_no_target_message() -> None:
    """Covers the `elif budget is not None` branch (budget set, no qualifying pick)."""
    ok = ComparisonTargetResult(
        label="q4",
        status="ok",
        report=_wreport("q4", 0.09, 4200),
        point=RankPoint("q4", 0.09, 4200),
        excluded_reason=None,
        error_type=None,
        message=None,
    )
    report = ComparisonReport(
        mode="weight",
        reference="ref",
        model=None,
        corpus=None,
        quantize_start=None,
        quantize_mode=None,
        budget="--min-tier excellent",
        results=(ok,),
        frontier=("q4",),
        dominated=(),
        budget_pick=None,  # no pick
        mlx_version="0.21",
        mlx_lm_version="0.31.3",
    )
    md = render_comparison_markdown(report)
    assert "No target clears the budget" in md
    assert "--min-tier excellent" in md


def test_comparison_markdown_frontier_and_cost_order() -> None:
    """Both ✓ and ✗ appear; cheaper target's row precedes pricier target's row (cost-ascending)."""
    cheaper = ComparisonTargetResult(
        label="q4",
        status="ok",
        report=_wreport("q4", 0.05, 3000),
        point=RankPoint("q4", 0.05, 3000),
        excluded_reason=None,
        error_type=None,
        message=None,
    )
    pricier = ComparisonTargetResult(
        label="q8",
        status="ok",
        report=_wreport("q8", 0.01, 7000),
        point=RankPoint("q8", 0.01, 7000),
        excluded_reason=None,
        error_type=None,
        message=None,
    )
    report = ComparisonReport(
        mode="weight",
        reference="ref",
        model=None,
        corpus=None,
        quantize_start=None,
        quantize_mode=None,
        budget=None,
        results=(cheaper, pricier),
        frontier=("q4",),
        dominated=(("q8", "q4"),),
        budget_pick=None,
        mlx_version="0.21",
        mlx_lm_version="0.31.3",
    )
    md = render_comparison_markdown(report)
    assert "✓" in md  # cheaper is on frontier
    assert "✗" in md  # pricier is dominated
    # cost-ascending: cheaper row must precede pricier row; use exact prefixes to avoid
    # substring collisions (e.g. "q4" matching inside "q4-bad")
    assert md.index("| `q4` |") < md.index("| `q8` |")


def test_weight_report_from_dict_round_trip_non_empty_warnings() -> None:
    """Rehydrating a WeightFidelityReport with non-empty warnings preserves the tuple."""
    original = WeightFidelityReport(
        quant_model_id="q4",
        quant_revision=None,
        reference_model_id="ref",
        reference_revision=None,
        quant_bits=4,
        quant_group_size=64,
        quant_mode="affine",
        per_layer=False,
        reference_bits=None,
        kl=ScalarSummary(0.09, 0.09, 0.09, 0.09),
        flip_rate=0.02,
        perplexity_ref=10.0,
        perplexity_quant=10.1,
        perplexity_delta=0.1,
        n_positions=10,
        n_chunks=2,
        corpus=CorpusProvenance(
            "wikitext-2-raw", "test", "ref", 512, 512, "none", "drop", "raw", 10
        ),
        mlx_version="0.21",
        mlx_lm_version="0.31.3",
        peak_memory_bytes=1,
        quant_model_bytes=4200,
        reference_model_bytes=8000,
        verdict="good",
        warnings=("a warning",),
    )
    rebuilt = weight_report_from_dict(dataclasses.asdict(original))
    assert rebuilt == original


def test_comparison_markdown_budget_pick_without_budget_uses_fallback() -> None:
    """If budget_pick is set but budget=None, the recommendation line must not render 'None'."""
    ok = ComparisonTargetResult(
        label="q4",
        status="ok",
        report=_wreport("q4", 0.05, 3000),
        point=RankPoint("q4", 0.05, 3000),
        excluded_reason=None,
        error_type=None,
        message=None,
    )
    report = ComparisonReport(
        mode="weight",
        reference="ref",
        model=None,
        corpus=None,
        quantize_start=None,
        quantize_mode=None,
        budget=None,
        results=(ok,),
        frontier=("q4",),
        dominated=(),
        budget_pick="q4",  # pick set, but no budget description
        mlx_version="0.21",
        mlx_lm_version="0.31.3",
    )
    md = render_comparison_markdown(report)
    assert "clearing None" not in md
    assert "Recommended" in md
    assert "`q4`" in md


def test_comparison_markdown_kv_mode_omits_weight_footer() -> None:
    """Covers the `mode != 'weight'` branch — KV mode has no weight-reload footer."""
    ok = ComparisonTargetResult(
        label="kv4",
        status="ok",
        report=_wreport("kv4", 0.05, 0),
        point=RankPoint("kv4", 0.05, 0),
        excluded_reason=None,
        error_type=None,
        message=None,
    )
    report = ComparisonReport(
        mode="kv",
        reference=None,
        model="org/model",
        corpus=None,
        quantize_start=0,
        quantize_mode="stress",
        budget=None,
        results=(ok,),
        frontier=("kv4",),
        dominated=(),
        budget_pick=None,
        mlx_version="0.21",
        mlx_lm_version="0.31.3",
    )
    md = render_comparison_markdown(report)
    assert "Weight compare" not in md
    assert "kv" in md


# ── Fix C: dominated row renders dominator label ──────────────────────────────


def test_human_bytes_adaptive_formatting() -> None:
    """_human_bytes must scale adaptively so KV byte-per-token costs are not hidden as '0.00 GB'.

    Tested via render_comparison_markdown because _human_bytes is module-private.
    The cost column in the ranked table must reflect the correct magnitude unit.
    """
    # KV-scale cost: 576 bytes-per-token → must show "576 B", NOT "0.00 GB"
    kv_row = ComparisonTargetResult(
        label="kv4",
        status="ok",
        report=_wreport("kv4", 0.02, 576),
        point=RankPoint("kv4", 0.02, 576),
        excluded_reason=None,
        error_type=None,
        message=None,
    )
    # GB-scale cost (weight mode, ~4.2 GB) → must still show "GB"
    gb_row = ComparisonTargetResult(
        label="q4",
        status="ok",
        report=_wreport("q4", 0.05, 4_200_000_000),
        point=RankPoint("q4", 0.05, 4_200_000_000),
        excluded_reason=None,
        error_type=None,
        message=None,
    )
    # MB-scale cost → must show "MB"
    mb_row = ComparisonTargetResult(
        label="q2",
        status="ok",
        report=_wreport("q2", 0.20, 280_000_000),
        point=RankPoint("q2", 0.20, 280_000_000),
        excluded_reason=None,
        error_type=None,
        message=None,
    )
    report = ComparisonReport(
        mode="kv",
        reference=None,
        model="org/model",
        corpus=None,
        quantize_start=0,
        quantize_mode="stress",
        budget=None,
        results=(kv_row, gb_row, mb_row),
        frontier=("kv4", "q4", "q2"),
        dominated=(),
        budget_pick=None,
        mlx_version="0.21",
        mlx_lm_version="0.31.3",
    )
    md = render_comparison_markdown(report)

    # Locate each row line
    kv4_lines = [line for line in md.splitlines() if "| `kv4` |" in line]
    q4_lines = [line for line in md.splitlines() if "| `q4` |" in line]
    q2_lines = [line for line in md.splitlines() if "| `q2` |" in line]
    assert len(kv4_lines) == 1
    assert len(q4_lines) == 1
    assert len(q2_lines) == 1

    # Byte-scale must show "B" unit, not "GB"
    assert "576 B" in kv4_lines[0], f"expected '576 B' in kv4 row: {kv4_lines[0]!r}"
    assert "0.00 GB" not in kv4_lines[0], f"'0.00 GB' must not appear in kv4 row: {kv4_lines[0]!r}"

    # GB-scale must show "GB"
    assert "GB" in q4_lines[0], f"expected 'GB' in q4 row: {q4_lines[0]!r}"

    # MB-scale must show "MB"
    assert "MB" in q2_lines[0], f"expected 'MB' in q2 row: {q2_lines[0]!r}"

    # None → "—" (tested via existing None-branch; verify no regression)
    # (covered implicitly by the existing excluded tests, but assert directly on the symbol)


def test_human_bytes_all_branches() -> None:
    """_human_bytes covers every magnitude band including the None sentinel."""
    assert _human_bytes(None) == "—"
    assert _human_bytes(0) == "0 B"
    assert _human_bytes(999) == "999 B"
    assert _human_bytes(1_000) == "1.0 KB"
    assert _human_bytes(500_000) == "500.0 KB"
    assert _human_bytes(1_000_000) == "1.0 MB"
    assert _human_bytes(250_000_000) == "250.0 MB"
    assert _human_bytes(1_000_000_000) == "1.00 GB"
    assert _human_bytes(4_200_000_000) == "4.20 GB"


def test_comparison_markdown_dominated_row_shows_dominator_label() -> None:
    """A dominated (✗) row in the ranked table must also render the dominator's label.

    docs/ranking-principles.md specifies that dominated targets are flagged with a
    dominator's label.  The fixture has q8 dominated by q4 (q4 is cheaper AND better).
    The rendered markdown row for q8 must include the string 'q4' as the dominator.
    """
    cheaper_better = ComparisonTargetResult(
        label="q4",
        status="ok",
        report=_wreport("q4", 0.05, 3000),
        point=RankPoint("q4", 0.05, 3000),
        excluded_reason=None,
        error_type=None,
        message=None,
    )
    dominated = ComparisonTargetResult(
        label="q8",
        status="ok",
        report=_wreport("q8", 0.09, 7000),
        point=RankPoint("q8", 0.09, 7000),
        excluded_reason=None,
        error_type=None,
        message=None,
    )
    report = ComparisonReport(
        mode="weight",
        reference="ref",
        model=None,
        corpus=None,
        quantize_start=None,
        quantize_mode=None,
        budget=None,
        results=(cheaper_better, dominated),
        frontier=("q4",),
        dominated=(("q8", "q4"),),  # q8 is dominated by q4
        budget_pick=None,
        mlx_version="0.21",
        mlx_lm_version="0.31.3",
    )
    md = render_comparison_markdown(report)

    # q8 row must carry ✗ (dominated) and name its dominator 'q4'
    # Find the q8 row line
    q8_lines = [line for line in md.splitlines() if "| `q8` |" in line]
    assert len(q8_lines) == 1, f"expected exactly one q8 row, got: {q8_lines}"
    q8_row = q8_lines[0]
    assert "✗" in q8_row
    assert "q4" in q8_row, f"dominator label 'q4' not found in q8 row: {q8_row!r}"
