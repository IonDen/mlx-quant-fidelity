"""Compare orchestration: aggregate per-target results into a memory-normalized ranking."""

from typing import TYPE_CHECKING

from mlx_quant_fidelity.policy import qualifies
from mlx_quant_fidelity.ranking import RankPoint, budget_pick, dominated_by, pareto_frontier
from mlx_quant_fidelity.report import ComparisonReport

if TYPE_CHECKING:
    from mlx_quant_fidelity.corpora.provenance import CorpusProvenance
    from mlx_quant_fidelity.report import ComparisonTargetResult


def _budget_label(max_kld: float | None, min_tier: str | None) -> str | None:
    parts = []
    if max_kld is not None:
        parts.append(f"--max-kld {max_kld}")
    if min_tier is not None:
        parts.append(f"--min-tier {min_tier}")
    return " ".join(parts) if parts else None


def assemble_comparison_report(
    results: "list[ComparisonTargetResult]",
    *,
    mode: str,
    reference: str | None,
    model: str | None,
    corpus: "CorpusProvenance | None",
    quantize_start: int | None,
    quantize_mode: str | None,
    max_kld: float | None,
    min_tier: str | None,
    mlx_version: str,
    mlx_lm_version: str,
) -> ComparisonReport:
    """Build a ComparisonReport: compute the Pareto frontier, dominated map, and budget pick.

    Only rankable results (status 'ok' with a `point`) enter the Pareto math; unrankable and
    failed results travel through unchanged, excluded from the frontier (spec audit #3/#5).
    """
    points: list[RankPoint] = [r.point for r in results if r.point is not None]
    frontier = pareto_frontier(points)
    dominated = dominated_by(points)
    qualifying = {
        r.label
        for r in results
        if r.report is not None
        and r.point is not None
        and qualifies(
            kl_mean=r.report.kl.mean, verdict=r.report.verdict, max_kld=max_kld, min_tier=min_tier
        )
    }
    budget = _budget_label(max_kld, min_tier)
    pick = budget_pick(points, qualifying=qualifying) if budget is not None else None
    return ComparisonReport(
        mode=mode,
        reference=reference,
        model=model,
        corpus=corpus,
        quantize_start=quantize_start,
        quantize_mode=quantize_mode,
        budget=budget,
        results=tuple(results),
        frontier=tuple(frontier),
        dominated=tuple(sorted(dominated.items())),
        budget_pick=pick,
        mlx_version=mlx_version,
        mlx_lm_version=mlx_lm_version,
    )
