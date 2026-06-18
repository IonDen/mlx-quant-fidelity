"""Compare orchestration: aggregate per-target results into a memory-normalized ranking."""

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mlx_quant_fidelity.policy import qualifies
from mlx_quant_fidelity.ranking import RankPoint, budget_pick, dominated_by, pareto_frontier
from mlx_quant_fidelity.report import (
    ComparisonReport,
    ComparisonTargetResult,
    weight_report_from_dict,
)

if TYPE_CHECKING:
    from mlx_quant_fidelity.corpora.provenance import CorpusProvenance


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


def _label_for_repo(repo: str) -> str:
    """The full repo id is the label — unique per distinct repo, unambiguous in the report."""
    return repo


def _partial_filename(repo: str) -> str:
    """Filesystem-safe partial JSON filename: '/' → '_', stays within artifacts_dir."""
    return repo.replace("/", "_") + ".json"


def _run_weight_target(
    quant: str, reference: str, partial_path: Path, max_chunks: int | None
) -> dict[str, object]:  # pragma: no cover - spawns a subprocess; covered by --run-slow
    """Spawn the weight worker for one target and return its parsed JSON envelope."""
    cmd = [
        sys.executable,
        "-m",
        "mlx_quant_fidelity.runners._worker",
        "--quant",
        quant,
        "--reference",
        reference,
        "--out",
        str(partial_path),
    ]
    if max_chunks is not None:
        cmd += ["--max-chunks", str(max_chunks)]
    subprocess.run(cmd, check=True)
    return json.loads(partial_path.read_text())  # type: ignore[no-any-return]


def _envelope_to_result(label: str, env: dict[str, object]) -> ComparisonTargetResult:
    if env.get("status") == "failed":
        # fix 3: absent keys yield None, not the string "None"
        return ComparisonTargetResult(
            label,
            "failed",
            None,
            None,
            None,
            env.get("error_type") or None,  # type: ignore[arg-type]
            env.get("message") or None,  # type: ignore[arg-type]
        )
    report = weight_report_from_dict(env["report"])  # type: ignore[arg-type]
    cost = report.quant_model_bytes
    if cost is None:
        return ComparisonTargetResult(label, "ok", report, None, "cost unavailable", None, None)
    return ComparisonTargetResult(
        label, "ok", report, RankPoint(label, report.kl.mean, cost), None, None, None
    )


def compare_weight_fidelity(
    quant_model_ids: list[str],
    reference_model_id: str,
    *,
    max_chunks: int | None = None,
    max_kld: float | None = None,
    min_tier: str | None = None,
    artifacts_dir: Path | None = None,
) -> ComparisonReport:
    """Rank N weight-quant repos vs one reference on quality-per-byte.

    Subprocess-per-target (each loads reference + quant); resumes by skipping targets whose
    partial JSON already exists. Mismatched/unrankable targets are isolated, not aborted.
    """
    if len(quant_model_ids) < 2:
        raise ValueError("compare needs at least 2 quant targets; use the `weights` probe for one.")
    labels = [_label_for_repo(r) for r in quant_model_ids]
    if len(set(labels)) != len(labels):
        duplicates = [lbl for lbl in labels if labels.count(lbl) > 1]
        raise ValueError(f"duplicate quant_model_ids produce the same label: {set(duplicates)}")
    # fix 5: reject malformed repo ids before any filesystem touch
    for repo in quant_model_ids:
        if "\x00" in repo:
            raise ValueError(f"repo id contains a NUL byte: {repo!r}")
        if len(_partial_filename(repo).encode()) > 255:
            raise ValueError(f"repo id {repo!r} produces a partial filename exceeding 255 bytes")
    # fix 2: filename-collision guard — distinct labels can still map to the same partial file
    filenames = [_partial_filename(r) for r in quant_model_ids]
    seen: dict[str, str] = {}
    for repo, fname in zip(quant_model_ids, filenames, strict=True):
        if fname in seen:
            raise ValueError(
                f"partial-filename collision: {repo!r} and {seen[fname]!r} both map to {fname!r}"
            )
        seen[fname] = repo
    out_dir = artifacts_dir or Path("_artifacts/compare/weight")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[ComparisonTargetResult] = []
    corpus = None
    for repo in quant_model_ids:
        label = _label_for_repo(repo)
        partial = out_dir / _partial_filename(repo)
        # fix 1: treat a corrupt/truncated partial as absent — fall through and re-run
        env: dict[str, object] | None = None
        if partial.exists():
            try:
                env = json.loads(partial.read_text())
            except (json.JSONDecodeError, OSError):
                env = None
        if env is None:
            env = _run_weight_target(
                repo, reference=reference_model_id, partial_path=partial, max_chunks=max_chunks
            )
        result = _envelope_to_result(label, env)
        # fix 4: corpus from the FIRST successful result (don't overwrite once set)
        if corpus is None and result.report is not None:
            corpus = result.report.corpus
        results.append(result)
    return assemble_comparison_report(
        results,
        mode="weight",
        reference=reference_model_id,
        model=None,
        corpus=corpus,
        quantize_start=None,
        quantize_mode=None,
        max_kld=max_kld,
        min_tier=min_tier,
        mlx_version=importlib.metadata.version("mlx"),
        mlx_lm_version=importlib.metadata.version("mlx-lm"),
    )
