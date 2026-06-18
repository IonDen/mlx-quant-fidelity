"""Compare orchestration: aggregate per-target results into a memory-normalized ranking."""

import dataclasses as _dc
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx

from mlx_quant_fidelity._memory_caps import install_memory_caps
from mlx_quant_fidelity.costs import kv_bytes_per_token
from mlx_quant_fidelity.errors import QuantFidelityError
from mlx_quant_fidelity.policy import qualifies
from mlx_quant_fidelity.probes.kv import _kv_head_dim, score_kv_config
from mlx_quant_fidelity.ranking import RankPoint, budget_pick, dominated_by, pareto_frontier
from mlx_quant_fidelity.report import (
    ComparisonReport,
    ComparisonTargetResult,
    fidelity_report_from_dict,
    weight_report_from_dict,
)

if TYPE_CHECKING:
    from mlx_quant_fidelity.corpora.provenance import CorpusProvenance

_asdict = _dc.asdict


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


# ── KV compare: one model load, loop configs in-process ───────────────────────


def _kv_config_label(bits: int, group_size: int) -> str:
    """Human-readable config label used in the report (e.g. '4:64')."""
    return f"{bits}:{group_size}"


def _kv_partial_filename(bits: int, group_size: int) -> str:
    """Filesystem-safe partial JSON filename: ':' sanitized to '_' (e.g. '4_64.json')."""
    return f"{bits}_{group_size}.json"


def _load_model(model_id: str, revision: str | None) -> object:  # pragma: no cover
    """Load model weights from a HuggingFace repo. Returns the model object only."""
    from mlx_lm import load

    return load(model_id, revision=revision)[0]


def _load_corpus_for_kv(
    model: object, model_id: str, max_chunks: int | None
) -> object:  # pragma: no cover
    """Build the WikiText-2 corpus for a KV-compare run.

    Uses mlx_lm's load_tokenizer so the model weights do not need to be reloaded
    just to get the tokenizer — the model is already in memory.
    """
    from mlx_lm.utils import load_tokenizer

    from mlx_quant_fidelity.corpora.wikitext import load_wikitext2

    tokenizer = load_tokenizer(model_id)  # type: ignore[no-untyped-call]
    return load_wikitext2(tokenizer, max_chunks=max_chunks, tokenizer_id=model_id)


def _kv_dims(model: object) -> tuple[int | None, int | None, int | None]:
    """Return (n_layers, n_kv_heads, head_dim) from a loaded model; any may be None."""
    from mlx_lm.models.cache import make_prompt_cache

    args = getattr(model, "args", None)
    n_layers = len(make_prompt_cache(model))
    _nkv = getattr(args, "num_key_value_heads", None)
    n_kv_heads = _nkv if _nkv is not None else getattr(args, "num_attention_heads", None)
    head_dim = _kv_head_dim(model)
    return n_layers, n_kv_heads, head_dim


def _kv_envelope_to_result(label: str, env: dict[str, object]) -> ComparisonTargetResult:
    """Convert a stored KV partial envelope to a ComparisonTargetResult."""
    if env.get("status") == "failed":
        return ComparisonTargetResult(
            label,
            "failed",
            None,
            None,
            None,
            env.get("error_type") or None,  # type: ignore[arg-type]
            env.get("message") or None,  # type: ignore[arg-type]
        )
    report = fidelity_report_from_dict(env["report"])  # type: ignore[arg-type]
    cost = env.get("cost")
    if cost is None:
        return ComparisonTargetResult(label, "ok", report, None, "cost unavailable", None, None)
    if not isinstance(cost, (int, float)):
        raise ValueError(f"unexpected cost type in partial: {type(cost)!r}")
    return ComparisonTargetResult(
        label,
        "ok",
        report,
        RankPoint(label, report.kl.mean, int(cost)),
        None,
        None,
        None,
    )


def compare_kv_fidelity(
    model_id: str,
    configs: list[tuple[int, int]],
    *,
    quantize_start: int = 0,
    max_chunks: int | None = None,
    max_kld: float | None = None,
    min_tier: str | None = None,
    artifacts_dir: Path | None = None,
    model_revision: str | None = None,
) -> ComparisonReport:
    """Rank N (bits, group_size) KV-cache configs on one model, quality-per-KV-byte-per-token.

    Loads the model ONCE and loops configs via score_kv_config (one model resident — that's
    the whole point vs weight compare which spawns per target). Writes a partial JSON per
    config and resumes by skipping configs whose valid partial already exists.

    Unsupported configs (CacheNotQuantizableError or any QuantFidelityError) are isolated
    as 'failed' results and excluded from the frontier; the run continues.

    Args:
        model_id: HuggingFace model ID.
        configs: List of (bits, group_size) tuples; must contain at least 2 distinct entries.
        quantize_start: Must be 0 (stress mode only in 0.x).
        max_chunks: Score at most this many corpus chunks (>= 1 if provided).
        max_kld: Optional KLD budget for the recommended pick.
        min_tier: Optional minimum tier for the recommended pick.
        artifacts_dir: Directory for partial JSON files (default: _artifacts/compare/kv).
        model_revision: HuggingFace model revision.

    Returns:
        A ComparisonReport with Pareto frontier, dominated map, and optional budget pick.

    Raises:
        ValueError: If fewer than 2 configs, quantize_start != 0, max_chunks < 1, or duplicates.
    """
    # ── Validation guards (score_kv_config has none; must live here) ──────────
    if len(configs) < 2:
        raise ValueError("compare needs at least 2 KV configs; use the `kv` probe for one.")
    if quantize_start != 0:
        raise ValueError(
            "deployment mode (quantize_start > 0) is not supported in 0.x; "
            "use stress mode (quantize_start=0) only."
        )
    if max_chunks is not None and max_chunks < 1:
        raise ValueError(f"max_chunks must be >= 1 (got {max_chunks}).")
    labels = [_kv_config_label(b, g) for b, g in configs]
    if len(set(labels)) != len(labels):
        duplicates = {lbl for lbl in labels if labels.count(lbl) > 1}
        raise ValueError(f"duplicate configs produce the same label: {duplicates}")

    out_dir = artifacts_dir or Path("_artifacts/compare/kv")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine which configs need scoring (resume: skip valid partials) ────
    def _read_partial(bits: int, gs: int) -> dict[str, object] | None:
        """Return parsed envelope if the partial is valid, else None (recompute)."""
        partial = out_dir / _kv_partial_filename(bits, gs)
        if not partial.exists():
            return None
        try:
            raw: dict[str, object] = json.loads(partial.read_text())
            return raw
        except (json.JSONDecodeError, OSError):
            return None  # corrupt/truncated — treat as absent, recompute

    pending = [(b, g) for b, g in configs if _read_partial(b, g) is None]

    n_layers: int | None = None
    n_kv_heads: int | None = None
    head_dim: int | None = None

    if pending:
        install_memory_caps()
        model = _load_model(model_id, model_revision)
        n_layers, n_kv_heads, head_dim = _kv_dims(model)
        corpus = _load_corpus_for_kv(model, model_id, max_chunks)
        for bits, gs in pending:
            mx.reset_peak_memory()
            partial = out_dir / _kv_partial_filename(bits, gs)
            try:
                fid_report = score_kv_config(
                    model,
                    corpus,  # type: ignore[arg-type]  # monkeypatched to object in tests
                    model_id=model_id,
                    model_revision=model_revision,
                    kv_bits=bits,
                    kv_group_size=gs,
                    quantize_start=quantize_start,
                    max_chunks=max_chunks,
                )
                cost: int | None
                if n_layers is not None and n_kv_heads is not None and head_dim is not None:
                    cost = kv_bytes_per_token(
                        n_layers=n_layers,
                        n_kv_heads=n_kv_heads,
                        head_dim=head_dim,
                        bits=bits,
                        group_size=gs,
                    )
                else:
                    cost = None
                envelope: dict[str, object] = {
                    "status": "ok",
                    "report": _asdict(fid_report),
                    "cost": cost,
                }
            except QuantFidelityError as exc:
                envelope = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            partial.write_text(json.dumps(envelope))
            mx.clear_cache()

    # ── Collect results (resumed or just-written) ──────────────────────────────
    results: list[ComparisonTargetResult] = []
    corpus_prov: CorpusProvenance | None = None
    for bits, gs in configs:
        label = _kv_config_label(bits, gs)
        try:
            env: dict[str, object] = json.loads(
                (out_dir / _kv_partial_filename(bits, gs)).read_text()
            )
            result = _kv_envelope_to_result(label, env)
        except (json.JSONDecodeError, OSError):
            result = ComparisonTargetResult(
                label,
                "failed",
                None,
                None,
                None,
                "CorruptPartial",
                f"partial for config {label!r} was corrupt or unreadable at collect time",
            )
        if corpus_prov is None and result.report is not None:
            corpus_prov = result.report.corpus
        results.append(result)

    return assemble_comparison_report(
        results,
        mode="kv",
        reference=None,
        model=model_id,
        corpus=corpus_prov,
        quantize_start=quantize_start,
        quantize_mode="stress" if quantize_start == 0 else "deployment",
        max_kld=max_kld,
        min_tier=min_tier,
        mlx_version=importlib.metadata.version("mlx"),
        mlx_lm_version=importlib.metadata.version("mlx-lm"),
    )
