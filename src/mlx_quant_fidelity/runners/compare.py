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
from mlx_quant_fidelity.errors import CompareConfigError, QuantizeStartError, ReportSchemaError
from mlx_quant_fidelity.policy import VALID_VERDICTS, qualifies
from mlx_quant_fidelity.probes.kv import (
    MAX_CHUNK_LENGTH,
    _kv_head_dim,
    packed_width_mismatch,
    score_kv_config,
)
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

# Bump the relevant constant when that mode's partial format or cost formula changes, so only
# that mode's old partials are rejected. The two modes' partials are independent — a KV-only
# change (e.g. adding chunk_length to the identity) must not force weight partials to recompute.
_KV_PARTIAL_SCHEMA_VERSION = 2
_WEIGHT_PARTIAL_SCHEMA_VERSION = 1


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
    if mode == "kv":
        for r in results:
            if r.status == "ok" and r.report is not None:
                probe_mode = r.report.quantize_mode  # type: ignore[union-attr]
                assert probe_mode == quantize_mode, (
                    f"probe mode {probe_mode!r} disagrees with requested {quantize_mode!r}"
                )
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


def _validate_compare_weights_args(quant_model_ids: list[str]) -> None:
    """Validate weight-compare arguments. Raise CompareConfigError on bad input."""
    if len(quant_model_ids) < 2:
        raise CompareConfigError(
            "compare needs at least 2 quant targets; use the `weights` probe for one."
        )
    labels = [_label_for_repo(r) for r in quant_model_ids]
    if len(set(labels)) != len(labels):
        duplicates = [lbl for lbl in labels if labels.count(lbl) > 1]
        raise CompareConfigError(
            f"duplicate quant_model_ids produce the same label: {set(duplicates)}"
        )
    for repo in quant_model_ids:
        if "\x00" in repo:
            raise CompareConfigError(f"repo id contains a NUL byte: {repo!r}")
        if len(_partial_filename(repo).encode()) > 255:
            raise CompareConfigError(
                f"repo id {repo!r} produces a partial filename exceeding 255 bytes"
            )
    filenames = [_partial_filename(r) for r in quant_model_ids]
    seen: dict[str, str] = {}
    for repo, fname in zip(quant_model_ids, filenames, strict=True):
        if fname in seen:
            raise CompareConfigError(
                f"partial-filename collision: {repo!r} and {seen[fname]!r} both map to {fname!r}"
            )
        seen[fname] = repo


def _run_weight_target(
    quant: str, reference: str, partial_path: Path, max_chunks: int | None
) -> dict[str, object]:  # pragma: no cover - spawns a subprocess; covered by --run-slow
    """Spawn the weight worker for one target and return its parsed JSON envelope.

    If the worker exits non-zero or writes no parseable envelope, returns a failed envelope
    so the orchestrator can isolate the failure rather than aborting the whole compare run.
    """
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
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        stderr_hint = result.stderr.strip()
    except subprocess.CalledProcessError as exc:
        stderr_hint = (exc.stderr or "").strip()
        return {
            "status": "failed",
            "error_type": "WorkerError",
            "message": stderr_hint or f"worker exited with code {exc.returncode}",
        }
    raw = ""
    try:
        raw = partial_path.read_text()
        return json.loads(raw)  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError):
        return {
            "status": "failed",
            "error_type": "WorkerError",
            "message": stderr_hint
            or (
                f"worker wrote unreadable partial: {raw[:200]!r}"
                if raw
                else "worker wrote no partial"
            ),
        }


def _envelope_to_result(label: str, env: dict[str, object]) -> ComparisonTargetResult:
    # env is unvalidated json.loads output at the caller; the annotation is a lie mypy
    # believes but the runtime value may not honor — defend against a non-dict top level.
    if not isinstance(env, dict):
        return ComparisonTargetResult(  # type: ignore[unreachable]
            label,
            "failed",
            None,
            None,
            None,
            "CorruptPartial",
            f"partial for {label!r} is not a JSON object",
        )
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
    report_body = env.get("report")
    if not isinstance(report_body, dict):
        return ComparisonTargetResult(
            label,
            "failed",
            None,
            None,
            None,
            "CorruptPartial",
            f"partial for {label!r} has no report body",
        )
    try:
        report = weight_report_from_dict(report_body)
    except ReportSchemaError as exc:
        return ComparisonTargetResult(label, "failed", None, None, None, "CorruptPartial", str(exc))
    if report.verdict not in VALID_VERDICTS:
        return ComparisonTargetResult(
            label,
            "failed",
            None,
            None,
            None,
            "CorruptPartial",
            f"partial for {label!r} has an invalid verdict {report.verdict!r}",
        )
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

    Raises:
        CompareConfigError: If fewer than 2 targets, duplicate ids, malformed repo ids, or
            filename collisions. Subclasses ValueError for backward compatibility.
    """
    _validate_compare_weights_args(quant_model_ids)
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
            if not isinstance(env, dict):
                env = None  # valid JSON but not an object — treat as absent, recompute
        # Full run-identity guard: recompute if the partial is absent, has a non-ok status,
        # or its stored run_identity doesn't match the current call's full identity.
        if env is not None:
            expected_identity: dict[str, object] = {
                "mode": "weight",
                "quant": repo,
                "reference": reference_model_id,
                "max_chunks": max_chunks,
                "schema_version": _WEIGHT_PARTIAL_SCHEMA_VERSION,
            }
            if env.get("run_identity") != expected_identity:
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


_SWEEP_BITS: tuple[int, ...] = (2, 3, 4, 6, 8)
_SWEEP_GROUP_SIZES: tuple[int, ...] = (32, 64, 128)


def generate_sweep_configs(head_dim: int) -> tuple[list[tuple[int, int]], list[tuple[str, str]]]:
    """Auto-generate the (bits, group_size) grid for ``compare kv --sweep``.

    Group sizes are drawn from ``(32, 64, 128)``, filtered to those dividing ``head_dim``;
    bits are ``(2, 3, 4, 6, 8)``. A combination where :func:`packed_width_mismatch` is True
    (the upstream mlx-lm ``QuantizedKVCache`` pre-allocation bug — see ``probes/kv.py``) is
    routed to ``skipped`` with a reason instead of the grid, so a sweep never emits a failed
    row for a known-broken width.

    Raises:
        CompareConfigError: If no group size in ``(32, 64, 128)`` divides ``head_dim``.
    """
    group_sizes = [gs for gs in _SWEEP_GROUP_SIZES if head_dim % gs == 0]
    if not group_sizes:
        raise CompareConfigError(
            f"head_dim={head_dim} is not divisible by any of {list(_SWEEP_GROUP_SIZES)}; "
            "pass --configs explicitly with a group size that divides the model's KV head_dim."
        )
    grid: list[tuple[int, int]] = []
    skipped: list[tuple[str, str]] = []
    for bits in _SWEEP_BITS:
        for gs in group_sizes:
            label = _kv_config_label(bits, gs)
            if packed_width_mismatch(head_dim, bits):
                skipped.append(
                    (
                        label,
                        f"bits={bits} at head_dim={head_dim} hits an upstream mlx-lm "
                        "QuantizedKVCache packed-width bug (pre-allocation width disagrees "
                        "with mx.quantize's packed width); excluded from the sweep.",
                    )
                )
                continue
            grid.append((bits, gs))
    return grid, skipped


def kv_geometry_from_config(
    config: dict[str, object],
) -> tuple[int | None, int | None, int | None]:
    """Derive ``(n_layers, n_kv_heads, head_dim)`` from a raw HuggingFace ``config.json`` dict.

    Honors ``text_config`` nesting (multimodal configs carry the language-model geometry
    there). ``head_dim`` falls back to ``hidden_size // num_attention_heads`` when absent;
    ``num_key_value_heads`` falls back to ``num_attention_heads`` (dense attention). A field
    that can't be resolved to an int comes back ``None`` rather than raising — the caller
    decides what's fatal.
    """
    nested = config.get("text_config")
    cfg = nested if isinstance(nested, dict) else config

    def _int_or_none(v: object) -> int | None:
        return v if isinstance(v, int) else None

    n_layers = _int_or_none(cfg.get("num_hidden_layers"))
    n_heads = _int_or_none(cfg.get("num_attention_heads"))
    n_kv_heads = _int_or_none(cfg.get("num_key_value_heads"))
    if n_kv_heads is None:
        n_kv_heads = n_heads
    head_dim = _int_or_none(cfg.get("head_dim"))
    if head_dim is None:
        hidden_size = _int_or_none(cfg.get("hidden_size"))
        if hidden_size is not None and n_heads:
            head_dim = hidden_size // n_heads
    return n_layers, n_kv_heads, head_dim


def filter_configs_by_kv_budget(
    configs: list[tuple[int, int]],
    *,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    max_kv_bytes_per_token: int,
) -> tuple[list[tuple[int, int]], list[tuple[str, str]]]:
    """Partition ``configs`` into ``(kept, skipped)`` by a KV-byte-per-token budget.

    Costed via :func:`~mlx_quant_fidelity.costs.kv_bytes_per_token` — the same denominator
    ``compare kv`` ranks on.
    """
    kept: list[tuple[int, int]] = []
    skipped: list[tuple[str, str]] = []
    for bits, gs in configs:
        cost = kv_bytes_per_token(
            n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim, bits=bits, group_size=gs
        )
        label = _kv_config_label(bits, gs)
        if cost > max_kv_bytes_per_token:
            skipped.append(
                (
                    label,
                    f"{cost} B/token exceeds the --max-kv-bytes-per-token budget of "
                    f"{max_kv_bytes_per_token}",
                )
            )
        else:
            kept.append((bits, gs))
    return kept, skipped


def _validate_compare_kv_args(
    configs: list[tuple[int, int]],
    *,
    quantize_start: int,
    max_chunks: int | None,
    chunk_length: int,
) -> None:
    """Validate KV-compare arguments. Raise CompareConfigError on bad input."""
    if len(configs) < 2:
        raise CompareConfigError("compare needs at least 2 KV configs; use the `kv` probe for one.")
    if not (2 <= chunk_length <= MAX_CHUNK_LENGTH):
        raise CompareConfigError(
            f"chunk_length={chunk_length} must be between 2 and MAX_CHUNK_LENGTH="
            f"{MAX_CHUNK_LENGTH} (a hard safety ceiling on the per-chunk paired fp32 logits)."
        )
    if quantize_start != 0 and not (1 <= quantize_start <= chunk_length - 2):
        raise QuantizeStartError(
            f"quantize_start={quantize_start} must be 0 (stress) or in "
            f"[1, {chunk_length - 2}] (deployment)."
        )
    if max_chunks is not None and max_chunks < 1:
        raise CompareConfigError(f"max_chunks must be >= 1 (got {max_chunks}).")
    labels = [_kv_config_label(b, g) for b, g in configs]
    if len(set(labels)) != len(labels):
        duplicates = {lbl for lbl in labels if labels.count(lbl) > 1}
        raise CompareConfigError(f"duplicate configs produce the same label: {duplicates}")


def _kv_partial_filename(bits: int, group_size: int) -> str:
    """Filesystem-safe partial JSON filename: ':' sanitized to '_' (e.g. '4_64.json')."""
    return f"{bits}_{group_size}.json"


def _load_model(model_id: str, revision: str | None) -> tuple[object, object]:  # pragma: no cover
    """Load model weights from a HuggingFace repo. Returns (model, tokenizer).

    Both objects come from the same revision-pinned load call so the corpus tokenizer
    is guaranteed to match the loaded model weights.
    """
    from mlx_lm import load

    loaded = load(model_id, revision=revision)
    return loaded[0], loaded[1]


def _load_corpus_for_kv(
    tokenizer: object, model_id: str, max_chunks: int | None, *, chunk_length: int
) -> object:  # pragma: no cover
    """Build the WikiText-2 corpus for a KV-compare run.

    Takes the tokenizer directly from the caller (already loaded at the correct revision)
    rather than re-fetching it, so a revision-pinned run always uses the matching tokenizer.
    """
    from mlx_quant_fidelity.corpora.wikitext import load_wikitext2

    return load_wikitext2(
        tokenizer,  # type: ignore[arg-type]
        chunk_length=chunk_length,
        max_chunks=max_chunks,
        tokenizer_id=model_id,
    )


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
    # env is unvalidated json.loads output at the caller; the annotation is a lie mypy
    # believes but the runtime value may not honor — defend against a non-dict top level.
    if not isinstance(env, dict):
        return ComparisonTargetResult(  # type: ignore[unreachable]
            label,
            "failed",
            None,
            None,
            None,
            "CorruptPartial",
            f"partial for {label!r} is not a JSON object",
        )
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
    report_body = env.get("report")
    if not isinstance(report_body, dict):
        return ComparisonTargetResult(
            label,
            "failed",
            None,
            None,
            None,
            "CorruptPartial",
            f"partial for {label!r} has no report body",
        )
    try:
        report = fidelity_report_from_dict(report_body)
    except ReportSchemaError as exc:
        return ComparisonTargetResult(label, "failed", None, None, None, "CorruptPartial", str(exc))
    if report.verdict not in VALID_VERDICTS:
        return ComparisonTargetResult(
            label,
            "failed",
            None,
            None,
            None,
            "CorruptPartial",
            f"partial for {label!r} has an invalid verdict {report.verdict!r}",
        )
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
    chunk_length: int = 512,
    skipped_configs: list[tuple[str, str]] | None = None,
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
        quantize_start: 0 = stress mode (default); ``1 ≤ N ≤ chunk_length - 2`` = deployment
            mode (first N positions computed with a full-precision cache, then the stored prefix
            converts too; metrics cover the post-boundary region).
        max_chunks: Score at most this many corpus chunks (>= 1 if provided).
        max_kld: Optional KLD budget for the recommended pick.
        min_tier: Optional minimum tier for the recommended pick.
        artifacts_dir: Directory for partial JSON files (default: _artifacts/compare/kv).
        model_revision: HuggingFace model revision.
        chunk_length: Tokens per chunk for the auto-loaded corpus (default 512). Must be in
            ``[2, MAX_CHUNK_LENGTH]``; see :func:`~mlx_quant_fidelity.probes.kv.measure_kv_fidelity`.
        skipped_configs: ``(label, reason)`` pairs (e.g. from :func:`generate_sweep_configs` or
            :func:`filter_configs_by_kv_budget`) that were excluded before scoring — from a
            known-broken packed width or a KV-byte budget. Each becomes a ``"skipped"``
            :class:`~mlx_quant_fidelity.report.ComparisonTargetResult`, excluded from the
            frontier and listed under "Excluded (not ranked)" in the markdown render.

    Returns:
        A ComparisonReport with Pareto frontier, dominated map, and optional budget pick.

    Raises:
        QuantizeStartError: If quantize_start is out of range (not 0 and not in
            [1, chunk_length - 2]).
        CompareConfigError: If fewer than 2 configs, max_chunks < 1, duplicate configs, or
            chunk_length is out of range. Subclasses ValueError for backward compatibility.
    """
    # ── Validation guards (score_kv_config has none; must live here) ──────────
    _validate_compare_kv_args(
        configs, quantize_start=quantize_start, max_chunks=max_chunks, chunk_length=chunk_length
    )

    out_dir = artifacts_dir or Path("_artifacts/compare/kv")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine which configs need scoring (resume: skip valid partials) ────
    def _read_partial(bits: int, gs: int) -> dict[str, object] | None:
        """Return parsed envelope if the partial matches the current run's full identity.

        Returns None (recompute) when the partial is absent, corrupt/truncated, has a
        non-ok status, or its stored run_identity doesn't exactly match the expected
        identity for this config + current call arguments.
        """
        partial = out_dir / _kv_partial_filename(bits, gs)
        if not partial.exists():
            return None
        try:
            raw: dict[str, object] = json.loads(partial.read_text())
        except (json.JSONDecodeError, OSError):
            return None  # corrupt/truncated — treat as absent, recompute
        if not isinstance(raw, dict):
            return None  # type: ignore[unreachable]  # valid JSON but not an object — recompute
        if raw.get("status") != "ok":
            return None
        expected_identity: dict[str, object] = {
            "mode": "kv",
            "model_id": model_id,
            "model_revision": model_revision,
            "bits": bits,
            "group_size": gs,
            "quantize_start": quantize_start,
            "max_chunks": max_chunks,
            "chunk_length": chunk_length,
            "schema_version": _KV_PARTIAL_SCHEMA_VERSION,
        }
        if raw.get("run_identity") != expected_identity:
            return None
        return raw

    pending = [(b, g) for b, g in configs if _read_partial(b, g) is None]

    n_layers: int | None = None
    n_kv_heads: int | None = None
    head_dim: int | None = None

    if pending:
        install_memory_caps()
        model, tokenizer = _load_model(model_id, model_revision)
        n_layers, n_kv_heads, head_dim = _kv_dims(model)
        corpus = _load_corpus_for_kv(tokenizer, model_id, max_chunks, chunk_length=chunk_length)
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
                run_identity: dict[str, object] = {
                    "mode": "kv",
                    "model_id": model_id,
                    "model_revision": model_revision,
                    "bits": bits,
                    "group_size": gs,
                    "quantize_start": quantize_start,
                    "max_chunks": max_chunks,
                    "chunk_length": chunk_length,
                    "schema_version": _KV_PARTIAL_SCHEMA_VERSION,
                }
                envelope: dict[str, object] = {
                    "status": "ok",
                    "report": _asdict(fid_report),
                    "cost": cost,
                    "run_identity": run_identity,
                }
            except Exception as exc:  # any config failure is data, not abort
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
        except (json.JSONDecodeError, OSError, ValueError):
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

    for skip_label, skip_reason in skipped_configs or []:
        results.append(
            ComparisonTargetResult(skip_label, "skipped", None, None, skip_reason, None, None)
        )

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
