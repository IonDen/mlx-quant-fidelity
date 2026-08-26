"""FidelityReport (frozen) + JSON / Markdown renderers. Consumes policy's verdict."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from mlx_quant_fidelity.corpora.provenance import CorpusProvenance
from mlx_quant_fidelity.errors import ReportSchemaError
from mlx_quant_fidelity.metrics import DepthBucketSummary, ScalarSummary

if TYPE_CHECKING:
    from mlx_quant_fidelity.ranking import RankPoint


@dataclass(frozen=True, slots=True)
class FidelityReport:
    """The complete result of a KV-fidelity measurement. Frozen; stable to diff."""

    model_id: str
    model_revision: str | None
    kv_bits: int | None
    kv_group_size: int | None
    quantize_start: int
    quantize_mode: str
    kl: ScalarSummary
    flip_rate: float
    perplexity_ref: float
    perplexity_quant: float
    perplexity_delta: float
    n_positions: int
    n_chunks: int
    corpus: CorpusProvenance
    mlx_version: str
    mlx_lm_version: str
    peak_memory_bytes: int
    cache_supported: bool
    verdict: str
    warnings: tuple[str, ...]
    device: str | None = None
    kl_by_depth: tuple[DepthBucketSummary, ...] | None = None
    kv_method: str = "stock"
    kv_method_params: dict[str, int] = dataclasses.field(default_factory=dict)
    kv_method_provenance: dict[str, str] = dataclasses.field(default_factory=dict)
    measured_kv_bytes_per_token: int | None = None
    drift_footing: str = "bundled"
    control_kl: ScalarSummary | None = None
    control_flip_rate: float | None = None
    working_set_bytes_per_token: int | None = None


@dataclass(frozen=True, slots=True)
class WeightFidelityReport:
    """The complete result of a weight-quant fidelity measurement (quant repo vs reference repo)."""

    quant_model_id: str
    quant_revision: str | None
    reference_model_id: str
    reference_revision: str | None
    quant_bits: int | None
    quant_group_size: int | None
    quant_mode: str | None
    per_layer: bool
    reference_bits: int | None
    kl: ScalarSummary
    flip_rate: float
    perplexity_ref: float
    perplexity_quant: float
    perplexity_delta: float
    n_positions: int
    n_chunks: int
    corpus: CorpusProvenance
    mlx_version: str
    mlx_lm_version: str
    peak_memory_bytes: int
    quant_model_bytes: int | None
    reference_model_bytes: int | None
    verdict: str
    warnings: tuple[str, ...]
    device: str | None = None


def render_json(report: FidelityReport | WeightFidelityReport) -> str:
    """Stable, sorted JSON for either report type."""
    return json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True)


def render_weight_markdown(report: WeightFidelityReport) -> str:
    """Human-readable weight-fidelity report. Always qualifies by corpus + context length."""
    bits = report.quant_bits if report.quant_bits is not None else "unknown"
    c = report.corpus
    lines = [
        f"# Weight-fidelity: `{report.quant_model_id}` @ {bits}-bit "
        f"(group {report.quant_group_size}) vs `{report.reference_model_id}`",
        "",
        f"**Verdict:** {report.verdict} (provisional tiers — WikiText-2, "
        "not validated against downstream accuracy)",
    ]
    if report.reference_bits is not None:
        lines += [
            "",
            f"> Reference is itself {report.reference_bits}-bit — drift is relative to an "
            "already-quantized reference, **not full precision**.",
        ]
    lines += [
        "",
        "| metric | value |",
        "|---|---|",
        f"| KL mean | {report.kl.mean:.4f} nats |",
        f"| KL median | {report.kl.median:.4f} nats |",
        f"| KL p99 | {report.kl.p99:.4f} nats |",
        f"| KL max | {report.kl.max:.4f} nats |",
        f"| flip rate | {report.flip_rate:.4f} |",
        f"| perplexity Δ | {report.perplexity_delta:+.4f} "
        f"({report.perplexity_ref:.3f} → {report.perplexity_quant:.3f}) |",
        "",
        f"Measured on **{c.name}/{c.split}**, {report.n_positions} positions across "
        f"{report.n_chunks} chunks of length {c.chunk_length} (tokenizer `{c.tokenizer_id}`). "
        "Fidelity is corpus- and context-length-specific; short-prose temp-0 drift "
        "under-predicts long-context/code degradation.",
        "",
        f"_mlx {report.mlx_version}, mlx-lm {report.mlx_lm_version}, "
        f"quant {report.quant_model_bytes} B, reference {report.reference_model_bytes} B, "
        f"peak {report.peak_memory_bytes / 1e9:.2f} GB"
        f"{f', on {report.device}' if report.device is not None else ''}._",
        *([f"\n> Note: {w}" for w in report.warnings]),
    ]
    return "\n".join(lines)


def fidelity_report_from_dict(d: dict[str, object]) -> FidelityReport:
    """Rehydrate a FidelityReport from `dataclasses.asdict` output (KV compare partials).

    Legacy dicts (pre-0.7.0) lack `drift_footing`; its default is derived from `kv_method`
    because a legacy TurboQuant/adapter dict is quantizer-only, not bundled, even though it
    predates the footing field.
    """
    try:
        kl = d["kl"]
        corpus = d["corpus"]
        if not isinstance(kl, dict) or not isinstance(corpus, dict):
            raise ReportSchemaError("persisted report 'kl'/'corpus' must be dicts")
        fields = {**d, "kl": ScalarSummary(**kl), "corpus": CorpusProvenance(**corpus)}
        fields["warnings"] = tuple(cast("list[str]", fields.get("warnings") or []))
        depth = d.get("kl_by_depth")
        if depth is not None:
            if not isinstance(depth, (list, tuple)) or not all(isinstance(b, dict) for b in depth):
                raise ReportSchemaError("persisted 'kl_by_depth' must be a list of bucket dicts")
            fields["kl_by_depth"] = tuple(DepthBucketSummary(**b) for b in depth)
        control_kl = d.get("control_kl")
        if control_kl is not None:
            if not isinstance(control_kl, dict):
                raise ReportSchemaError("persisted 'control_kl' must be a dict or null")
            fields["control_kl"] = ScalarSummary(**control_kl)
        if "drift_footing" not in d:
            fields["drift_footing"] = (
                "bundled" if d.get("kv_method", "stock") == "stock" else "quantizer_only"
            )
        return FidelityReport(**fields)  # type: ignore[arg-type]
    except ReportSchemaError:
        raise
    except (KeyError, TypeError) as exc:
        raise ReportSchemaError(f"persisted FidelityReport is malformed: {exc}") from exc


def method_bits_text(report: FidelityReport) -> str:
    """Bits label for a report's title/badge: 'N-bit' for stock, else derived from the method.

    Falls back through: an explicit `kv_bits` -> paired k/v bits in `kv_method_params` -> a
    v-only bits label (adapter methods that quantize only V) -> the bare method name when no
    bits info is available at all. Never interpolates `kv_bits` directly when it is `None`.
    """
    if report.kv_bits is not None:
        return f"{report.kv_bits}-bit"
    params = report.kv_method_params
    if "k_bits" in params and "v_bits" in params:
        return f"k{params['k_bits']}v{params['v_bits']}-bit"
    if "v_bits" in params:
        return f"v{params['v_bits']}-bit"
    return report.kv_method


def render_markdown(report: FidelityReport) -> str:
    """Human-readable report. Always qualifies the number by corpus + context length."""
    c = report.corpus
    group = "—" if report.kv_group_size is None else str(report.kv_group_size)
    method_tag = "" if report.kv_method == "stock" else f" via {report.kv_method}"
    lines = [
        f"# KV-fidelity: `{report.model_id}` @ {method_bits_text(report)} (group {group}){method_tag}",
        "",
        f"**Verdict:** {report.verdict} · **mode:** {report.quantize_mode} "
        f"(quantize_start={report.quantize_start})",
        "",
        "| metric | value |",
        "|---|---|",
        f"| KL mean | {report.kl.mean:.4f} nats |",
        f"| KL median | {report.kl.median:.4f} nats |",
        f"| KL p99 | {report.kl.p99:.4f} nats |",
        f"| KL max | {report.kl.max:.4f} nats |",
        f"| flip rate | {report.flip_rate:.4f} |",
        f"| perplexity Δ | {report.perplexity_delta:+.4f} "
        f"({report.perplexity_ref:.3f} → {report.perplexity_quant:.3f}) |",
        "",
        f"Measured on **{c.name}/{c.split}**, {report.n_positions} positions across "
        f"{report.n_chunks} chunks of length {c.chunk_length} (tokenizer `{c.tokenizer_id}`). "
        "Fidelity is corpus- and context-length-specific; short-prose temp-0 drift "
        "under-predicts long-context/code degradation.",
        "",
        f"_mlx {report.mlx_version}, mlx-lm {report.mlx_lm_version}, "
        f"model rev `{report.model_revision}`, peak {report.peak_memory_bytes / 1e9:.2f} GB"
        f"{f', on {report.device}' if report.device is not None else ''}._",
        *([f"\n> Note: {w}" for w in report.warnings]),
    ]
    if report.kl_by_depth:
        lines += [
            "",
            "**Drift by position depth** (stress mode; positions pooled across chunks):",
            "",
            "| positions | KL mean | KL p99 |",
            "|---|---|---|",
        ]
        lines += [
            f"| {b.start}-{b.end - 1} | {b.kl_mean:.4f} | {b.kl_p99:.4f} |"
            for b in report.kl_by_depth
        ]
    if report.quantize_mode == "deployment":
        lines.append(
            f"\n> **Deployment mode:** metrics and the {report.n_positions} positions "
            f"cover only the post-boundary region. The first {report.quantize_start} positions "
            "per window were "
            "computed with a full-precision cache and are excluded; at the boundary, their stored "
            "K/V entries are converted too. Compare drift with stress only in a matched "
            "measurement "
            "(see docs/measurement-principles.md)."
        )
    if report.drift_footing != "bundled" or report.control_kl is not None:
        lines.append(f"\n_drift footing: {report.drift_footing}._")
    if report.control_kl is not None:
        ctrl = report.control_kl
        control_flip = (
            "—" if report.control_flip_rate is None else f"{report.control_flip_rate:.4f}"
        )
        lines += [
            "",
            "**Quantizer-only control** (dequantize → standard SDPA; same corpus, third forward):",
            "",
            "| lane | KL mean | KL p99 | flip |",
            "|---|---|---|---|",
            f"| bundled (deployed path) | {report.kl.mean:.4f} | {report.kl.p99:.4f} | "
            f"{report.flip_rate:.4f} |",
            f"| quantizer-only | {ctrl.mean:.4f} | {ctrl.p99:.4f} | {control_flip} |",
            "",
            "> The difference between the lanes reflects the attention-path change plus "
            "compounded layer-wise divergence; it is not a pure kernel-numerics metric.",
        ]
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ComparisonTargetResult:
    """One target in a comparison: a successful, skipped, or failed measurement."""

    label: str
    status: str  # "ok" | "failed" | "skipped"
    report: FidelityReport | WeightFidelityReport | None
    point: RankPoint | None  # set iff status=="ok" AND cost resolved (rankable)
    excluded_reason: str | None  # e.g. "cost unavailable"; None when ranked
    error_type: str | None  # set iff status=="failed"
    message: str | None  # set iff status=="failed"


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """A memory-normalized method comparison: ranked frontier + dominated + optional budget pick."""

    mode: str  # "weight" | "kv"
    reference: str | None  # reference repo (weight) or None (kv)
    model: str | None  # the single model (kv) or None (weight)
    corpus: CorpusProvenance | None
    quantize_start: int | None
    quantize_mode: str | None
    budget: str | None  # human description of the applied budget, or None
    results: tuple[ComparisonTargetResult, ...]
    frontier: tuple[str, ...]
    dominated: tuple[tuple[str, str], ...]  # (label, dominator)
    budget_pick: str | None
    mlx_version: str
    mlx_lm_version: str


def weight_report_from_dict(d: dict[str, object]) -> WeightFidelityReport:
    """Rehydrate a WeightFidelityReport from `dataclasses.asdict` output (subprocess partials)."""
    try:
        kl = d["kl"]
        corpus = d["corpus"]
        if not isinstance(kl, dict) or not isinstance(corpus, dict):
            raise ReportSchemaError("persisted report 'kl'/'corpus' must be dicts")
        fields = {**d, "kl": ScalarSummary(**kl), "corpus": CorpusProvenance(**corpus)}
        fields["warnings"] = tuple(cast("list[str]", fields.get("warnings") or []))
        return WeightFidelityReport(**fields)  # type: ignore[arg-type]
    except ReportSchemaError:
        raise
    except (KeyError, TypeError) as exc:
        raise ReportSchemaError(f"persisted WeightFidelityReport is malformed: {exc}") from exc


def render_comparison_json(report: ComparisonReport) -> str:
    """Stable, sorted JSON for a comparison report."""
    return json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True)


def _human_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    if n < 1_000:
        return f"{n} B"
    if n < 1_000_000:
        return f"{n / 1e3:.1f} KB"
    if n < 1_000_000_000:
        return f"{n / 1e6:.1f} MB"
    return f"{n / 1e9:.2f} GB"


def render_comparison_markdown(report: ComparisonReport) -> str:
    """Human-readable comparison: ranked table (cost ascending) + excluded rows + recommendation."""
    target = report.reference or report.model or "?"
    lines = [f"# Quant comparison ({report.mode}) vs `{target}`", ""]
    if report.mode == "kv" and report.quantize_mode == "deployment":
        lines += [f"_mode: {report.quantize_mode} (quantize_start={report.quantize_start})_", ""]
    lines += [
        "| target | cost | KL mean | KL p99 | flip | verdict | frontier |",
        "|---|---|---|---|---|---|---|",
    ]
    dominated_by: dict[str, str] = dict(report.dominated)
    ranked = [r for r in report.results if r.point is not None]
    for r in sorted(ranked, key=lambda r: r.point.cost_bytes):  # type: ignore[union-attr]
        assert r.report is not None
        assert r.point is not None
        if r.label in report.frontier:
            mark = "✓"
        else:
            dominator = dominated_by.get(r.label)
            mark = f"✗ dominated by `{dominator}`" if dominator is not None else "✗"
        lines.append(
            f"| `{r.label}` | {_human_bytes(r.point.cost_bytes)} | {r.report.kl.mean:.4f} | "
            f"{r.report.kl.p99:.4f} | {r.report.flip_rate:.4f} | {r.report.verdict} | {mark} |"
        )
    excluded = [r for r in report.results if r.point is None]
    if excluded:
        lines += ["", "**Excluded (not ranked):**"]
        for r in excluded:
            why = r.excluded_reason or f"{r.error_type}: {r.message}"
            lines.append(f"- `{r.label}` — {why}")
    lines += [""]
    if report.budget_pick is not None:
        lines.append(
            f"**Recommended** (cheapest clearing {report.budget or 'budget'}): `{report.budget_pick}`"
        )
    elif report.budget is not None:
        lines.append(f"No target clears the budget ({report.budget}).")
    if report.mode == "weight":
        lines += [
            "",
            "> Weight compare reloads the reference once per target — N targets ≈ Nx a "
            "single `weights` run. Fidelity is corpus- and context-length-specific.",
        ]
    return "\n".join(lines)
