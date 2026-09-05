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


METHOD_NOT_RECORDED_WARNING = (
    "Quantization method is not recorded in the repo config — mlx-lm's DWQ, AWQ and dynamic "
    "quantizers write a quantization block that records only geometry (bits, group sizes, "
    "per-module overrides), never the recipe. This report describes the geometry of the loaded "
    "model, not the method that produced it."
)
"""The standing caveat every weight report carries. Lives here, not in ``probes.weights``, so
the renderer can re-add it to a comparison assembled from partials written before 0.8.0 without
``report.py`` importing the probe (and with it mlx-lm)."""


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
    quant_geometry: tuple[tuple[int, int, int], ...] | None = None
    quant_n_full_precision: int | None = None
    quant_bits_per_weight: float | None = None
    quant_precision: str | None = None  # "uniform" | "mixed"


def weight_bits_text(report: WeightFidelityReport) -> str:
    """`4-bit`, `mixed 4/5-bit`, or `?-bit` — the short precision label for badges."""
    if report.quant_geometry:
        bits = sorted({b for b, _, _ in report.quant_geometry})
        if len(bits) > 1:
            return "mixed " + "/".join(str(b) for b in bits) + "-bit"
        return f"{bits[0]}-bit"
    return f"{report.quant_bits}-bit" if report.quant_bits is not None else "?-bit"


def weight_precision_text(report: WeightFidelityReport) -> str | None:
    """Headline precision text for a measured report; None for a legacy (pre-0.8.0) report."""
    bpw = report.quant_bits_per_weight
    if bpw is None:
        return None
    geometry = report.quant_geometry or ()
    groups = sorted({g for _, g, _ in geometry}) or (
        [report.quant_group_size] if report.quant_group_size is not None else []
    )
    parts = ["group " + "/".join(str(g) for g in groups), f"{bpw:.2f} bits/weight"]
    extras: list[str] = []
    bits = sorted({b for b, _, _ in geometry})
    if len(bits) > 1:
        per_bits = {b: sum(n for bb, _, n in geometry if bb == b) for b in bits}
        extras.append(
            f"{per_bits[bits[0]]} modules at {bits[0]}-bit, "
            + ", ".join(f"{per_bits[b]} at {b}-bit" for b in bits[1:])
        )
    if not geometry:
        extras.append("0 quantized modules")
    n_full = report.quant_n_full_precision or 0
    if n_full:
        noun = "module" if n_full == 1 else "modules"
        extras.append(f"{n_full} quantizable {noun} left at full precision")
    tail = "; " + "; ".join(extras) if extras else ""
    return f"{weight_bits_text(report)} ({', '.join(parts)}{tail})"


def render_json(report: FidelityReport | WeightFidelityReport) -> str:
    """Stable, sorted JSON for either report type."""
    return json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True)


def render_weight_markdown(report: WeightFidelityReport) -> str:
    """Human-readable weight-fidelity report. Always qualifies by corpus + context length."""
    precision = weight_precision_text(report)
    if precision is None:  # legacy report: the pre-0.8.0 headline, byte-for-byte
        bits = report.quant_bits if report.quant_bits is not None else "unknown"
        precision = f"{bits}-bit (group {report.quant_group_size})"
    c = report.corpus
    lines = [
        f"# Weight-fidelity: `{report.quant_model_id}` @ {precision} vs `{report.reference_model_id}`",
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
    # KV-only (weight mode leaves these None): the quantizer-only value this row is actually
    # RANKED on, its verdict, and the footing that ranking uses — always "quantizer_only" for
    # a `compare kv` row. `report.kl`/`report.verdict` stay the method's own native (possibly
    # bundled) numbers; these three carry the ranking-time numbers, with a native fallback
    # wherever they're None (see `assemble_comparison_report`/`_kv_envelope_to_result`).
    ranked_kl: float | None = None
    ranked_verdict: str | None = None
    ranked_footing: str | None = None


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
        geometry = d.get("quant_geometry")
        if geometry is not None:
            ok = isinstance(geometry, (list, tuple)) and all(
                isinstance(t, (list, tuple))
                and len(t) == 3
                and all(isinstance(x, int) and not isinstance(x, bool) for x in t)
                for t in geometry
            )
            if not ok:
                raise ReportSchemaError(
                    "persisted 'quant_geometry' must be a list of [bits, group_size, n_modules] int triples"
                )
            fields["quant_geometry"] = tuple(
                (int(t[0]), int(t[1]), int(t[2])) for t in cast("list[list[int]]", geometry)
            )
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


def _kv_row_kl_and_flip(
    report: FidelityReport | WeightFidelityReport, ranked_kl: float | None
) -> tuple[float, float, float]:
    """(KL mean, KL p99, flip) a kv-mode row shows in its ranked columns.

    `kl`/`flip_rate` are shared by both report types; `control_kl`/`control_flip_rate` are
    KV-only — `getattr` with a default guards a row whose `.report` happens to be a
    `WeightFidelityReport` (existing kv-mode tests build rows that way) as well as a legacy
    partial with no control lane recorded.
    """
    kl_mean = ranked_kl if ranked_kl is not None else report.kl.mean
    control_kl = getattr(report, "control_kl", None)
    if control_kl is not None:
        control_flip = getattr(report, "control_flip_rate", None)
        flip = control_flip if control_flip is not None else report.flip_rate
        return kl_mean, control_kl.p99, flip
    return kl_mean, report.kl.p99, report.flip_rate


def render_comparison_markdown(report: ComparisonReport) -> str:
    """Human-readable comparison: ranked table (cost ascending) + excluded rows + recommendation.

    A ``kv``-mode table carries two extra columns a ``weight``-mode table does not: ``bundled
    KL`` (the method's own native number, shown only where it differs from the ranked one —
    i.e. only for a stock row that ran the quantizer-only control) and ``resident +/token``
    (the method's working-set overhead, from ``FidelityReport.working_set_bytes_per_token``).
    """
    target = report.reference or report.model or "?"
    lines = [f"# Quant comparison ({report.mode}) vs `{target}`", ""]
    if report.mode == "kv" and report.quantize_mode == "deployment":
        lines += [f"_mode: {report.quantize_mode} (quantize_start={report.quantize_start})_", ""]
    is_kv = report.mode == "kv"
    if is_kv:
        lines += [
            "| target | cost | KL mean | KL p99 | flip | bundled KL | resident +/token | "
            "verdict | frontier |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    else:
        lines += [
            "| target | cost | KL mean | KL p99 | flip | bits/wt | verdict | frontier |",
            "|---|---|---|---|---|---|---|---|",
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
        if is_kv:
            kl_mean, kl_p99, flip = _kv_row_kl_and_flip(r.report, r.ranked_kl)
            verdict = r.ranked_verdict if r.ranked_verdict is not None else r.report.verdict
            bundled = (
                f"{r.report.kl.mean:.4f}"
                if r.ranked_kl is not None and getattr(r.report, "kv_method", None) == "stock"
                else "—"
            )
            resident = _human_bytes(getattr(r.report, "working_set_bytes_per_token", None))
            lines.append(
                f"| `{r.label}` | {_human_bytes(r.point.cost_bytes)} | {kl_mean:.4f} | "
                f"{kl_p99:.4f} | {flip:.4f} | {bundled} | {resident} | {verdict} | {mark} |"
            )
        else:
            bpw_value = getattr(r.report, "quant_bits_per_weight", None)
            bpw = f"{bpw_value:.2f}" if bpw_value is not None else "—"
            lines.append(
                f"| `{r.label}` | {_human_bytes(r.point.cost_bytes)} | {r.report.kl.mean:.4f} | "
                f"{r.report.kl.p99:.4f} | {r.report.flip_rate:.4f} | {bpw} | {r.report.verdict} | {mark} |"
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
    seen_warnings: set[str] = set()
    for r in report.results:
        if r.report is None:
            continue
        for w in r.report.warnings:
            if w not in seen_warnings:
                seen_warnings.add(w)
                lines.append(f"\n> Note: {w}")
    if is_kv:
        lines.append(
            "\n_ranked on quantizer-only drift; stock rows carry their bundled deployment "
            "drift alongside._"
        )
    if report.mode == "weight":
        # A comparison resumed from partials written before 0.8.0 has rows with no method
        # caveat of their own; the caveat is a property of the weight probe, not of a row.
        if METHOD_NOT_RECORDED_WARNING not in seen_warnings:
            lines.append(f"\n> Note: {METHOD_NOT_RECORDED_WARNING}")
        lines += [
            "",
            "> Weight compare reloads the reference once per target — N targets ≈ Nx a "
            "single `weights` run. Fidelity is corpus- and context-length-specific.",
        ]
    return "\n".join(lines)
