"""FidelityReport (frozen) + JSON / Markdown renderers. Consumes policy's verdict."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_quant_fidelity.corpora.provenance import CorpusProvenance
    from mlx_quant_fidelity.metrics import ScalarSummary


@dataclass(frozen=True, slots=True)
class FidelityReport:
    """The complete result of a KV-fidelity measurement. Frozen; stable to diff."""

    model_id: str
    model_revision: str | None
    kv_bits: int
    kv_group_size: int
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


def render_json(report: FidelityReport | WeightFidelityReport) -> str:
    """Stable, sorted JSON for either report type."""
    return json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True)


def render_weight_markdown(report: WeightFidelityReport) -> str:
    """Human-readable weight-fidelity report. Always qualifies by corpus + context length."""
    c = report.corpus
    lines = [
        f"# Weight-fidelity: `{report.quant_model_id}` @ {report.quant_bits}-bit "
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
        f"peak {report.peak_memory_bytes / 1e9:.2f} GB._",
        *([f"\n> Note: {w}" for w in report.warnings]),
    ]
    return "\n".join(lines)


def render_markdown(report: FidelityReport) -> str:
    """Human-readable report. Always qualifies the number by corpus + context length."""
    c = report.corpus
    return "\n".join(
        [
            f"# KV-fidelity: `{report.model_id}` @ {report.kv_bits}-bit (group {report.kv_group_size})",
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
            f"model rev `{report.model_revision}`, peak {report.peak_memory_bytes / 1e9:.2f} GB._",
            *([f"\n> Note: {w}" for w in report.warnings]),
        ]
    )
