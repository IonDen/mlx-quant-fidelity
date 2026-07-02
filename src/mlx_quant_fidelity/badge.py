"""Policy-driven fidelity badge renderer (shields.io). Consumes the verdict; computes no thresholds."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_quant_fidelity.report import FidelityReport, WeightFidelityReport

_COLORS = {"good": "brightgreen", "marginal": "yellow", "bad": "red"}


def badge_color(verdict: str) -> str:
    """Map a verdict to a shields.io color. Raises ValueError on an unknown verdict."""
    try:
        return _COLORS[verdict]
    except KeyError:
        raise ValueError(
            f"unknown verdict {verdict!r}; expected one of {sorted(_COLORS)}"
        ) from None


def badge_for_report(report: "FidelityReport | WeightFidelityReport") -> dict[str, str]:
    """Assemble {label, message, color} from a report. Message carries corpus + length + mode."""
    from mlx_quant_fidelity.report import FidelityReport

    corpus = report.corpus
    if isinstance(report, FidelityReport):
        label = "KV fidelity"
        message = (
            f"{report.verdict} · {report.kv_bits}-bit · "
            f"{corpus.name}/{corpus.chunk_length} · {report.quantize_mode}"
        )
    else:
        label = "Weight fidelity"
        bits = report.quant_bits if report.quant_bits is not None else "?"
        message = (
            f"{report.verdict} · {bits}-bit · {corpus.name}/{corpus.chunk_length} · provisional"
        )
    return {"label": label, "message": message, "color": badge_color(report.verdict)}


def _shields_escape(text: str) -> str:
    """shields.io static-badge field escaping: '-'→'--', '_'→'__', space→'_'."""
    return text.replace("-", "--").replace("_", "__").replace(" ", "_")


def render_badge_markdown(report: "FidelityReport | WeightFidelityReport") -> str:
    """A markdown image line with a static img.shields.io/badge URL (fields escaped)."""
    f = badge_for_report(report)
    label, message = _shields_escape(f["label"]), _shields_escape(f["message"])
    url = f"https://img.shields.io/badge/{label}-{message}-{f['color']}"
    return f"![{f['label']}]({url})"
