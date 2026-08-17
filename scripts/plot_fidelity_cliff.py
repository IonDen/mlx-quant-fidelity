"""Render the fidelity-cliff chart from committed sample reports.

Reads only ``_artifacts/samples/``; loads no model and uses no GPU, so every plotted
number is reproducible from committed data:

    uv sync --group docs
    uv run python scripts/plot_fidelity_cliff.py

Output: ``docs/assets/charts/fidelity-cliff.svg``.

Bars rather than lines: the committed samples cover bit widths 4 and 8 only, and a line
between two points would imply an interpolation nobody measured.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "assets" / "charts" / "fidelity-cliff.svg"

# docs/threshold-policy.md — the good and marginal mean-KLD ceilings.
KV_BANDS = (0.01, 0.10)
WEIGHT_BANDS = (0.01, 0.20)

VERDICT_COLORS = {"good": "#16A34A", "marginal": "#CA8A04", "bad": "#DC2626"}


@dataclass(frozen=True)
class CliffPoint:
    """One measured configuration, ready to plot."""

    label: str
    bits: int
    kl_mean: float
    verdict: str


def _short_model_name(model_id: str) -> str:
    """Reduce a Hub repo id to the family name used in chart labels."""
    tail = model_id.rsplit("/", 1)[-1]
    for suffix in ("-Instruct-4bit", "-Instruct-8bit", "-Instruct-bf16", "-Instruct"):
        tail = tail.removesuffix(suffix)
    return tail


def collect_kv_points(samples_dir: Path) -> list[CliffPoint]:
    """Collect KV-cache measurements from committed sample reports.

    Long-window ``-cl4096`` variants are excluded: they measure a different chunk
    length, and mixing them into one chart would misattribute the numbers.

    Args:
        samples_dir: Directory of committed ``kv`` sample JSON reports.

    Returns:
        Points sorted by bit width, then label.
    """
    points: list[CliffPoint] = []
    for path in sorted(samples_dir.glob("*.json")):
        if path.stem.endswith("-cl4096"):
            continue
        data: dict[str, Any] = json.loads(path.read_text())
        if "kv_bits" not in data:
            continue
        points.append(
            CliffPoint(
                label=f"{_short_model_name(data['model_id'])} · {data['kv_bits']}-bit KV",
                bits=int(data["kv_bits"]),
                kl_mean=float(data["kl"]["mean"]),
                verdict=str(data["verdict"]),
            )
        )
    return sorted(points, key=lambda p: (p.bits, p.label))


def collect_weight_points(weights_dir: Path) -> list[CliffPoint]:
    """Collect weight-quantization measurements from committed sample reports.

    The label names the reference precision, because one committed pair compares
    4-bit against 8-bit rather than bf16.

    Args:
        weights_dir: Directory of committed ``weights`` sample JSON reports.

    Returns:
        Points sorted by bit width, then label.
    """
    points: list[CliffPoint] = []
    for path in sorted(weights_dir.glob("*.json")):
        data: dict[str, Any] = json.loads(path.read_text())
        if "quant_bits" not in data:
            continue
        reference = data.get("reference_bits")
        ref_label = "bf16" if reference in (None, 16) else f"{reference}-bit"
        points.append(
            CliffPoint(
                label=(
                    f"{_short_model_name(data['quant_model_id'])} · "
                    f"{data['quant_bits']}-bit vs {ref_label}"
                ),
                bits=int(data["quant_bits"]),
                kl_mean=float(data["kl"]["mean"]),
                verdict=str(data["verdict"]),
            )
        )
    return sorted(points, key=lambda p: (p.bits, p.label))


def render(kv: list[CliffPoint], weights: list[CliffPoint], out_path: Path) -> None:
    """Draw both panels and write the SVG.

    matplotlib is imported here rather than at module scope so the collectors above
    stay importable without the dev-only ``docs`` dependency group.

    Args:
        kv: KV-cache panel points.
        weights: Weight-quantization panel points.
        out_path: Destination SVG path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for ax, points, bands, title in (
        (axes[0], kv, KV_BANDS, "KV-cache quantization"),
        (axes[1], weights, WEIGHT_BANDS, "Weight quantization"),
    ):
        labels = [p.label for p in points]
        values = [p.kl_mean for p in points]
        colors = [VERDICT_COLORS[p.verdict] for p in points]
        positions = range(len(points))

        ax.axvspan(1e-6, bands[0], color="#16A34A", alpha=0.14, zorder=0)
        ax.axvspan(bands[0], bands[1], color="#CA8A04", alpha=0.14, zorder=0)
        ax.axvspan(bands[1], 1e2, color="#DC2626", alpha=0.14, zorder=0)
        for boundary in bands:
            ax.axvline(boundary, color="#6B7280", alpha=0.5, linewidth=0.8, linestyle="--")

        ax.barh(list(positions), values, color=colors, height=0.6, zorder=2)
        ax.set_yticks(list(positions), labels, fontsize=9)
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 2e1)
        ax.invert_yaxis()
        ax.set_xlabel("mean KL divergence (nats, log scale) — lower is better", fontsize=9)
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "What quantization costs: WikiText-2, 512-token chunks, quantized from the first token",
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.01,
        "Green / amber / red bands are the good / marginal / bad thresholds. "
        "Apple M1 Max. Every value is read from a committed report under _artifacts/samples/.",
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight")


def main() -> None:
    """Collect the committed samples and render the chart."""
    samples = REPO_ROOT / "_artifacts" / "samples"
    kv = collect_kv_points(samples)
    weights = collect_weight_points(samples / "weights")
    render(kv, weights, OUT_PATH)
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)} — {len(kv)} KV, {len(weights)} weight points")


if __name__ == "__main__":
    main()
