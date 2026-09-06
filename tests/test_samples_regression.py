"""The stock method reproduces the committed 0.5.x samples: Markdown byte-identical, JSON gains
exactly the additive schema keys (see NEW_KEYS), each at a backward-compatible default."""

import json
from pathlib import Path

import pytest

from mlx_quant_fidelity.badge import render_badge_markdown
from mlx_quant_fidelity.report import (
    fidelity_report_from_dict,
    render_json,
    render_markdown,
    render_weight_markdown,
    weight_report_from_dict,
)

SAMPLES = Path(__file__).resolve().parents[1] / "_artifacts" / "samples"
KV_JSON = sorted(p for p in SAMPLES.glob("*.json") if p.name.startswith(("llama-", "qwen")))
NEW_KEYS = {
    "kv_method",
    "kv_method_params",
    "kv_method_provenance",
    "measured_kv_bytes_per_token",
    "drift_footing",
    "control_kl",
    "control_flip_rate",
    "working_set_bytes_per_token",
    # 0.9.0 per-layer partial coverage — default to full-coverage on these pre-0.9.0 samples.
    "kv_partial",
    "kv_layers_total",
    "kv_layers_quantized",
    "kv_layers_skipped",
}


def test_the_glob_finds_the_committed_kv_samples():
    assert len(KV_JSON) >= 6  # guards against a silent empty parametrization


@pytest.mark.parametrize("path", KV_JSON, ids=[p.stem for p in KV_JSON])
def test_markdown_rerender_matches_committed(path):
    # every committed .md is the render plus one trailing newline (verified against 0.5.1)
    report = fidelity_report_from_dict(json.loads(path.read_text()))
    assert render_markdown(report) + "\n" == path.with_suffix(".md").read_text()


@pytest.mark.parametrize("path", KV_JSON, ids=[p.stem for p in KV_JSON])
def test_json_rerender_adds_exactly_the_method_and_footing_keys(path):
    legacy = json.loads(path.read_text())
    now = json.loads(render_json(fidelity_report_from_dict(legacy)))
    assert set(now) - set(legacy) == NEW_KEYS
    assert now["kv_method"] == "stock"
    assert now["drift_footing"] == "bundled"
    assert now["control_kl"] is None
    # 0.9.0: these full-quantizable samples are never partial, so the new keys take their
    # full-coverage defaults and the committed headline is unchanged.
    assert now["kv_partial"] is False
    assert now["kv_layers_total"] is None
    assert now["kv_layers_quantized"] is None
    assert now["kv_layers_skipped"] is None
    for key in legacy:
        assert now[key] == legacy[key], key


def test_badge_sample_matches_committed():
    # kv-badge.md was rendered from llama-3.2-1b-4bit-kv4.json (4-bit, window 512, stress, `bad`)
    src = json.loads((SAMPLES / "llama-3.2-1b-4bit-kv4.json").read_text())
    assert (
        render_badge_markdown(fidelity_report_from_dict(src)).strip()
        == (SAMPLES / "kv-badge.md").read_text().strip()
    )


@pytest.mark.parametrize(
    "sample",
    ["compare/kv-llama-3.2-1b-methods.md", "compare/weight-qwen3-0.6b-ladder.md"],
)
def test_readme_tables_match_committed_samples(sample):
    """Reds if a README table drifts from the committed sample it claims to reproduce, and reds
    if a sample loses its table (the loop would otherwise assert nothing)."""
    readme = (SAMPLES.parents[1] / "README.md").read_text()
    rows = [line for line in (SAMPLES / sample).read_text().splitlines() if line.startswith("|")]
    assert len(rows) >= 3  # header, separator, at least one target row
    for row in rows:
        assert row in readme, row


WEIGHT_JSON = sorted((SAMPLES / "weights").glob("*.json"))
WEIGHT_NEW_KEYS = {
    "quant_geometry",
    "quant_n_full_precision",
    "quant_bits_per_weight",
    "quant_precision",
}


def test_the_glob_finds_the_committed_weight_samples():
    assert len(WEIGHT_JSON) >= 5


@pytest.mark.parametrize("path", WEIGHT_JSON, ids=[p.stem for p in WEIGHT_JSON])
def test_weight_markdown_rerender_matches_committed(path):
    """The five 0.5.0-era weight samples keep their captured headline byte-for-byte."""
    report = weight_report_from_dict(json.loads(path.read_text()))
    assert render_weight_markdown(report) + "\n" == path.with_suffix(".md").read_text()


@pytest.mark.parametrize("path", WEIGHT_JSON, ids=[p.stem for p in WEIGHT_JSON])
def test_weight_json_rerender_adds_exactly_the_geometry_keys(path):
    legacy = json.loads(path.read_text())
    now = json.loads(render_json(weight_report_from_dict(legacy)))
    assert set(now) - set(legacy) == WEIGHT_NEW_KEYS
    assert all(now[k] is None for k in WEIGHT_NEW_KEYS)
    for key in legacy:
        assert now[key] == legacy[key], key
