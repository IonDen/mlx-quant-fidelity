"""The stock method reproduces the committed 0.5.x samples: Markdown byte-identical, JSON +4 keys."""

import json
from pathlib import Path

import pytest

from mlx_quant_fidelity.badge import render_badge_markdown
from mlx_quant_fidelity.report import fidelity_report_from_dict, render_json, render_markdown

SAMPLES = Path(__file__).resolve().parents[1] / "_artifacts" / "samples"
KV_JSON = sorted(p for p in SAMPLES.glob("*.json") if p.name.startswith(("llama-", "qwen")))
NEW_KEYS = {"kv_method", "kv_method_params", "kv_method_provenance", "measured_kv_bytes_per_token"}


def test_the_glob_finds_the_committed_kv_samples():
    assert len(KV_JSON) >= 6  # guards against a silent empty parametrization


@pytest.mark.parametrize("path", KV_JSON, ids=[p.stem for p in KV_JSON])
def test_markdown_rerender_matches_committed(path):
    # every committed .md is the render plus one trailing newline (verified against 0.5.1)
    report = fidelity_report_from_dict(json.loads(path.read_text()))
    assert render_markdown(report) + "\n" == path.with_suffix(".md").read_text()


@pytest.mark.parametrize("path", KV_JSON, ids=[p.stem for p in KV_JSON])
def test_json_rerender_adds_exactly_the_four_method_keys(path):
    legacy = json.loads(path.read_text())
    now = json.loads(render_json(fidelity_report_from_dict(legacy)))
    assert set(now) - set(legacy) == NEW_KEYS
    assert now["kv_method"] == "stock"
    for key in legacy:
        assert now[key] == legacy[key], key


def test_badge_sample_matches_committed():
    # kv-badge.md was rendered from llama-3.2-1b-4bit-kv4.json (4-bit, window 512, stress, `bad`)
    src = json.loads((SAMPLES / "llama-3.2-1b-4bit-kv4.json").read_text())
    assert (
        render_badge_markdown(fidelity_report_from_dict(src)).strip()
        == (SAMPLES / "kv-badge.md").read_text().strip()
    )


def test_readme_method_table_matches_committed_sample():
    readme = (SAMPLES.parents[1] / "README.md").read_text()
    table = (SAMPLES / "compare" / "kv-llama-3.2-1b-methods.md").read_text()
    for line in table.splitlines():
        if line.startswith("|"):
            assert line in readme, line
