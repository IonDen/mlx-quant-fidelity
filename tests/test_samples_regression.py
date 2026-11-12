"""The stock method reproduces the committed 0.5.x samples: Markdown byte-identical, JSON gains
exactly the additive schema keys (see NEW_KEYS), each at a backward-compatible default. The
committed comparison samples re-render to their committed Markdown byte for byte."""

import json
from pathlib import Path

import pytest

from mlx_quant_fidelity.badge import render_badge_markdown
from mlx_quant_fidelity.report import (
    comparison_report_from_dict,
    fidelity_report_from_dict,
    render_comparison_markdown,
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
# The nested corpus dict gains these additive keys, at their pre-existing-report default.
CORPUS_NEW_KEYS = {"dataset_revision"}


def _assert_unchanged_but_for_corpus_keys(legacy, now):
    """Every legacy value re-renders identically; the corpus dict only gains CORPUS_NEW_KEYS."""
    for key in legacy:
        if key == "corpus":
            assert set(now[key]) - set(legacy[key]) == CORPUS_NEW_KEYS
            assert all(now[key][k] is None for k in CORPUS_NEW_KEYS)
            assert {k: v for k, v in now[key].items() if k in legacy[key]} == legacy[key]
        else:
            assert now[key] == legacy[key], key


def test_the_glob_finds_the_committed_kv_samples():
    """Reds if the KV samples move or the glob stops matching, which would leave the
    parametrized re-render tests below with nothing to check."""
    assert len(KV_JSON) >= 6


@pytest.mark.parametrize("path", KV_JSON, ids=[p.stem for p in KV_JSON])
def test_markdown_rerender_matches_committed(path):
    """Reds if a renderer change alters a committed KV sample's Markdown (every committed .md is
    the render plus one trailing newline, verified against 0.5.1)."""
    report = fidelity_report_from_dict(json.loads(path.read_text()))
    assert render_markdown(report) + "\n" == path.with_suffix(".md").read_text()


@pytest.mark.parametrize("path", KV_JSON, ids=[p.stem for p in KV_JSON])
def test_json_rerender_adds_exactly_the_method_and_footing_keys(path):
    """Reds if a schema change adds an unlisted key, drops a legacy one, or gives a new key a
    default that changes what a pre-0.6.0 sample means."""
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
    _assert_unchanged_but_for_corpus_keys(legacy, now)


def test_badge_sample_matches_committed():
    """Reds if the badge renderer drifts from the committed kv-badge.md, which was rendered from
    llama-3.2-1b-4bit-kv4.json (4-bit, window 512, stress, `bad`)."""
    src = json.loads((SAMPLES / "llama-3.2-1b-4bit-kv4.json").read_text())
    assert (
        render_badge_markdown(fidelity_report_from_dict(src)).strip()
        == (SAMPLES / "kv-badge.md").read_text().strip()
    )


@pytest.mark.parametrize(
    "sample",
    [
        "compare/kv-llama-3.2-1b-methods.md",
        "compare/weight-qwen3-0.6b-ladder.md",
        "compare/weight-qwen3-4b-2507-ladder.md",
    ],
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
    """Reds if the weight samples move or the glob stops matching (an empty parametrization)."""
    assert len(WEIGHT_JSON) >= 5


@pytest.mark.parametrize("path", WEIGHT_JSON, ids=[p.stem for p in WEIGHT_JSON])
def test_weight_markdown_rerender_matches_committed(path):
    """The five 0.5.0-era weight samples keep their captured headline byte-for-byte."""
    report = weight_report_from_dict(json.loads(path.read_text()))
    assert render_weight_markdown(report) + "\n" == path.with_suffix(".md").read_text()


@pytest.mark.parametrize("path", WEIGHT_JSON, ids=[p.stem for p in WEIGHT_JSON])
def test_weight_json_rerender_adds_exactly_the_geometry_keys(path):
    """Reds if a pre-0.8.0 weight sample gains an unlisted key or a non-null geometry field on
    re-render (a legacy report must stay legacy)."""
    legacy = json.loads(path.read_text())
    now = json.loads(render_json(weight_report_from_dict(legacy)))
    assert set(now) - set(legacy) == WEIGHT_NEW_KEYS
    assert all(now[k] is None for k in WEIGHT_NEW_KEYS)
    _assert_unchanged_but_for_corpus_keys(legacy, now)


COMPARE_JSON = sorted((SAMPLES / "compare").glob("*.json"))


def test_the_glob_finds_the_committed_comparison_samples():
    """Reds if the comparison samples move or the glob stops matching (an empty
    parametrization would let the re-render guard below pass on nothing)."""
    assert len(COMPARE_JSON) >= 4


@pytest.mark.parametrize("path", COMPARE_JSON, ids=[p.stem for p in COMPARE_JSON])
def test_comparison_markdown_rerender_matches_committed(path):
    """Reds if a renderer change (a separator row, a column, a footer note) ships without the
    committed comparison .md being regenerated: the committed JSON, rehydrated and rendered,
    must equal the committed Markdown plus one trailing newline, byte for byte.

    Only the Markdown is compared. The JSON is not expected to re-render byte-identically:
    older samples gain a `null` `corpus.dataset_revision` on re-render, which is expected."""
    report = comparison_report_from_dict(json.loads(path.read_text()))
    assert render_comparison_markdown(report) + "\n" == path.with_suffix(".md").read_text()


SAMPLE_FILES = sorted(p for p in SAMPLES.rglob("*") if p.is_file())
OLD_METHOD_SENTENCE = "Quantization method is not recorded in the repo config"


def test_no_committed_sample_carries_the_old_method_not_recorded_sentence():
    """Reds if a committed sample still carries the old method-not-recorded sentence. The weight
    rehydrator swaps that wording for the current one on read, so the re-render guards above
    would pass on a JSON that still holds it; this reads the files as text instead."""
    assert len(SAMPLE_FILES) >= 30  # the scan must actually reach the samples
    carriers = [
        str(p.relative_to(SAMPLES))
        for p in SAMPLE_FILES
        if OLD_METHOD_SENTENCE in p.read_text(encoding="utf-8")
    ]
    assert carriers == []


def _kv_rows(stem):
    """The measured rows of a committed comparison sample: (method, cost_bytes, measured bytes)."""
    from mlx_quant_fidelity.probes.kv_methods import parse_method_spec

    data = json.loads((SAMPLES / "compare" / f"{stem}.json").read_text())
    return [
        (
            parse_method_spec(row["label"]),
            row["point"]["cost_bytes"],
            row["report"]["measured_kv_bytes_per_token"],
        )
        for row in data["results"]
        if row["point"] is not None
    ]


def test_cost_formula_equals_committed_stored_bytes_on_v2_lite():
    """Reds unless each method's formula reproduces the bytes the real run stored. Checks offline
    that the analytic cost of each committed row equals its measured_kv_bytes_per_token (the
    stored bytes), not its cost_bytes (the formula's own output).
    Bug: K and V are costed at one shared width, hidden // heads = 128 (the old rule gave
    117504 / 62208 / 89856)."""
    rows = _kv_rows("kv-deepseek-v2-lite-methods")
    # DeepSeek-V2-Lite: 27 layers, 16 KV heads, K = 128 + 64 = 192, V = 128 (config.json a33a8ad)
    for method, _cost, measured in rows:
        cost = method.bytes_per_token(n_layers=27, n_kv_heads=16, head_dim=192, v_dim=128)
        assert cost == measured, method.label
    assert len(rows) == 3


@pytest.mark.parametrize(
    ("sample", "geo"),
    [
        ("kv-llama-3.2-1b-methods", {"n_layers": 16, "n_kv_heads": 8, "head_dim": 64}),
        ("kv-qwen2.5-0.5b", {"n_layers": 24, "n_kv_heads": 2, "head_dim": 64}),
    ],
)
def test_symmetric_samples_keep_their_cost(sample, geo):
    """The K == V proof. Reds if the split changes any symmetric number, for every shipped method
    (bug: a formula counts V twice or at another width, e.g. K + 2 * V in stock)."""
    rows = _kv_rows(sample)
    for method, cost_bytes, measured in rows:
        assert method.bytes_per_token(**geo) == cost_bytes == measured, method.label
    assert len(rows) >= 3


def test_v2_lite_sample_cost_equals_stored_bytes():
    """Reds on a stale or half-patched V2-Lite sample: a cost_bytes still counting K and V at one
    shared width, hidden // heads = 128 (the old sample had 117504 / 62208 / 89856), which
    undercounts the 192-wide keys, or the analytic-mismatch note surviving
    in the JSON warnings or as a Markdown note. The formula test above reads only the measured
    bytes, so it stays green when the committed cost column is stale."""
    stem = "kv-deepseek-v2-lite-methods"
    rows = _kv_rows(stem)
    for method, cost_bytes, measured in rows:
        assert cost_bytes == measured, method.label
    assert len(rows) == 3
    data = json.loads((SAMPLES / "compare" / f"{stem}.json").read_text())
    warnings = [w for row in data["results"] for w in (row.get("report") or {}).get("warnings", [])]
    stale = [w for w in warnings if "differs from the analytic" in w or "not covered" in w]
    assert stale == []
    md_lines = (SAMPLES / "compare" / f"{stem}.md").read_text().splitlines()
    assert [line for line in md_lines if line.startswith("> Note: measured KV bytes")] == []


def test_measurement_principles_table_matches_v2_lite_sample():
    """Reds if the V2-Lite table in docs/measurement-principles.md drifts from the committed
    sample: a target's cost cell differs (e.g. the old 62.2 KB for 4:64) or a row goes missing."""
    doc = (SAMPLES.parents[1] / "docs" / "measurement-principles.md").read_text()
    sample = (SAMPLES / "compare" / "kv-deepseek-v2-lite-methods.md").read_text()
    target_rows = [line for line in sample.splitlines() if line.startswith("| `")]
    assert len(target_rows) >= 3
    for row in target_rows:
        target, cost = (cell.strip() for cell in row.split("|")[1:3])
        assert f"| {target} | {cost} |" in doc, row
