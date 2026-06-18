"""Tests for compare_kv_fidelity: one-load loop, resume, isolation, validation guards."""

import dataclasses
import json
import types

import pytest

from mlx_quant_fidelity.runners import compare as cmp


def _fid(label_bits: tuple[int, int], kl_mean: float):  # type: ignore[return]
    from mlx_quant_fidelity.corpora.provenance import CorpusProvenance
    from mlx_quant_fidelity.metrics import ScalarSummary
    from mlx_quant_fidelity.report import FidelityReport

    bits, gs = label_bits
    return FidelityReport(
        model_id="m",
        model_revision=None,
        kv_bits=bits,
        kv_group_size=gs,
        quantize_start=0,
        quantize_mode="stress",
        kl=ScalarSummary(kl_mean, kl_mean, kl_mean, kl_mean),
        flip_rate=0.01,
        perplexity_ref=10.0,
        perplexity_quant=10.1,
        perplexity_delta=0.1,
        n_positions=10,
        n_chunks=2,
        corpus=CorpusProvenance("wikitext-2-raw", "test", "m", 512, 512, "none", "drop", "raw", 10),
        mlx_version="0.21",
        mlx_lm_version="0.31.3",
        peak_memory_bytes=1,
        cache_supported=True,
        verdict="good",
        warnings=(),
    )


def _patch_kv_compare(
    monkeypatch,
    reports: dict,
    *,
    dims: tuple[int, int, int] = (16, 8, 64),
) -> list:
    """Patch all real-model helpers; return the call list for score_kv_config."""
    monkeypatch.setattr(cmp, "install_memory_caps", lambda: (0, 0))
    monkeypatch.setattr(cmp, "_load_model", lambda model_id, revision: object())
    monkeypatch.setattr(cmp, "_kv_dims", lambda model: dims)
    calls: list[tuple[int, int]] = []

    def fake_score(model, corpus, *, kv_bits, kv_group_size, **kw):  # type: ignore[return]
        calls.append((kv_bits, kv_group_size))
        rep = reports.get((kv_bits, kv_group_size))
        if isinstance(rep, Exception):
            raise rep
        return rep

    monkeypatch.setattr(cmp, "score_kv_config", fake_score)
    monkeypatch.setattr(cmp, "_load_corpus_for_kv", lambda model, model_id, max_chunks: object())
    return calls


def test_compare_kv_ranks_configs(monkeypatch, tmp_path):
    """Both configs score and land on the Pareto frontier (different cost, different quality)."""
    reports = {(4, 64): _fid((4, 64), 0.09), (8, 64): _fid((8, 64), 0.01)}
    _patch_kv_compare(monkeypatch, reports)
    report = cmp.compare_kv_fidelity("m", [(4, 64), (8, 64)], artifacts_dir=tmp_path)
    assert report.mode == "kv"
    assert report.model == "m"
    # 8-bit has better quality but bigger cost; 4-bit cheaper but worse quality →
    # neither dominates the other → both on the frontier
    assert set(report.frontier) == {"4:64", "8:64"}


def test_compare_kv_isolates_unsupported_config(monkeypatch, tmp_path):
    """A CacheNotQuantizableError for one config is isolated; others complete normally."""
    from mlx_quant_fidelity.errors import CacheNotQuantizableError

    reports = {
        (4, 64): _fid((4, 64), 0.09),
        (3, 999): CacheNotQuantizableError("no divide"),
    }
    _patch_kv_compare(monkeypatch, reports)
    report = cmp.compare_kv_fidelity("m", [(4, 64), (3, 999)], artifacts_dir=tmp_path)
    failed = next(r for r in report.results if r.label == "3:999")
    assert failed.status == "failed"
    assert failed.error_type == "CacheNotQuantizableError"
    assert "3:999" not in report.frontier
    # Fix 9: positive isolation assert — the good config IS on the frontier alone
    assert "4:64" in report.frontier
    assert len(report.frontier) == 1


def test_compare_kv_requires_two_configs(tmp_path):
    """Fewer than 2 configs raises ValueError before any model load."""
    with pytest.raises(ValueError, match="at least 2"):
        cmp.compare_kv_fidelity("m", [(4, 64)], artifacts_dir=tmp_path)


def test_compare_kv_rejects_deployment_mode(monkeypatch, tmp_path):
    """quantize_start != 0 raises ValueError (only stress mode in 0.x)."""
    _patch_kv_compare(monkeypatch, {})
    with pytest.raises(ValueError, match="stress mode"):
        cmp.compare_kv_fidelity(
            "m", [(4, 64), (8, 64)], quantize_start=5000, artifacts_dir=tmp_path
        )


def test_compare_kv_rejects_bad_max_chunks(monkeypatch, tmp_path):
    """max_chunks < 1 raises ValueError."""
    _patch_kv_compare(monkeypatch, {})
    with pytest.raises(ValueError, match="max_chunks"):
        cmp.compare_kv_fidelity("m", [(4, 64), (8, 64)], max_chunks=0, artifacts_dir=tmp_path)


def test_compare_kv_rejects_duplicate_configs(monkeypatch, tmp_path):
    """Duplicate (bits, group_size) tuples raise ValueError."""
    _patch_kv_compare(monkeypatch, {})
    with pytest.raises(ValueError, match="duplicate"):
        cmp.compare_kv_fidelity("m", [(4, 64), (4, 64), (8, 64)], artifacts_dir=tmp_path)


def test_compare_kv_resume_skips_existing_partial(monkeypatch, tmp_path):
    """A config whose partial JSON already exists is not re-scored."""
    rep = _fid((4, 64), 0.09)
    partial_name = "4_64.json"  # ':' sanitized to '_'
    (tmp_path / partial_name).write_text(
        json.dumps({"status": "ok", "report": dataclasses.asdict(rep), "cost": 1234})
    )
    calls = _patch_kv_compare(monkeypatch, {(8, 64): _fid((8, 64), 0.01)})
    cmp.compare_kv_fidelity("m", [(4, 64), (8, 64)], artifacts_dir=tmp_path)
    # only (8, 64) should have been scored — (4, 64) resumed from partial
    assert calls == [(8, 64)]


def test_compare_kv_corrupt_partial_reruns(monkeypatch, tmp_path):
    """A corrupt/truncated partial triggers a re-run, not a crash."""
    (tmp_path / "4_64.json").write_text("{bad json")
    calls = _patch_kv_compare(
        monkeypatch,
        {(4, 64): _fid((4, 64), 0.09), (8, 64): _fid((8, 64), 0.01)},
    )
    report = cmp.compare_kv_fidelity("m", [(4, 64), (8, 64)], artifacts_dir=tmp_path)
    # Fix 12: tighten the assert — both configs must have been scored, in order
    assert calls == [(4, 64), (8, 64)]
    assert len(report.results) == 2


def test_compare_kv_partial_filename_sanitizes_colon(monkeypatch, tmp_path):
    """The partial file for '4:64' is written as '4_64.json' (colon → underscore)."""
    reports = {(4, 64): _fid((4, 64), 0.09), (8, 64): _fid((8, 64), 0.01)}
    _patch_kv_compare(monkeypatch, reports)
    cmp.compare_kv_fidelity("m", [(4, 64), (8, 64)], artifacts_dir=tmp_path)
    assert (tmp_path / "4_64.json").exists()
    assert (tmp_path / "8_64.json").exists()
    # No colon-containing filenames should exist
    assert not any(":" in f.name for f in tmp_path.iterdir())


def test_compare_kv_cost_unavailable_when_dims_none(monkeypatch, tmp_path):
    """When _kv_dims returns any None, the config is unrankable (excluded_reason set)."""
    reports = {(4, 64): _fid((4, 64), 0.09), (8, 64): _fid((8, 64), 0.01)}
    _patch_kv_compare(monkeypatch, reports, dims=(None, None, None))
    report = cmp.compare_kv_fidelity("m", [(4, 64), (8, 64)], artifacts_dir=tmp_path)
    for r in report.results:
        if r.status == "ok":
            assert r.point is None
            assert r.excluded_reason == "cost unavailable"
    assert report.frontier == ()


def test_compare_kv_label_uses_colon(monkeypatch, tmp_path):
    """Report labels use '4:64' format (colon), not the filename format."""
    reports = {(4, 64): _fid((4, 64), 0.09), (8, 64): _fid((8, 64), 0.01)}
    _patch_kv_compare(monkeypatch, reports)
    report = cmp.compare_kv_fidelity("m", [(4, 64), (8, 64)], artifacts_dir=tmp_path)
    labels = {r.label for r in report.results}
    assert labels == {"4:64", "8:64"}


def test_compare_kv_model_loaded_once(monkeypatch, tmp_path):
    """_load_model is called exactly once and _load_corpus_for_kv is called exactly once."""
    reports = {(4, 64): _fid((4, 64), 0.09), (8, 64): _fid((8, 64), 0.01)}
    load_calls: list[str] = []

    def fake_load(model_id: str, revision) -> object:
        load_calls.append(model_id)
        return object()

    monkeypatch.setattr(cmp, "install_memory_caps", lambda: (0, 0))
    monkeypatch.setattr(cmp, "_load_model", fake_load)
    monkeypatch.setattr(cmp, "_kv_dims", lambda model: (16, 8, 64))

    # Fix 10: count corpus-build calls with a wrapper, assert exactly once
    corpus_calls: list[object] = []

    def counting_load_corpus(model, model_id, max_chunks):  # type: ignore[return]
        corpus_calls.append(model_id)
        return object()

    def fake_score(model, corpus, *, kv_bits, kv_group_size, **kw):  # type: ignore[return]
        return reports[(kv_bits, kv_group_size)]

    monkeypatch.setattr(cmp, "score_kv_config", fake_score)
    monkeypatch.setattr(cmp, "_load_corpus_for_kv", counting_load_corpus)
    cmp.compare_kv_fidelity("m", [(4, 64), (8, 64)], artifacts_dir=tmp_path)
    assert len(load_calls) == 1
    assert len(corpus_calls) == 1


# ── Fix 7a: _kv_dims unit tests (closes coverage gap) ─────────────────────────


def _make_fake_model(*, num_key_value_heads=None, num_attention_heads=None, n_layers=3):
    """Build a minimal fake model object for _kv_dims testing (no real MLX model required)."""
    args = types.SimpleNamespace(
        num_key_value_heads=num_key_value_heads,
        num_attention_heads=num_attention_heads,
    )
    model = types.SimpleNamespace(args=args)
    return model, n_layers


def test_kv_dims_uses_num_key_value_heads(monkeypatch):
    """_kv_dims returns num_key_value_heads when present."""
    model, n_layers = _make_fake_model(num_key_value_heads=4, num_attention_heads=8)
    import mlx_lm.models.cache as cache_mod

    import mlx_quant_fidelity.runners.compare as compare_mod

    monkeypatch.setattr(cache_mod, "make_prompt_cache", lambda m: [None] * n_layers)
    monkeypatch.setattr(compare_mod, "_kv_head_dim", lambda m: 64)

    result = compare_mod._kv_dims(model)
    assert result == (n_layers, 4, 64)


def test_kv_dims_falls_back_to_num_attention_heads(monkeypatch):
    """_kv_dims falls back to num_attention_heads when num_key_value_heads is absent."""
    model, n_layers = _make_fake_model(num_key_value_heads=None, num_attention_heads=8)
    import mlx_lm.models.cache as cache_mod

    import mlx_quant_fidelity.runners.compare as compare_mod

    monkeypatch.setattr(cache_mod, "make_prompt_cache", lambda m: [None] * n_layers)
    monkeypatch.setattr(compare_mod, "_kv_head_dim", lambda m: 128)

    result = compare_mod._kv_dims(model)
    assert result == (n_layers, 8, 128)


def test_kv_dims_zero_kv_heads_not_silenced(monkeypatch):
    """_kv_dims does NOT silently fall through when num_key_value_heads=0 (fix 3).

    With the old `or` short-circuit, 0 would have been treated as falsy and the code
    would have fallen through to num_attention_heads. The fix uses `is not None` so
    0 is returned as the explicit value, not replaced.
    """
    model, n_layers = _make_fake_model(num_key_value_heads=0, num_attention_heads=8)
    import mlx_lm.models.cache as cache_mod

    import mlx_quant_fidelity.runners.compare as compare_mod

    monkeypatch.setattr(cache_mod, "make_prompt_cache", lambda m: [None] * n_layers)
    monkeypatch.setattr(compare_mod, "_kv_head_dim", lambda m: 64)

    result = compare_mod._kv_dims(model)
    # Must return 0 for n_kv_heads — NOT fall through to num_attention_heads=8
    assert result == (n_layers, 0, 64)


# ── Fix 7b: corrupt-at-collect test ───────────────────────────────────────────


def test_compare_kv_corrupt_at_collect_yields_failed_result(monkeypatch, tmp_path):
    """A partial that becomes corrupt AFTER the pending check (at collect time) is a failed result.

    Simulates a write failure or filesystem corruption between the pending-detection phase and
    the collect loop. We seed both partials as valid (so pending=[]), then inject a read_text
    side-effect that returns corrupt JSON on the second read of 4_64.json (the collect-phase
    read). The run must complete and that config becomes a 'failed' ComparisonTargetResult
    with error_type='CorruptPartial', while the other config's result is intact.
    """
    import pathlib

    rep_4 = _fid((4, 64), 0.09)
    rep_8 = _fid((8, 64), 0.01)

    valid_4 = json.dumps({"status": "ok", "report": dataclasses.asdict(rep_4), "cost": 1000})
    valid_8 = json.dumps({"status": "ok", "report": dataclasses.asdict(rep_8), "cost": 2000})

    # Pre-seed both partials as valid
    (tmp_path / "4_64.json").write_text(valid_4)
    (tmp_path / "8_64.json").write_text(valid_8)

    # Track read_text calls per path name; on the 2nd call for 4_64.json return corrupt data
    read_counts: dict[str, int] = {}
    original_read_text = pathlib.Path.read_text

    def patched_read_text(self, *args, **kwargs):  # type: ignore[return]
        name = self.name
        read_counts[name] = read_counts.get(name, 0) + 1
        if name == "4_64.json" and read_counts[name] >= 2:
            return "{corrupt json at collect time!!!"
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", patched_read_text)

    monkeypatch.setattr(cmp, "install_memory_caps", lambda: (0, 0))
    monkeypatch.setattr(cmp, "_load_model", lambda model_id, revision: object())
    monkeypatch.setattr(cmp, "_kv_dims", lambda model: (16, 8, 64))
    monkeypatch.setattr(cmp, "_load_corpus_for_kv", lambda m, mid, mc: object())
    # score_kv_config should NOT be called (pending=[]) — use a sentinel that would raise
    score_calls: list[object] = []
    monkeypatch.setattr(cmp, "score_kv_config", lambda *a, **kw: score_calls.append(None))

    report = cmp.compare_kv_fidelity("m", [(4, 64), (8, 64)], artifacts_dir=tmp_path)

    # scoring must not have been triggered (all-resumed path up to collect)
    assert score_calls == []

    # The corrupt config must be a failed result with CorruptPartial error_type
    failed = next(r for r in report.results if r.label == "4:64")
    assert failed.status == "failed"
    assert failed.error_type == "CorruptPartial"

    # The good config must be intact
    good = next(r for r in report.results if r.label == "8:64")
    assert good.status == "ok"

    # Overall run must have completed (2 results)
    assert len(report.results) == 2


# ── Fix 8: all-cached resume (pending=[]) ─────────────────────────────────────


def test_compare_kv_all_resumed_skips_model_load(monkeypatch, tmp_path):
    """When ALL configs have valid partials, _load_model is called 0 times."""
    rep_4 = _fid((4, 64), 0.09)
    rep_8 = _fid((8, 64), 0.01)

    (tmp_path / "4_64.json").write_text(
        json.dumps({"status": "ok", "report": dataclasses.asdict(rep_4), "cost": 1000})
    )
    (tmp_path / "8_64.json").write_text(
        json.dumps({"status": "ok", "report": dataclasses.asdict(rep_8), "cost": 2000})
    )

    load_calls: list[str] = []

    def fail_if_loaded(model_id: str, revision) -> object:
        load_calls.append(model_id)
        return object()

    def assert_not_scored(*a, **kw):  # type: ignore[return]
        raise AssertionError("score_kv_config should not be called when all configs are cached")

    monkeypatch.setattr(cmp, "install_memory_caps", lambda: (0, 0))
    monkeypatch.setattr(cmp, "_load_model", fail_if_loaded)
    monkeypatch.setattr(cmp, "_kv_dims", lambda model: (16, 8, 64))
    monkeypatch.setattr(cmp, "_load_corpus_for_kv", lambda m, mid, mc: object())
    monkeypatch.setattr(cmp, "score_kv_config", assert_not_scored)

    report = cmp.compare_kv_fidelity("m", [(4, 64), (8, 64)], artifacts_dir=tmp_path)

    # _load_model must not have been called
    assert load_calls == []

    # Report data must come from the pre-seeded partials
    labels = {r.label for r in report.results}
    assert labels == {"4:64", "8:64"}
    result_4 = next(r for r in report.results if r.label == "4:64")
    assert result_4.status == "ok"
    assert result_4.point is not None
    assert result_4.point.cost_bytes == 1000


# ── Fix 11: resume data integrity ─────────────────────────────────────────────


def test_compare_kv_resume_data_integrity(monkeypatch, tmp_path):
    """Rehydrated result carries the partial's exact data (cost and kl.mean)."""
    rep = _fid((4, 64), 0.09)
    seeded_cost = 1234
    seeded_kl_mean = 0.09

    (tmp_path / "4_64.json").write_text(
        json.dumps({"status": "ok", "report": dataclasses.asdict(rep), "cost": seeded_cost})
    )
    calls = _patch_kv_compare(monkeypatch, {(8, 64): _fid((8, 64), 0.01)})
    report = cmp.compare_kv_fidelity("m", [(4, 64), (8, 64)], artifacts_dir=tmp_path)

    # (4, 64) was resumed, (8, 64) was scored
    assert calls == [(8, 64)]

    result_4 = next(r for r in report.results if r.label == "4:64")
    assert result_4.point is not None
    assert result_4.point.cost_bytes == seeded_cost
    assert result_4.report is not None
    assert result_4.report.kl.mean == seeded_kl_mean


# ── Fix 2: defensive ValueError for non-numeric cost in envelope ───────────────


def test_kv_envelope_to_result_raises_on_bad_cost_type():
    """_kv_envelope_to_result raises ValueError when cost is a non-numeric type (fix 2).

    Under normal operation cost is always int|None, but a hand-edited or externally
    produced partial could carry a string — the assert was stripped under python -O,
    the ValueError fires unconditionally.

    This test calls _kv_envelope_to_result directly to assert the function itself raises.
    The collect-loop isolation behaviour (ValueError caught → failed result) is covered by
    test_compare_kv_collect_loop_isolates_bad_cost_partial.
    """
    rep = _fid((4, 64), 0.05)
    env = {"status": "ok", "report": dataclasses.asdict(rep), "cost": "not_a_number"}
    with pytest.raises(ValueError, match="unexpected cost type"):
        cmp._kv_envelope_to_result("4:64", env)


# ── Fix A: collect-loop isolates bad-cost partial instead of aborting ─────────


def test_compare_kv_collect_loop_isolates_bad_cost_partial(monkeypatch, tmp_path):
    """A valid-JSON partial with a non-numeric cost becomes a 'failed' result in the collect
    loop; the other config's result is unaffected and the overall run completes.

    This distinguishes the collect-loop path from _kv_envelope_to_result's own ValueError
    guard: the collect loop must CATCH ValueError and isolate it, not propagate it.
    """
    rep_4 = _fid((4, 64), 0.05)
    rep_8 = _fid((8, 64), 0.01)

    # Pre-seed 4_64.json with a bad cost so it's skipped in PENDING but fails at collect
    (tmp_path / "4_64.json").write_text(
        json.dumps({"status": "ok", "report": dataclasses.asdict(rep_4), "cost": "bad-cost"})
    )
    # Pre-seed 8_64.json with a valid partial so no model load occurs
    (tmp_path / "8_64.json").write_text(
        json.dumps({"status": "ok", "report": dataclasses.asdict(rep_8), "cost": 2000})
    )

    _patch_kv_compare(monkeypatch, {})  # no scoring should happen (both pre-seeded)

    # Must NOT raise — bad cost is isolated, not propagated
    report = cmp.compare_kv_fidelity("m", [(4, 64), (8, 64)], artifacts_dir=tmp_path)

    assert len(report.results) == 2

    failed = next(r for r in report.results if r.label == "4:64")
    assert failed.status == "failed"
    assert failed.error_type == "CorruptPartial"
    assert "4:64" in (failed.message or "")

    good = next(r for r in report.results if r.label == "8:64")
    assert good.status == "ok"
    assert "8:64" in report.frontier


# ── Fix B (KV): stale-reference resume recomputes instead of resuming ─────────


def test_compare_kv_stale_model_partial_is_recomputed(monkeypatch, tmp_path):
    """A partial measured against model 'A' is not resumed when compare is called with model 'B'.

    Pre-seed a valid ok partial for config (4,64) that records model_id='model-A'.
    Call compare_kv_fidelity with model_id='model-B' in the same artifacts_dir.
    Assert that score_kv_config is called for (4,64) — the stale partial is treated as absent.
    The existing-match resume test (test_compare_kv_resume_skips_existing_partial) uses
    the SAME model_id and must still pass.
    """
    rep_a = _fid((4, 64), 0.05)
    # report carries model_id='m'; write it as if it was for 'model-A'
    rep_dict = dataclasses.asdict(rep_a)
    rep_dict["model_id"] = "model-A"
    (tmp_path / "4_64.json").write_text(
        json.dumps({"status": "ok", "report": rep_dict, "cost": 1000})
    )

    # model-B call should recompute (4,64) and also score (8,64)
    reports = {(4, 64): _fid((4, 64), 0.07), (8, 64): _fid((8, 64), 0.01)}
    calls = _patch_kv_compare(monkeypatch, reports)

    cmp.compare_kv_fidelity("model-B", [(4, 64), (8, 64)], artifacts_dir=tmp_path)

    # (4,64) must have been re-scored — stale partial from model-A should not be resumed
    assert (4, 64) in calls


# ── Lines 376, 379: _read_partial returns None for non-ok / bad-report partials ──


def test_compare_kv_failed_partial_is_recomputed(monkeypatch, tmp_path):
    """A KV partial with status='failed' causes _read_partial to return None (line 376),
    putting that config in pending and re-scoring it.
    """
    # Pre-seed a failed partial for (4, 64)
    (tmp_path / "4_64.json").write_text(
        json.dumps({"status": "failed", "error_type": "CacheNotQuantizableError", "message": "x"})
    )
    reports = {(4, 64): _fid((4, 64), 0.09), (8, 64): _fid((8, 64), 0.01)}
    calls = _patch_kv_compare(monkeypatch, reports)
    cmp.compare_kv_fidelity("m", [(4, 64), (8, 64)], artifacts_dir=tmp_path)
    # (4, 64) must have been re-scored — the failed partial is not resumed
    assert (4, 64) in calls


def test_compare_kv_ok_partial_with_null_report_is_recomputed(monkeypatch, tmp_path):
    """A KV partial with status='ok' but report=null causes _read_partial to return None
    (line 379 — isinstance check fails), putting that config in pending and re-scoring it.
    """
    # Pre-seed a partial with ok status but report=null (not a dict)
    (tmp_path / "4_64.json").write_text(json.dumps({"status": "ok", "report": None, "cost": 100}))
    reports = {(4, 64): _fid((4, 64), 0.09), (8, 64): _fid((8, 64), 0.01)}
    calls = _patch_kv_compare(monkeypatch, reports)
    cmp.compare_kv_fidelity("m", [(4, 64), (8, 64)], artifacts_dir=tmp_path)
    # (4, 64) must have been re-scored — the null-report partial is not resumed
    assert (4, 64) in calls
