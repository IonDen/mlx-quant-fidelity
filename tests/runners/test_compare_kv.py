"""Tests for compare_kv_fidelity: one-load loop, resume, isolation, validation guards."""

import dataclasses
import json

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
    assert (4, 64) in calls  # corrupt partial triggered a re-run
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
    """_load_model is called exactly once per compare_kv_fidelity call."""
    reports = {(4, 64): _fid((4, 64), 0.09), (8, 64): _fid((8, 64), 0.01)}
    load_calls: list[str] = []

    def fake_load(model_id: str, revision) -> object:
        load_calls.append(model_id)
        return object()

    monkeypatch.setattr(cmp, "install_memory_caps", lambda: (0, 0))
    monkeypatch.setattr(cmp, "_load_model", fake_load)
    monkeypatch.setattr(cmp, "_kv_dims", lambda model: (16, 8, 64))

    def fake_score(model, corpus, *, kv_bits, kv_group_size, **kw):  # type: ignore[return]
        return reports[(kv_bits, kv_group_size)]

    monkeypatch.setattr(cmp, "score_kv_config", fake_score)
    monkeypatch.setattr(cmp, "_load_corpus_for_kv", lambda model, model_id, max_chunks: object())
    cmp.compare_kv_fidelity("m", [(4, 64), (8, 64)], artifacts_dir=tmp_path)
    assert len(load_calls) == 1
