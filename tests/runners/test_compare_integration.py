"""Real-model integration tests for compare_kv_fidelity and compare_weight_fidelity.

Both tests are ``@pytest.mark.slow`` and ``@pytest.mark.network`` — they require
downloading real HuggingFace model repos and running full probes. Skipped by default;
run with ``--run-slow --run-network`` on the MAIN THREAD only.
DO NOT run in CI or in a subagent.
"""

import pytest

from mlx_quant_fidelity.runners.compare import compare_kv_fidelity, compare_weight_fidelity

_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
_REF = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
_Q8 = "mlx-community/Qwen2.5-0.5B-Instruct-8bit"
# Cached pins, so both tests resolve offline: an unpinned `main` needs a cached refs/main.
_MODEL_REVISION = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
_REF_REVISION = "56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
_Q8_REVISION = "6474d4c1a05bd7bd634a9c993560e4e7272eda3d"


@pytest.mark.slow
@pytest.mark.network
def test_compare_kv_two_configs_real_model(tmp_path):
    report = compare_kv_fidelity(
        _MODEL,
        [(4, 64), (8, 64)],
        max_chunks=2,
        artifacts_dir=tmp_path,
        model_revision=_MODEL_REVISION,
    )
    assert report.mode == "kv"
    assert report.model == _MODEL
    assert len(report.frontier) >= 1
    assert [r.status for r in report.results] == ["ok", "ok"]
    assert {r.label for r in report.results} == {"4:64", "8:64"}


@pytest.mark.slow
@pytest.mark.network
def test_compare_weight_two_targets_real_model(tmp_path):
    # The two targets need different pins, and `quant_revision` applies to every target, so each
    # target carries its own inline `repo@revision` (the label stays the bare repo id).
    report = compare_weight_fidelity(
        [f"{_MODEL}@{_MODEL_REVISION}", f"{_Q8}@{_Q8_REVISION}"],
        _REF,
        max_chunks=2,
        artifacts_dir=tmp_path,
        reference_revision=_REF_REVISION,
    )
    assert report.mode == "weight"
    assert report.reference == _REF
    assert {r.label for r in report.results} == {_MODEL, _Q8}
    assert len(report.frontier) >= 1
    assert [r.status for r in report.results] == ["ok", "ok"]
