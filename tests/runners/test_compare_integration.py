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


@pytest.mark.slow
@pytest.mark.network
def test_compare_kv_two_configs_real_model(tmp_path):
    report = compare_kv_fidelity(_MODEL, [(4, 64), (8, 64)], max_chunks=2, artifacts_dir=tmp_path)
    assert report.mode == "kv"
    assert report.model == _MODEL
    assert len(report.frontier) >= 1
    assert all(r.status in ("ok", "failed") for r in report.results)


@pytest.mark.slow
@pytest.mark.network
def test_compare_weight_two_targets_real_model(tmp_path):
    report = compare_weight_fidelity([_MODEL, _Q8], _REF, max_chunks=2, artifacts_dir=tmp_path)
    assert report.mode == "weight"
    assert report.reference == _REF
    assert {r.label for r in report.results} == {_MODEL, _Q8}
    assert len(report.frontier) >= 1
    assert all(r.status in ("ok", "failed") for r in report.results)
