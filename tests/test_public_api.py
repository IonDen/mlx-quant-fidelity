import mlx_quant_fidelity as qf


def test_public_exports():
    # the flagship KV API
    assert hasattr(qf, "measure_kv_fidelity")
    assert hasattr(qf, "FidelityReport")
    assert "measure_kv_fidelity" in qf.__all__
    assert "FidelityReport" in qf.__all__
    # the weight API
    assert hasattr(qf, "measure_weight_fidelity")
    assert hasattr(qf, "WeightFidelityReport")
    assert "measure_weight_fidelity" in qf.__all__
    assert "WeightFidelityReport" in qf.__all__


def test_compare_public_exports():
    assert hasattr(qf, "compare_weight_fidelity")
    assert hasattr(qf, "compare_kv_fidelity")
    assert hasattr(qf, "ComparisonReport")
    assert "compare_weight_fidelity" in qf.__all__
    assert "compare_kv_fidelity" in qf.__all__
    assert "ComparisonReport" in qf.__all__
