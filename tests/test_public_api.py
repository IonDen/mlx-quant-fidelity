import mlx_quant_fidelity as qf


def test_public_exports():
    assert hasattr(qf, "measure_weight_fidelity")
    assert hasattr(qf, "WeightFidelityReport")
    assert "measure_weight_fidelity" in qf.__all__
    assert "WeightFidelityReport" in qf.__all__
