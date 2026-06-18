"""mlx-quant-fidelity — measure MLX quantization quality loss."""

from mlx_quant_fidelity._version import __version__
from mlx_quant_fidelity.probes.kv import measure_kv_fidelity
from mlx_quant_fidelity.probes.weights import measure_weight_fidelity
from mlx_quant_fidelity.report import ComparisonReport, FidelityReport, WeightFidelityReport
from mlx_quant_fidelity.runners.compare import compare_kv_fidelity, compare_weight_fidelity

__all__ = [
    "ComparisonReport",
    "FidelityReport",
    "WeightFidelityReport",
    "__version__",
    "compare_kv_fidelity",
    "compare_weight_fidelity",
    "measure_kv_fidelity",
    "measure_weight_fidelity",
]
