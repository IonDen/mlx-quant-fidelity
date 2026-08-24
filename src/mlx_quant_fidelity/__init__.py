"""mlx-quant-fidelity — measure MLX quantization quality loss."""

from mlx_quant_fidelity._version import __version__
from mlx_quant_fidelity.badge import badge_color, badge_for_report, render_badge_markdown
from mlx_quant_fidelity.errors import MethodUnavailableError
from mlx_quant_fidelity.probes.kv import measure_kv_fidelity
from mlx_quant_fidelity.probes.kv_methods import (
    KVCacheMethod,
    StockKVMethod,
    TurboQuantKVMethod,
    parse_method_spec,
)
from mlx_quant_fidelity.probes.weights import measure_weight_fidelity
from mlx_quant_fidelity.report import ComparisonReport, FidelityReport, WeightFidelityReport
from mlx_quant_fidelity.runners.compare import compare_kv_fidelity, compare_weight_fidelity

__all__ = [
    "ComparisonReport",
    "FidelityReport",
    "KVCacheMethod",
    "MethodUnavailableError",
    "StockKVMethod",
    "TurboQuantKVMethod",
    "WeightFidelityReport",
    "__version__",
    "badge_color",
    "badge_for_report",
    "compare_kv_fidelity",
    "compare_weight_fidelity",
    "measure_kv_fidelity",
    "measure_weight_fidelity",
    "parse_method_spec",
    "render_badge_markdown",
]
