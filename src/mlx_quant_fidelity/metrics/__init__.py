"""Pure, stream-friendly fidelity metrics (mx.array in -> per-position scalars out)."""

from mlx_quant_fidelity.metrics.flip import top_token_flips
from mlx_quant_fidelity.metrics.kl import kl_divergence

__all__ = ["kl_divergence", "top_token_flips"]
