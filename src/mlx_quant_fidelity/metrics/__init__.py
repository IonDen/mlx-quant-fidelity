"""Pure, stream-friendly fidelity metrics (mx.array in -> per-position scalars out)."""

from mlx_quant_fidelity.metrics.kl import kl_divergence

__all__ = ["kl_divergence"]
