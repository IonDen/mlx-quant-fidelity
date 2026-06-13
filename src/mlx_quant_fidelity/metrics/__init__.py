"""Pure, stream-friendly fidelity metrics (mx.array in -> per-position scalars out)."""

from mlx_quant_fidelity.metrics.flip import top_token_flips
from mlx_quant_fidelity.metrics.kl import kl_divergence
from mlx_quant_fidelity.metrics.perplexity import perplexity, perplexity_delta, token_nll

__all__ = ["kl_divergence", "perplexity", "perplexity_delta", "token_nll", "top_token_flips"]
