"""Package-rooted exceptions. Catch QuantFidelityError to catch all of them."""


class QuantFidelityError(Exception):
    """Base class for all mlx-quant-fidelity errors."""


class CacheNotQuantizableError(QuantFidelityError):
    """The model's KV cache does not support quantization (no working to_quantized)."""


class ExactZeroError(QuantFidelityError):
    """KLD and flip were exactly zero where quantization was expected to engage."""


class CorpusError(QuantFidelityError):
    """The evaluation corpus could not be loaded or tokenized."""


class MemorySafetyError(QuantFidelityError):
    """Wired-memory caps could not be installed before a model load."""


class ModelMismatchError(QuantFidelityError):
    """The quant and reference repos are not a comparable pair (architecture / vocab / not quantized)."""


class InsufficientMemoryError(QuantFidelityError):
    """The two models' combined size exceeds the device's recommended working set."""
