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
    """Wired-memory caps could not be installed before a model load.

    Reserved: not raised by the current release. When caps cannot be installed on a device
    that reports a working set, the run continues and the report carries a warning instead.
    """


class ModelMismatchError(QuantFidelityError):
    """The quant and reference repos are not a comparable pair (architecture / vocab / not quantized)."""


class InsufficientMemoryError(QuantFidelityError):
    """The two models' combined size exceeds the device's recommended working set."""


class CompareConfigError(QuantFidelityError, ValueError):
    """Invalid `compare` invocation arguments (target/config count, duplicates, malformed ids).

    Subclasses ValueError too, preserving the documented `Raises: ValueError` contract so
    existing callers that catch ValueError around the public compare functions keep working.
    """


class ReportSchemaError(QuantFidelityError):
    """A persisted report dict is structurally malformed (missing key / wrong type)."""


class QuantizeStartError(QuantFidelityError, ValueError):
    """Invalid deployment boundary (`quantize_start`) for the corpus window.

    Subclasses ValueError too, preserving the compare path's documented ValueError
    contract (matching CompareConfigError).
    """


class MethodUnavailableError(QuantFidelityError):
    """A third-party KV-cache method's package is missing, is the wrong package, or is incompatible."""


class LogitsBudgetError(CorpusError):
    """The per-chunk paired fp32 logits (plus method + control working-set bytes) exceed the cap.

    A CorpusError subclass so existing ``except CorpusError`` callers keep catching it; the
    pre-flight gate raises this specific type so a caller can tell an over-budget refusal
    apart from other corpus problems.
    """


class NonFiniteMetricError(QuantFidelityError):
    """A measured metric came back NaN (or a perplexity input non-finite).

    A NaN would sail past every threshold comparison and could be ranked or recommended as
    if it were a real measurement, so the probe refuses it. A +inf KL is different: it is
    the documented zero-probability policy and stays legal.
    """


class UntrustedModelCodeError(QuantFidelityError):
    """The repo's config.json names its own model code, which mlx-lm would execute on load."""


class ModelNotAccessibleError(QuantFidelityError):
    """A model's config.json could not be fetched or read (typo, gated repo, offline, bad path).

    The message names the model id that was being read, so a two-repo command points at the
    right repo.
    """
