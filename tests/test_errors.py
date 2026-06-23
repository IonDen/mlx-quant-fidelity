import pytest

from mlx_quant_fidelity.errors import (
    CacheNotQuantizableError,
    ExactZeroError,
    InsufficientMemoryError,
    ModelMismatchError,
    QuantFidelityError,
)


def test_subclasses_share_base():
    assert issubclass(CacheNotQuantizableError, QuantFidelityError)
    assert issubclass(ExactZeroError, QuantFidelityError)


def test_message_preserved():
    with pytest.raises(QuantFidelityError, match="boom"):
        raise CacheNotQuantizableError("boom")


def test_new_errors_are_quant_fidelity_errors():
    assert issubclass(ModelMismatchError, QuantFidelityError)
    assert issubclass(InsufficientMemoryError, QuantFidelityError)


def test_compare_config_error_is_quant_fidelity_error():
    from mlx_quant_fidelity.errors import CompareConfigError, QuantFidelityError

    assert issubclass(CompareConfigError, QuantFidelityError)
    assert issubclass(CompareConfigError, ValueError)  # documented back-compat contract
