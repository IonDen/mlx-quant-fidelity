import pytest

from mlx_quant_fidelity.errors import (
    CacheNotQuantizableError,
    ExactZeroError,
    QuantFidelityError,
)


def test_subclasses_share_base():
    assert issubclass(CacheNotQuantizableError, QuantFidelityError)
    assert issubclass(ExactZeroError, QuantFidelityError)


def test_message_preserved():
    with pytest.raises(QuantFidelityError, match="boom"):
        raise CacheNotQuantizableError("boom")
