"""Pure geometry helpers over hand-built module trees at explicit dtypes.

Every fixture is arithmetic-first: the expected numbers are derived by hand from the storage
layout (uint32-packed weights, fp16 scales and biases per group), never from running the code.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_quant_fidelity.probes.weights import bits_per_weight, measured_geometry


class _Quant(nn.Module):
    """A 4-bit, group-32 quantized [8 out x 64 in] linear: weight uint32[8, 8] (256 B),
    scales fp16[8, 2] (32 B), biases fp16[8, 2] (32 B) = 320 B = 2560 bits over 512 params."""

    def __init__(self, bits: int = 4, group_size: int = 32) -> None:
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.weight = mx.zeros((8, 64 * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((8, 64 // group_size), dtype=mx.float16)
        self.biases = mx.zeros((8, 64 // group_size), dtype=mx.float16)


class _Dense(nn.Module):
    """An unquantized fp32 [4 x 4] linear (64 B, 16 params) that IS quantizable."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = mx.zeros((4, 4), dtype=mx.float32)

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> nn.Module:  # pragma: no cover
        raise NotImplementedError


class _Norm(nn.Module):
    """A parameterised module with NO to_quantized (a norm) — never quantizable."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = mx.ones((4,), dtype=mx.float32)


class _Tree(nn.Module):
    def __init__(self, **children: nn.Module) -> None:
        super().__init__()
        for name, child in children.items():
            setattr(self, name, child)


def test_bits_per_weight_single_quantized_module_is_exactly_5():
    """Reds if scale/bias bytes are dropped (-> 4.0) or packed uint32 words are counted as one
    param each (-> 40.0)."""
    assert bits_per_weight(_Tree(q=_Quant())) == 5.0


def test_bits_per_weight_counts_unquantized_params_in_the_denominator():
    """Reds if unquantized modules are excluded: (2560 + 512) / (512 + 16)."""
    assert math.isclose(bits_per_weight(_Tree(q=_Quant(), d=_Dense())), 3072 / 528, rel_tol=1e-9)


def test_bits_per_weight_numerator_covers_arrays_on_non_leaf_modules():
    """An array living directly on a parent module (the mamba `A_log` shape) counts as bytes but
    not as leaf params, exactly like mlx-lm's tree_reduce numerator. A leaves-only numerator
    reds here with 3072 / 528."""
    root = _Tree(q=_Quant(), d=_Dense())
    root.A_log = mx.zeros((16,), dtype=mx.float32)  # 64 B on a non-leaf module
    assert math.isclose(bits_per_weight(root), 3584 / 528, rel_tol=1e-9)


def test_bits_per_weight_rejects_a_parameterless_model():
    """Reds if a zero denominator is silently returned as inf/None instead of raising."""
    with pytest.raises(ValueError, match="no parameters"):
        bits_per_weight(_Tree())


def test_measured_geometry_groups_by_bits_and_group_size_and_counts_full_precision():
    """Reds if geometry is keyed by bits only (loses the group-32 module), if a norm is counted
    as full precision, or if the output is not sorted by (bits, group_size)."""
    tree = _Tree(
        a=_Quant(4, 64), b=_Quant(4, 64), c=_Quant(5, 64), e=_Quant(4, 32), d=_Dense(), n=_Norm()
    )
    geometry, n_full = measured_geometry(tree)
    assert geometry == ((4, 32, 1), (4, 64, 2), (5, 64, 1))
    assert n_full == 1


def test_measured_geometry_is_uniform_when_all_modules_share_one_bit_width():
    """The AWQ-shaped case: a dummy top-level nominal must not leak in — only the modules'
    own bits count. Reds if anything but the modules is consulted."""
    geometry, n_full = measured_geometry(_Tree(a=_Quant(8, 64), b=_Quant(8, 64)))
    assert geometry == ((8, 64, 2),)
    assert n_full == 0
