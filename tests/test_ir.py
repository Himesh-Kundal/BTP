"""Tests for IR data structures and invariants (I1, I2, I5)."""

import numpy as np
import pytest

from ir.linear import LinearOp


def test_linear_op_valid():
    """Verify standard creation of valid LinearOp."""
    w = np.array([[3, 7, -2], [1, 0, -1]], dtype=np.int8)  # N=2, K=3
    b = np.array([10, -5], dtype=np.int32)
    op = LinearOp(
        name="toy_linear",
        in_features=3,
        out_features=2,
        weight=w,
        bias=b,
    )
    assert op.name == "toy_linear"
    assert op.in_features == 3
    assert op.out_features == 2
    assert op.weight.shape == (2, 3)
    assert op.bias is not None and op.bias.shape == (2,)


def test_invariant_i1_shape_mismatches():
    """Invariant I1: Reject shape inconsistencies."""
    w = np.array([[3, 7, -2]], dtype=np.int8)  # N=1, K=3

    # Mismatch in in_features
    with pytest.raises(ValueError, match="in_features"):
        LinearOp(name="bad_k", in_features=4, out_features=1, weight=w)

    # Mismatch in out_features
    with pytest.raises(ValueError, match="out_features"):
        LinearOp(name="bad_n", in_features=3, out_features=2, weight=w)

    # Mismatch in bias shape
    bad_bias = np.array([1, 2], dtype=np.int32)  # shape (2,) for N=1
    with pytest.raises(ValueError, match="bias shape"):
        LinearOp(name="bad_bias", in_features=3, out_features=1, weight=w, bias=bad_bias)

    # Mismatch in input_shape
    with pytest.raises(ValueError, match="input_shape"):
        LinearOp(name="bad_in_shape", in_features=3, out_features=1, weight=w, input_shape=(1, 5))


def test_invariant_i2_weight_range():
    """Invariant I2: Weights must be strictly within [-127, 127]."""
    # -128 is NOT allowed for quantized weights per N2
    w_under = np.array([[-128, 0, 1]], dtype=np.int8)
    with pytest.raises(ValueError, match="outside valid range"):
        LinearOp(name="w_under", in_features=3, out_features=1, weight=w_under)

    # Valid boundary values: -127 and 127
    w_valid = np.array([[-127, 0, 127]], dtype=np.int8)
    op = LinearOp(name="w_bound", in_features=3, out_features=1, weight=w_valid)
    assert op.weight[0, 0] == -127
    assert op.weight[0, 2] == 127


def test_invariant_i2_weight_dtype():
    """Weights must have int8 dtype."""
    w_float = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(TypeError, match="int8"):
        LinearOp(name="w_float", in_features=3, out_features=1, weight=w_float)  # type: ignore[arg-type]
