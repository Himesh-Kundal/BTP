"""Tests for bias quantization into INT32 domain per rule N5 (T3.4)."""

import numpy as np
import pytest

from compiler.quantization import (
    dequantize_bias,
    quantize_bias,
    quantize_linear_op,
    quantize_weights,
)
from ir.linear import LinearOp
from ir.tensor import TensorKind
from verification.reference import linear_reference_int64


def test_quantize_bias_scaling_and_rounding():
    """Verify bias scaling sb = sx * sw and round-half-to-even."""
    s_x = 0.5
    s_w = np.array([0.5, 0.25, 0.5], dtype=np.float64)
    # Scales: [0.25, 0.125, 0.25] (exact in binary float64)
    # Biases:
    # 0.25 * 100 = 25.0 -> 100
    # 0.125 * 2.5 = 0.3125 -> 2.5 rounds to 2 (even)
    # 0.25 * -3.5 = -0.875 -> -3.5 rounds to -4 (even)
    b_float = np.array([25.0, 0.3125, -0.875], dtype=np.float64)

    b_q = quantize_bias(b_float, s_x=s_x, s_w=s_w)

    assert b_q.dtype == np.int32
    assert b_q.shape == (3,)
    assert b_q[0] == 100
    assert b_q[1] == 2
    assert b_q[2] == -4


def test_quantize_bias_clipping():
    """Verify bias values exceeding 32-bit integer range are clipped."""
    s_x = 1.0
    s_w = np.array([1.0, 1.0], dtype=np.float64)
    # Huge positive and huge negative bias
    b_float = np.array([3e9, -3e9], dtype=np.float64)

    b_q = quantize_bias(b_float, s_x=s_x, s_w=s_w)
    assert b_q[0] == 2147483647  # 2^31 - 1
    assert b_q[1] == -2147483648  # -2^31


def test_quantize_bias_shape_mismatch():
    """Mismatched bias shape raises ValueError."""
    s_x = 0.1
    s_w = np.array([0.01, 0.02], dtype=np.float64)
    b_float = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    with pytest.raises(ValueError, match="shape"):
        quantize_bias(b_float, s_x=s_x, s_w=s_w)


def test_dequantize_bias():
    """Verify dequantize_bias: b_float = b_q * (s_x * s_w)."""
    s_x = 0.05
    s_w = np.array([0.02, 0.04], dtype=np.float64)
    b_q = np.array([1000, -2000], dtype=np.int32)

    b_float = dequantize_bias(b_q, s_x=s_x, s_w=s_w)
    expected = np.array([1.0, -4.0], dtype=np.float64)
    assert np.allclose(b_float, expected)


def test_synthetic_bias_integration_with_reference():
    """Verify synthetic bias execution against independent integer reference oracle."""
    # Synthetic 2x3 layer with synthetic bias
    w_float = np.array([[0.3, 0.7, -0.2], [0.1, -0.4, 0.5]], dtype=np.float64)
    b_float = np.array([0.05, -0.08], dtype=np.float64)
    x_float = np.array([[1.2, -0.5, 0.8], [-0.3, 1.0, -1.2]], dtype=np.float64)

    # Quantize weights
    w_q, s_w = quantize_weights(w_float)
    s_x = 0.02

    # Quantize activations and bias
    from compiler.calibration import quantize_activations

    x_q = quantize_activations(x_float, s_x)
    b_q = quantize_bias(b_float, s_x=s_x, s_w=s_w)

    # Reference integer execution: Y = X_q @ W_q.T + b_q
    y_ref = linear_reference_int64(x_q, w_q, b_q)
    assert y_ref.shape == (2, 2)
    assert y_ref.dtype == np.int64

    # Direct integer verification
    expected_row0 = np.dot(x_q[0].astype(np.int64), w_q.T.astype(np.int64)) + b_q.astype(np.int64)
    assert np.array_equal(y_ref[0], expected_row0)


def test_quantize_linear_op_with_synthetic_bias():
    """Verify quantize_linear_op converts unquantized LinearOp with float bias to int32 bias."""
    w_float = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
    b_float = np.array([0.5], dtype=np.float32)

    raw_op = LinearOp(
        name="synthetic_bias_op",
        in_features=3,
        out_features=1,
        weight=w_float,
        bias=b_float,
        act_scale=0.05,
        is_quantized=False,
    )

    quant_op = quantize_linear_op(raw_op)
    assert quant_op.is_quantized
    assert quant_op.bias is not None
    assert quant_op.bias.dtype == "int32"
    assert quant_op.bias.kind == TensorKind.CONSTANT
    assert quant_op.bias.data is not None
    assert quant_op.bias.data.dtype == np.int32
