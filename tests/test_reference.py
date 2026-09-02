"""Tests for the independent integer reference oracle (N1-N7).

These tests check:
1. Exact int64 linear layer arithmetic (Y = X @ W.T + b_q)
2. Hand-calculated cases including the 1x3 [3, 7, -2] example
3. Input range coverage: -128 to 127
4. Weight range coverage: -127 to 127
5. Symmetric per-channel weight quantization with half-to-even rounding
6. Symmetric per-tensor activation quantization
7. Bias quantization
8. Dequantization
"""

import numpy as np
import pytest

from verification.reference import (
    dequantize,
    linear_reference_int64,
    quantize_act_sym,
    quantize_bias,
    quantize_weight_sym,
)


def test_hand_calculated_1x3_toy():
    """Case 1: Hand-calculated 1x3 toy layer with weights [3, 7, -2], zero bias."""
    W = np.array([[3, 7, -2]], dtype=np.int8)  # shape (1, 3)
    X = np.array([[10, -4, 5]], dtype=np.int8)  # shape (1, 3)
    # Expected: 10*3 + (-4)*7 + 5*(-2) = 30 - 28 - 10 = -8
    Y = linear_reference_int64(X, W)
    assert Y.shape == (1, 1)
    assert Y[0, 0] == -8
    assert Y.dtype == np.int64


def test_hand_calculated_1x3_with_bias():
    """Case 2: 1x3 toy layer with weights [3, 7, -2] and non-zero bias."""
    W = np.array([[3, 7, -2]], dtype=np.int8)
    X = np.array([[-1, 2, -3]], dtype=np.int8)
    b = np.array([15], dtype=np.int32)
    # Expected: (-1)*3 + 2*7 + (-3)*(-2) + 15 = -3 + 14 + 6 + 15 = 32
    Y = linear_reference_int64(X, W, b)
    assert Y[0, 0] == 32


def test_extreme_negative_input():
    """Case 3: Extreme negative input -128 with -127 weight."""
    W = np.array([[-127]], dtype=np.int8)
    X = np.array([[-128]], dtype=np.int8)
    # Expected: (-128) * (-127) = +16256
    Y = linear_reference_int64(X, W)
    assert Y[0, 0] == 16256


def test_extreme_positive_input():
    """Case 4: Extreme positive input 127 with 127 weight."""
    W = np.array([[127]], dtype=np.int8)
    X = np.array([[127]], dtype=np.int8)
    # Expected: 127 * 127 = 16129
    Y = linear_reference_int64(X, W)
    assert Y[0, 0] == 16129


def test_sign_aligned_worst_case_accumulation():
    """Case 5: Sign-aligned worst-case accumulation across 4 lanes."""
    W = np.array([[127, -127, 127, -127]], dtype=np.int8)
    X = np.array([[-128, 127, -128, 127]], dtype=np.int8)
    # Expected: (-128)*127 + 127*(-127) + (-128)*127 + 127*(-127)
    # = -16256 - 16129 - 16256 - 16129 = -64770
    Y = linear_reference_int64(X, W)
    assert Y[0, 0] == -64770


def test_matrix_2x2():
    """Case 6: Multi-row, multi-column 2x2 matrix with bias."""
    X = np.array([[1, 2], [3, 4]], dtype=np.int8)
    W = np.array([[5, 6], [7, 8]], dtype=np.int8)
    b = np.array([10, 20], dtype=np.int32)
    # Row 0: [1*5 + 2*6 + 10, 1*7 + 2*8 + 20] = [17 + 10, 23 + 20] = [27, 43]
    # Row 1: [3*5 + 4*6 + 10, 3*7 + 4*8 + 20] = [39 + 10, 53 + 20] = [49, 73]
    Y = linear_reference_int64(X, W, b)
    np.testing.assert_array_equal(Y, np.array([[27, 43], [49, 73]], dtype=np.int64))


def test_quantize_weight_symmetric_round_half_even():
    """Case 7: Weight per-channel symmetric quantization and half-to-even rounding."""
    # Row 0: max|w| = 2.54 -> scale = 0.02.
    # Elements: [0.0, 1.27, -2.54] / 0.02 = [0.0, 63.5, -127.0]
    # 63.5 with round-half-to-even rounds to 64 (even integer).
    # Row 1: all zeros -> scale must be 1.0, quantized to 0.
    W_float = np.array([
        [0.0, 1.27, -2.54],
        [0.0, 0.0, 0.0],
    ], dtype=np.float64)

    W_q, s_w = quantize_weight_sym(W_float)

    assert W_q.dtype == np.int8
    np.testing.assert_allclose(s_w, [0.02, 1.0], rtol=1e-7)
    np.testing.assert_array_equal(W_q[0], np.array([0, 64, -127], dtype=np.int8))
    np.testing.assert_array_equal(W_q[1], np.array([0, 0, 0], dtype=np.int8))


def test_quantize_activation_symmetric_and_clipping():
    """Case 8: Per-tensor activation symmetric quantization with clipping."""
    # stat = 5.08 -> s_x = 5.08 / 127 = 0.04
    # -5.08 / 0.04 = -127.0
    # 2.54 / 0.04 = 63.5 -> 64
    # 10.0 / 0.04 = 250.0 -> clipped to 127
    # -10.0 / 0.04 = -250.0 -> clipped to -127 (quantizer range is [-127, 127] per N2/N3)
    X_float = np.array([[-5.08, 0.0, 2.54, 10.0, -10.0]], dtype=np.float64)
    stat = 5.08
    X_q, s_x = quantize_act_sym(X_float, stat)

    assert X_q.dtype == np.int8
    assert pytest.approx(s_x, rel=1e-7) == 0.04
    np.testing.assert_array_equal(X_q[0], np.array([-127, 0, 64, 127, -127], dtype=np.int8))


def test_quantize_bias_and_dequantize():
    """Case 9: Bias quantization and end-to-end dequantization."""
    b_float = np.array([0.1016], dtype=np.float64)
    s_x = 0.04
    s_w = np.array([0.02], dtype=np.float64)
    # s_x * s_w = 0.0008
    # 0.1016 / 0.0008 = 127.0 -> b_q = 127
    b_q = quantize_bias(b_float, s_x, s_w)
    assert b_q.dtype == np.int32
    assert b_q[0] == 127

    # Dequantization: Y = [[-8]], s_x = 0.04, s_w = [0.02]
    # y_float = -8 * 0.04 * 0.02 = -0.0064
    Y_int32 = np.array([[-8]], dtype=np.int32)
    Y_dequant = dequantize(Y_int32, s_x, s_w)
    np.testing.assert_allclose(Y_dequant, np.array([[-0.0064]]), rtol=1e-7)
