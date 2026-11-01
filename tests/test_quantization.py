"""Property and unit tests for symmetric INT8 weight quantization per rule N2 (T3.2)."""

from pathlib import Path

import numpy as np
import pytest

from compiler.quantization import (
    dequantize_weights,
    quantize_linear_op,
    quantize_weights,
)
from ir.linear import load_linear_from_artifact


def test_quantize_weights_basic_and_properties():
    """Verify basic weight quantization and mathematical properties."""
    # Synthetic matrix [3, 4]
    w = np.array(
        [
            [1.27, -0.635, 0.0, 0.635],
            [10.0, -10.0, 5.0, -5.0],
            [0.0, 0.0, 0.0, 0.0],  # Zero-row case
        ],
        dtype=np.float64,
    )

    w_q, s_w = quantize_weights(w)

    # Check shapes and dtypes
    assert w_q.shape == (3, 4)
    assert w_q.dtype == np.int8
    assert s_w.shape == (3,)
    assert s_w.dtype == np.float64

    # Row 0: max_abs = 1.27, scale = 1.27 / 127 = 0.01
    assert np.isclose(s_w[0], 0.01)
    assert w_q[0, 0] == 127
    assert w_q[0, 1] == -64  # -0.635 / 0.01 = -63.5 -> round to even: -64
    assert w_q[0, 2] == 0
    assert w_q[0, 3] == 64  # 0.635 / 0.01 = 63.5 -> round to even: 64

    # Row 1: max_abs = 10.0, scale = 10.0 / 127
    assert np.isclose(s_w[1], 10.0 / 127.0)
    assert w_q[1, 0] == 127
    assert w_q[1, 1] == -127

    # Row 2 (zero-row case per rule N2): scale must be 1.0, row must be 0
    assert s_w[2] == 1.0
    assert np.all(w_q[2, :] == 0)

    # Invariant I2: values strictly in [-127, 127]
    assert np.all(w_q >= -127)
    assert np.all(w_q <= 127)
    assert not np.any(w_q == -128)


def test_quantization_error_bound_property():
    """Property test: |W - s_w * W_q| <= s_w / 2 for every element."""
    rng = np.random.default_rng(42)
    # Random matrices with various dynamic ranges
    for _ in range(10):
        scale_target = rng.uniform(0.01, 100.0)
        w = rng.normal(0, scale_target, size=(8, 16)).astype(np.float64)

        w_q, s_w = quantize_weights(w)

        # Dequantize
        w_dequant = dequantize_weights(w_q, s_w)

        # Absolute error per element
        abs_err = np.abs(w - w_dequant)
        max_allowed_err = (s_w[:, np.newaxis] / 2.0) + 1e-12

        assert np.all(abs_err <= max_allowed_err)


def test_round_half_to_even_behavior():
    """Verify round-half-to-even behaviour on exact half-integers."""
    # Scale = 1.0 (so max_abs = 127)
    # Values that scale to: 0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5
    row = np.array([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 127.0], dtype=np.float64)
    w = row.reshape(1, -1)

    w_q, s_w = quantize_weights(w)
    assert np.isclose(s_w[0], 1.0)

    # 0.5 -> 0 (even)
    assert w_q[0, 0] == 0
    # 1.5 -> 2 (even)
    assert w_q[0, 1] == 2
    # 2.5 -> 2 (even)
    assert w_q[0, 2] == 2
    # 3.5 -> 4 (even)
    assert w_q[0, 3] == 4
    # -0.5 -> 0 (even)
    assert w_q[0, 4] == 0
    # -1.5 -> -2 (even)
    assert w_q[0, 5] == -2
    # -2.5 -> -2 (even)
    assert w_q[0, 6] == -2


def test_invalid_dimension_raises():
    """Weight must be 2D."""
    with pytest.raises(ValueError, match="2D"):
        quantize_weights(np.array([1.0, 2.0, 3.0]))


def test_quantize_smollm_layer0_weights():
    """Verify quantization on real SmolLM2 layer 0 weights."""
    weight_path = (
        Path("artifacts")
        / "SmolLM2-135M@93efa2f097d5"
        / "layer_0"
        / "q_proj"
        / "weight.npy"
    )
    if not weight_path.exists():
        pytest.skip("SmolLM2 layer 0 weight artifact not available")

    w_float = np.load(weight_path)
    w_q, s_w = quantize_weights(w_float)

    assert w_q.shape == (576, 576)
    assert w_q.dtype == np.int8
    assert np.all(w_q >= -127)
    assert np.all(w_q <= 127)
    assert not np.any(w_q == -128)
    assert np.all(s_w > 0.0)


def test_quantize_linear_op_pass():
    """Verify quantize_linear_op converts unquantized LinearOp to quantized IR."""
    artifact_dir = (
        Path("artifacts")
        / "SmolLM2-135M@93efa2f097d5"
        / "layer_0"
        / "q_proj"
    )
    if not artifact_dir.exists():
        pytest.skip("SmolLM2 layer 0 artifact not available")

    raw_op = load_linear_from_artifact(artifact_dir)
    assert not raw_op.is_quantized
    assert raw_op.weight.dtype == "float32"

    quant_op = quantize_linear_op(raw_op)
    assert quant_op.is_quantized
    assert quant_op.weight.dtype == "int8"
    assert quant_op.weight_scale is not None
    assert quant_op.weight_scale.shape == (576,)
    assert quant_op.quant["scheme"] == "symmetric"
    assert quant_op.quant["rounding"] == "round_half_to_even"
