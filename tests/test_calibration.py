"""Tests for static activation calibration and symmetric INT8 quantization per rule N3 (T3.3)."""

from pathlib import Path

import numpy as np
import pytest

from compiler.calibration import (
    calibrate_layer,
    compute_activation_scale,
    compute_activation_stat,
    dequantize_activations,
    quantize_activations,
)
from ir.linear import LinearOp


def test_absmax_calibration_and_quantization():
    """Verify absmax activation calibration scale and round-half-to-even quantization."""
    # Synthetic activations with known absmax = 2.54
    x = np.array([[-2.54, 0.0, 1.27], [2.54, -1.27, 0.02]], dtype=np.float32)

    stat = compute_activation_stat(x, method="absmax")
    assert np.isclose(stat, 2.54)

    scale = compute_activation_scale(stat)
    assert np.isclose(scale, 0.02)

    x_q = quantize_activations(x, scale)
    assert x_q.dtype == np.int8
    assert x_q.shape == (2, 3)

    # -2.54 / 0.02 = -127
    assert x_q[0, 0] == -127
    assert x_q[0, 1] == 0
    # 1.27 / 0.02 = 63.5 -> round to even: 64
    assert x_q[0, 2] == 64
    # 2.54 / 0.02 = 127
    assert x_q[1, 0] == 127
    # -1.27 / 0.02 = -63.5 -> round to even: -64
    assert x_q[1, 1] == -64

    # Range strictly [-127, 127], never -128
    assert np.all(x_q >= -127)
    assert np.all(x_q <= 127)
    assert not np.any(x_q == -128)


def test_percentile_calibration_and_clipping():
    """Verify percentile calibration handles outliers and clips to [-127, 127]."""
    rng = np.random.default_rng(123)
    # 10,000 normal values with std=1.0 (99.9% within ~3.29)
    normal_vals = rng.normal(0, 1.0, size=10000)
    # Add extreme outliers
    normal_vals[0] = 50.0
    normal_vals[1] = -60.0

    stat_absmax = compute_activation_stat(normal_vals, method="absmax")
    assert stat_absmax == 60.0

    stat_pct = compute_activation_stat(normal_vals, method="percentile", percentile=99.9)
    assert stat_pct < 5.0  # Much tighter scale than 60.0

    scale_pct = compute_activation_scale(stat_pct)
    x_q = quantize_activations(normal_vals, scale_pct)

    # Outliers must be clipped strictly to [-127, 127]
    assert x_q[0] == 127
    assert x_q[1] == -127
    assert np.all(x_q >= -127)
    assert np.all(x_q <= 127)
    assert not np.any(x_q == -128)


def test_zero_activation_fallback():
    """Verify all-zero activations fallback gracefully."""
    zeros = np.zeros((4, 4), dtype=np.float32)
    stat = compute_activation_stat(zeros, method="absmax")
    scale = compute_activation_scale(stat)
    assert scale == 1.0
    x_q = quantize_activations(zeros, scale)
    assert np.all(x_q == 0)


def test_dequantize_activations():
    """Verify dequantization arithmetic: x_float = x_q * scale."""
    x_q = np.array([-127, 0, 127], dtype=np.int8)
    scale = 0.05
    x_f = dequantize_activations(x_q, scale)
    assert np.allclose(x_f, [-6.35, 0.0, 6.35])


def test_calibrate_real_smollm_activations():
    """Verify activation calibration on real SmolLM2 captured activations."""
    act_path = (
        Path("artifacts")
        / "SmolLM2-135M@93efa2f097d5"
        / "layer_0"
        / "q_proj"
        / "calib"
        / "x_sample.npy"
    )
    if not act_path.exists():
        pytest.skip("SmolLM2 layer 0 x_sample artifact not available")

    x_act = np.load(act_path)
    stat1 = compute_activation_stat(x_act, method="absmax")
    scale1 = compute_activation_scale(stat1)
    x_q1 = quantize_activations(x_act, scale1)

    # Rerun determinism check
    stat2 = compute_activation_stat(x_act, method="absmax")
    scale2 = compute_activation_scale(stat2)
    x_q2 = quantize_activations(x_act, scale2)

    assert stat1 == stat2
    assert scale1 == scale2
    assert np.array_equal(x_q1, x_q2)
    assert np.all(x_q1 >= -127)
    assert np.all(x_q1 <= 127)
    assert not np.any(x_q1 == -128)


def test_calibrate_layer_pass():
    """Verify annotate LinearOp with calibration scale and provenance."""
    w = np.array([[3, 7, -2]], dtype=np.int8)
    op = LinearOp(name="test_calib_op", in_features=3, out_features=1, weight=w)

    x_calib = np.array([[1.0, -2.0, 1.5], [0.5, 0.2, -1.0]], dtype=np.float32)
    calibrated_op = calibrate_layer(
        op,
        x_calib,
        method="absmax",
        calibration_id="seed42_fixed_prompts",
    )

    assert calibrated_op.act_scale is not None
    assert calibrated_op.act_scale > 0.0
    assert calibrated_op.quant["act_method"] == "absmax"
    assert calibrated_op.quant["calibration_id"] == "seed42_fixed_prompts"
    assert "act_stat" in calibrated_op.quant
