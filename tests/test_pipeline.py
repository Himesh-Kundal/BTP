"""Tests for the end-to-end compiler pipeline driver (T3.7)."""

from pathlib import Path

import numpy as np
import pytest

from compiler.pipeline import compile_layer_from_artifact, compile_linear
from ir.linear import LinearOp
from ir.tensor import TensorKind


def test_compile_linear_synthetic_op():
    """Verify compile_linear pipeline on synthetic unquantized LinearOp."""
    w_float = np.array([[0.5, -1.2, 2.5], [1.0, 0.0, -2.0]], dtype=np.float32)  # N=2, K=3
    b_float = np.array([0.1, -0.2], dtype=np.float32)
    x_calib = np.array([[1.0, -0.5, 0.8], [0.2, 0.4, -1.5]], dtype=np.float32)

    raw_op = LinearOp(
        name="toy_synthetic",
        in_features=3,
        out_features=2,
        weight=w_float,
        bias=b_float,
        is_quantized=False,
    )

    compiled_op = compile_linear(
        raw_op,
        x_calib=x_calib,
        calib_method="absmax",
        calibration_id="test_synthetic_calib",
    )

    assert compiled_op.is_quantized
    assert compiled_op.weight.dtype == "int8"
    assert compiled_op.weight.shape == (2, 3)
    assert compiled_op.weight.kind == TensorKind.CONSTANT
    assert np.all(compiled_op.weight.data >= -127)
    assert np.all(compiled_op.weight.data <= 127)
    assert not np.any(compiled_op.weight.data == -128)

    assert compiled_op.bias is not None
    assert compiled_op.bias.dtype == "int32"
    assert compiled_op.bias.shape == (2,)

    assert compiled_op.weight_scale is not None
    assert compiled_op.weight_scale.shape == (2,)
    assert compiled_op.act_scale is not None
    assert compiled_op.act_scale > 0.0

    assert compiled_op.acc_dtype == "int32"
    assert compiled_op.acc_bits_min == 17
    assert compiled_op.hw["arch"] == "fixed"
    assert compiled_op.quant["calibration_id"] == "test_synthetic_calib"


def test_compiled_linear_serialization_roundtrip(tmp_path: Path):
    """Verify compiled LinearOp saves and loads losslessly."""
    w_float = np.array([[0.3, 0.7], [-0.5, 0.2]], dtype=np.float32)
    x_calib = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)

    raw_op = LinearOp(
        name="roundtrip_op",
        in_features=2,
        out_features=2,
        weight=w_float,
        is_quantized=False,
    )

    compiled = compile_linear(raw_op, x_calib=x_calib)
    json_path = compiled.save(tmp_path)

    reloaded = LinearOp.load(json_path)
    assert reloaded.name == compiled.name
    assert reloaded.is_quantized
    assert np.array_equal(reloaded.weight.data, compiled.weight.data)
    assert np.array_equal(reloaded.weight_scale, compiled.weight_scale)
    assert reloaded.act_scale == compiled.act_scale
    assert reloaded.acc_bits_min == compiled.acc_bits_min
    assert reloaded.acc_dtype == compiled.acc_dtype
    assert reloaded.hw == compiled.hw
    assert reloaded.quant == compiled.quant


def test_compile_real_smollm_layer0_q_proj():
    """Verify compilation of real SmolLM2-135M layer 0 q_proj."""
    artifact_dir = (
        Path("artifacts")
        / "SmolLM2-135M@93efa2f097d5"
        / "layer_0"
        / "q_proj"
    )
    if not artifact_dir.exists():
        pytest.skip("SmolLM2 layer 0 artifact not available")

    compiled_op = compile_layer_from_artifact(artifact_dir)

    assert compiled_op.name == "q_proj"
    assert compiled_op.in_features == 576
    assert compiled_op.out_features == 576
    assert compiled_op.is_quantized
    assert compiled_op.weight.dtype == "int8"
    assert compiled_op.weight.kind == TensorKind.CONSTANT
    assert compiled_op.weight_scale is not None
    assert compiled_op.weight_scale.shape == (576,)
    assert compiled_op.act_scale is not None
    assert compiled_op.act_scale > 0.0
    assert compiled_op.acc_bits_min == 25
    assert compiled_op.acc_dtype == "int32"
    assert compiled_op.hw["arch"] == "fixed"


def test_pipeline_rejects_oversized_k():
    """Verify pipeline rejects oversized layer that overflows INT32."""
    w_huge = np.zeros((1, 200000), dtype=np.float32)
    x_calib = np.ones((1, 200000), dtype=np.float32)

    raw_op = LinearOp(
        name="oversized_op",
        in_features=200000,
        out_features=1,
        weight=w_huge,
        is_quantized=False,
    )

    with pytest.raises(ValueError, match="exceeds signed 32-bit capacity"):
        compile_linear(raw_op, x_calib=x_calib)
