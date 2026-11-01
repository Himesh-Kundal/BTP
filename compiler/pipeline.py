"""End-to-end compiler pipeline driver for linear layers (Phase 3)."""

from pathlib import Path

import numpy as np

from compiler.accumulator import check_linear_op_accumulator
from compiler.calibration import calibrate_layer
from compiler.quantization import quantize_linear_op
from ir.linear import LinearOp, load_linear_from_artifact
from ir.tensor import TensorKind


def compile_linear(
    op: LinearOp,
    x_calib: np.ndarray,
    calib_method: str = "absmax",
    calib_percentile: float = 99.99,
    calibration_id: str = "calib_default",
    arch: str = "fixed",
) -> LinearOp:
    """Execute complete Phase 3 compiler passes on a LinearOp.

    Passes:
    1. Activation calibration (Rule N3): compute scalar static scale s_x.
    2. Weight & bias quantization (Rule N2, N5): compute per-channel s_w, quantize W to int8, b to int32.
    3. Accumulator bound check (Rule N4, N8): verify worst-case bound < 2^31, annotate acc_bits_min.
    4. Hardware architecture annotation (Design F): set arch and enforce CONSTANT weight kind.

    Args:
        op: Raw unquantized or partially compiled LinearOp.
        x_calib: Activation sample array for calibration.
        calib_method: 'absmax' or 'percentile'.
        calib_percentile: Percentile threshold if method='percentile'.
        calibration_id: Unique string tracking calibration provenance.
        arch: 'fixed' (Design F) or 'programmable' (Design P).

    Returns:
        compiled_op: Fully specialized and validated quantized LinearOp.
    """
    # Pass 1: Activation calibration
    calib_op = calibrate_layer(
        op,
        x_calib=x_calib,
        method=calib_method,
        percentile=calib_percentile,
        calibration_id=calibration_id,
    )

    # Pass 2: Weight and bias quantization
    quant_op = quantize_linear_op(calib_op)

    # Pass 3: Accumulator overflow bound analysis
    bound_op = check_linear_op_accumulator(quant_op)

    # Pass 4: Architecture specialization
    hw_meta = dict(bound_op.hw)
    hw_meta["arch"] = arch
    bound_op.hw = hw_meta

    if arch == "fixed":
        bound_op.weight.kind = TensorKind.CONSTANT

    return bound_op


def compile_layer_from_artifact(
    artifact_dir: Path | str,
    calib_x_path: Path | str | None = None,
    name: str | None = None,
    arch: str = "fixed",
) -> LinearOp:
    """Load an unquantized layer from artifacts, load calibration data, and compile."""
    art_path = Path(artifact_dir)
    raw_op = load_linear_from_artifact(art_path, name=name)

    if calib_x_path is not None:
        x_file = Path(calib_x_path)
    else:
        x_file = art_path / "calib" / "x_sample.npy"

    if not x_file.is_file():
        raise FileNotFoundError(f"Calibration activation file not found at {x_file}")

    x_calib = np.load(x_file)
    return compile_linear(raw_op, x_calib=x_calib, arch=arch)
