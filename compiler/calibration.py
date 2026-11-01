"""Static per-tensor activation calibration and symmetric INT8 quantization per rule N3."""

from typing import Any

import numpy as np

from ir.linear import LinearOp


def compute_activation_stat(
    x: np.ndarray,
    method: str = "absmax",
    percentile: float = 99.99,
) -> float:
    """Compute calibration statistic over activation samples.

    Args:
        x: Sample activation tensor in float.
        method: 'absmax' (default) or 'percentile'.
        percentile: Percentile threshold if method='percentile' (e.g. 99.99).

    Returns:
        stat: Positive float statistic value.
    """
    x_f64 = np.asarray(x, dtype=np.float64)
    abs_x = np.abs(x_f64)

    if method == "absmax":
        stat = float(np.max(abs_x))
    elif method == "percentile":
        stat = float(np.percentile(abs_x, percentile))
    else:
        raise ValueError(f"Unknown calibration method '{method}'; expected 'absmax' or 'percentile'")

    return stat


def compute_activation_scale(stat: float) -> float:
    """Compute per-tensor static activation scale: s_x = stat / 127.0.

    If stat is 0.0 (all zeros), returns scale = 1.0.
    """
    if stat <= 0.0:
        return 1.0
    return float(stat / 127.0)


def quantize_activations(x: np.ndarray, scale: float) -> np.ndarray:
    """Symmetric per-tensor INT8 activation quantization per rule N3.

    Args:
        x: Input activations in float.
        scale: Positive float static scale s_x.

    Returns:
        x_q: Quantized int8 array with values strictly in [-127, 127].
    """
    if scale <= 0.0:
        raise ValueError(f"Activation scale must be positive, got {scale}")

    x_f64 = np.asarray(x, dtype=np.float64)
    scaled = x_f64 / scale
    rounded = np.rint(scaled)  # round-half-to-even
    clipped = np.clip(rounded, -127.0, 127.0)
    return clipped.astype(np.int8)


def dequantize_activations(x_q: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize INT8 activations to float64: X_float = X_q * scale."""
    return x_q.astype(np.float64) * scale


def calibrate_layer(
    op: LinearOp,
    x_calib: np.ndarray,
    method: str = "absmax",
    percentile: float = 99.99,
    calibration_id: str = "calib_default",
) -> LinearOp:
    """Run calibration pass on a LinearOp using recorded activation samples."""
    stat = compute_activation_stat(x_calib, method=method, percentile=percentile)
    scale = compute_activation_scale(stat)

    quant_meta: dict[str, Any] = dict(op.quant)
    quant_meta["act_stat"] = stat
    quant_meta["act_scale"] = scale
    quant_meta["act_method"] = method
    quant_meta["act_bits"] = 8
    quant_meta["act_range"] = [-127, 127]
    quant_meta["calibration_id"] = calibration_id
    if method == "percentile":
        quant_meta["act_percentile"] = percentile

    op.act_scale = scale
    op.quant = quant_meta
    return op
