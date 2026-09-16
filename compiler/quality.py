"""Quantization numeric quality evaluation metrics for Claim C2 per rule N9."""

from typing import Any

import numpy as np

from compiler.calibration import (
    compute_activation_scale,
    compute_activation_stat,
    quantize_activations,
)
from compiler.quantization import quantize_bias, quantize_weights
from verification.reference import linear_reference_int64


def compute_max_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute maximum absolute error: max(|y_true - y_pred|)."""
    return float(np.max(np.abs(y_true - y_pred)))


def compute_relative_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute relative L2 Frobenius error: ||y_true - y_pred||_2 / (||y_true||_2 + eps)."""
    diff_norm = np.linalg.norm(y_true - y_pred)
    true_norm = np.linalg.norm(y_true)
    if true_norm == 0.0:
        return 0.0 if diff_norm == 0.0 else float("inf")
    return float(diff_norm / true_norm)


def compute_cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute cosine similarity over flattened output vectors."""
    y_true_flat = y_true.reshape(-1).astype(np.float64)
    y_pred_flat = y_pred.reshape(-1).astype(np.float64)

    dot = np.dot(y_true_flat, y_pred_flat)
    norm_true = np.linalg.norm(y_true_flat)
    norm_pred = np.linalg.norm(y_pred_flat)

    if norm_true == 0.0 and norm_pred == 0.0:
        return 1.0
    if norm_true == 0.0 or norm_pred == 0.0:
        return 0.0

    cos_sim = dot / (norm_true * norm_pred)
    return float(np.clip(cos_sim, -1.0, 1.0))


def compute_sqnr_db(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Signal-to-Quantization-Noise Ratio in decibels (dB).

    SQNR = 10 * log10(E[y_true^2] / (E[(y_true - y_pred)^2] + eps))
    """
    diff = y_true.astype(np.float64) - y_pred.astype(np.float64)
    noise_power = np.mean(diff**2)
    signal_power = np.mean(y_true.astype(np.float64) ** 2)

    if noise_power == 0.0:
        return 100.0  # Cap perfect reconstruction at 100 dB

    if signal_power == 0.0:
        return 0.0

    sqnr = 10.0 * np.log10(signal_power / noise_power)
    return float(sqnr)


def evaluate_layer_c2(
    w_float: np.ndarray,
    x_float: np.ndarray,
    b_float: np.ndarray | None = None,
    calib_method: str = "absmax",
) -> dict[str, Any]:
    """Evaluate Claim C2 on a layer: compare dequantized integer reference with float reference.

    Args:
        w_float: Float weight matrix of shape [N, K].
        x_float: Float activation tensor of shape [M, K] or [B, S, K].
        b_float: Optional float bias vector of shape [N].
        calib_method: 'absmax' or 'percentile'.

    Returns:
        Dictionary with max_abs_error, relative_l2, cosine_similarity, sqnr_db, and scale info.
    """
    # 2D normalization for activations
    k = w_float.shape[1]
    x_2d = x_float.reshape(-1, k).astype(np.float64)
    w_f64 = w_float.astype(np.float64)

    # 1. Float reference output
    y_float = np.matmul(x_2d, w_f64.T)
    if b_float is not None:
        y_float += b_float.astype(np.float64)

    # 2. Quantize weights (Rule N2)
    w_q, s_w = quantize_weights(w_float)

    # 3. Calibrate and quantize activations (Rule N3)
    stat = compute_activation_stat(x_2d, method=calib_method)
    s_x = compute_activation_scale(stat)
    x_q = quantize_activations(x_2d, s_x)

    # 4. Quantize bias if present (Rule N5)
    b_q: np.ndarray | None = None
    if b_float is not None:
        b_q = quantize_bias(b_float, s_x=s_x, s_w=s_w)

    # 5. Integer reference execution in int64 (Rule N1, N4)
    y_int = linear_reference_int64(x_q, w_q, b_q)

    # 6. Dequantization per Rule N6: Y_dequant = s_x * s_w[n] * Y_int
    scales = s_x * s_w  # shape [N]
    y_dequant = y_int.astype(np.float64) * scales[np.newaxis, :]

    # 7. Compute metrics
    max_abs_err = compute_max_abs_error(y_float, y_dequant)
    rel_l2 = compute_relative_l2(y_float, y_dequant)
    cos_sim = compute_cosine_similarity(y_float, y_dequant)
    sqnr = compute_sqnr_db(y_float, y_dequant)

    return {
        "max_abs_error": max_abs_err,
        "relative_l2": rel_l2,
        "cosine_similarity": cos_sim,
        "sqnr_db": sqnr,
        "s_x": float(s_x),
        "s_w_mean": float(np.mean(s_w)),
        "s_w_max": float(np.max(s_w)),
        "shape_n": int(w_float.shape[0]),
        "shape_k": int(w_float.shape[1]),
        "num_tokens": int(x_2d.shape[0]),
    }
