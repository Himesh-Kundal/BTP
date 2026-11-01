"""Independent integer reference oracle (N1-N7).

This module is the single source of truth for golden linear layer outputs.
It uses NumPy INT64 arithmetic to compute exact integer outputs without
overflow, independent of any hardware implementation or PyTorch runtime.
"""


import numpy as np


def quantize_weight_sym(
    w_float: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-output-channel weight quantization per rule N2.

    Args:
        w_float: Weight matrix of shape [N, K] in float.

    Returns:
        (W_q, s_w):
            W_q: Quantized int8 matrix of shape [N, K] with values in [-127, 127].
            s_w: Per-channel scale vector of shape [N] in float64.
    """
    w_f64 = np.asarray(w_float, dtype=np.float64)
    if w_f64.ndim != 2:
        raise ValueError(f"Weight tensor must be 2D [N, K], got shape {w_f64.shape}")

    n_rows = w_f64.shape[0]
    s_w = np.zeros(n_rows, dtype=np.float64)
    w_q = np.zeros_like(w_f64, dtype=np.int8)

    for n in range(n_rows):
        row = w_f64[n, :]
        max_abs = np.max(np.abs(row))
        if max_abs == 0.0:
            scale = 1.0
            q_row = np.zeros_like(row, dtype=np.int8)
        else:
            scale = max_abs / 127.0
            scaled = row / scale
            rounded = np.rint(scaled)  # round-half-to-even
            clipped = np.clip(rounded, -127.0, 127.0)
            q_row = clipped.astype(np.int8)

        s_w[n] = scale
        w_q[n, :] = q_row

    return w_q, s_w


def quantize_act_sym(
    x_float: np.ndarray,
    stat: float,
) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor activation quantization per rule N3.

    Args:
        x_float: Activation tensor of shape [..., K] in float.
        stat: Calibration statistic (e.g. absmax or recorded percentile).

    Returns:
        (X_q, s_x):
            X_q: Quantized int8 tensor with values in [-127, 127].
            s_x: Scalar scale in float64.
    """
    x_f64 = np.asarray(x_float, dtype=np.float64)
    if stat <= 0.0:
        raise ValueError(f"Calibration statistic must be positive, got {stat}")

    s_x = float(stat) / 127.0
    scaled = x_f64 / s_x
    rounded = np.rint(scaled)  # round-half-to-even
    clipped = np.clip(rounded, -127.0, 127.0)
    x_q = clipped.astype(np.int8)
    return x_q, s_x


def quantize_bias(
    b_float: np.ndarray,
    s_x: float,
    s_w: np.ndarray,
) -> np.ndarray:
    """Quantize bias vector per rule N5.

    b_q[n] = clip(rint(b[n] / (s_x * s_w[n])), int32)

    Args:
        b_float: Bias vector of shape [N].
        s_x: Scalar activation scale.
        s_w: Per-channel weight scale vector of shape [N].

    Returns:
        b_q: Quantized int32 bias vector of shape [N].
    """
    b_f64 = np.asarray(b_float, dtype=np.float64)
    s_w_f64 = np.asarray(s_w, dtype=np.float64)

    if b_f64.shape != s_w_f64.shape:
        raise ValueError(f"Bias shape {b_f64.shape} does not match weight scale shape {s_w_f64.shape}")

    scales = s_x * s_w_f64
    scaled = b_f64 / scales
    rounded = np.rint(scaled)
    # Clip to signed 32-bit integer range
    clipped = np.clip(rounded, -2147483648.0, 2147483647.0)
    return clipped.astype(np.int32)


def linear_reference_int64(
    x: np.ndarray,
    w: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Exact integer linear layer oracle per rule N1 and N4.

    Computes Y = X @ W.T + b_q in exact INT64 arithmetic.

    Args:
        x: Input tensor of shape [..., K] (int8, allowing -128..127).
        w: Weight matrix of shape [N, K] (int8, within -127..127).
        bias: Optional bias vector of shape [N] (int32).

    Returns:
        Y: Exact output array of shape [..., N] in int64.
    """
    x_arr = np.asarray(x, dtype=np.int64)
    w_arr = np.asarray(w, dtype=np.int64)

    if w_arr.ndim != 2:
        raise ValueError(f"w must be 2D [N, K], got {w_arr.shape}")
    if x_arr.shape[-1] != w_arr.shape[1]:
        raise ValueError(f"Feature dimension mismatch: x {x_arr.shape} vs w {w_arr.shape}")

    # Compute dot product in int64
    y = np.matmul(x_arr, w_arr.T)

    if bias is not None:
        b_arr = np.asarray(bias, dtype=np.int64)
        if b_arr.shape != (w_arr.shape[0],):
            raise ValueError(f"Bias shape {b_arr.shape} does not match out_features {w_arr.shape[0]}")
        y = y + b_arr

    return y


def dequantize(
    y_int32: np.ndarray,
    s_x: float,
    s_w: np.ndarray,
) -> np.ndarray:
    """Dequantize raw accumulator outputs to float per rule N6.

    y_float ≈ s_x * s_w[n] * Y[m, n]

    Args:
        y_int32: Accumulator outputs of shape [..., N].
        s_x: Scalar activation scale.
        s_w: Per-channel weight scale vector of shape [N].

    Returns:
        y_float: Dequantized float64 array of shape [..., N].
    """
    y_f64 = np.asarray(y_int32, dtype=np.float64)
    s_w_f64 = np.asarray(s_w, dtype=np.float64)
    return y_f64 * (s_x * s_w_f64)
