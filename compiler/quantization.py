"""Symmetric INT8 weight quantization compiler passes per rule N2."""

import numpy as np

from ir.linear import LinearOp
from ir.tensor import TensorKind, TensorRef


def quantize_weights(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-output-channel INT8 weight quantization per rule N2.

    Computes per-output-channel scale:
        s_w[n] = max(|W[n, :]|) / 127.0 (in float64)
    If the row is all zero, s_w[n] = 1.0 and the row is all zero.
    Quantized weights:
        W_q = clip(rint(W / s_w), -127, 127)
    where rint is round-half-to-even.

    Args:
        w: Weight array of shape [N, K].

    Returns:
        (W_q, s_w):
            W_q: np.ndarray of shape [N, K], dtype int8, values in [-127, 127].
            s_w: np.ndarray of shape [N], dtype float64.
    """
    w_f64 = np.asarray(w, dtype=np.float64)
    if w_f64.ndim != 2:
        raise ValueError(f"Weight matrix must be 2D [N, K], got shape {w_f64.shape}")

    n_rows, _ = w_f64.shape
    s_w = np.zeros(n_rows, dtype=np.float64)
    w_q = np.zeros_like(w_f64, dtype=np.int8)

    for n in range(n_rows):
        row = w_f64[n, :]
        max_abs = float(np.max(np.abs(row)))
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


def dequantize_weights(w_q: np.ndarray, s_w: np.ndarray) -> np.ndarray:
    """Dequantize INT8 weights to float64 for verification: W_float = s_w * W_q."""
    return s_w[:, np.newaxis] * w_q.astype(np.float64)


def quantize_bias(
    b_float: np.ndarray,
    s_x: float,
    s_w: np.ndarray,
) -> np.ndarray:
    """Quantize bias vector per rule N5.

    b_q[n] = clip(rint(b[n] / (s_x * s_w[n])), -2^31, 2^31 - 1)

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
        raise ValueError(
            f"Bias shape {b_f64.shape} does not match weight scale shape {s_w_f64.shape}"
        )

    scales = s_x * s_w_f64
    scales = np.where(scales == 0.0, 1.0, scales)
    scaled = b_f64 / scales
    rounded = np.rint(scaled)  # round-half-to-even
    clipped = np.clip(rounded, -2147483648.0, 2147483647.0)
    return clipped.astype(np.int32)


def dequantize_bias(
    b_q: np.ndarray,
    s_x: float,
    s_w: np.ndarray,
) -> np.ndarray:
    """Dequantize INT32 bias to float64 for verification: b_float = b_q * (s_x * s_w)."""
    scales = s_x * np.asarray(s_w, dtype=np.float64)
    return b_q.astype(np.float64) * scales


def quantize_linear_op(op: LinearOp) -> LinearOp:
    """Pass to quantize weights and bias of an unquantized LinearOp into INT8/INT32."""
    if op.weight.data is None:
        raise ValueError(f"Cannot quantize LinearOp '{op.name}' without in-memory weight data")

    w_q, s_w = quantize_weights(op.weight.data)

    quant_weight_ref = TensorRef(
        name=f"{op.name}_w",
        shape=w_q.shape,
        dtype="int8",
        kind=TensorKind.CONSTANT,
        data=w_q,
    )

    quant_bias_ref: TensorRef | None = None
    if op.bias is not None:
        if op.bias.dtype in ("float32", "float64") and op.bias.data is not None:
            s_x = op.act_scale if op.act_scale is not None else 1.0
            b_q = quantize_bias(op.bias.data, s_x=s_x, s_w=s_w)
            quant_bias_ref = TensorRef(
                name=f"{op.name}_b",
                shape=b_q.shape,
                dtype="int32",
                kind=TensorKind.CONSTANT,
                data=b_q,
            )
        elif op.bias.dtype == "int32":
            quant_bias_ref = op.bias

    quant_meta = dict(op.quant)
    quant_meta["scheme"] = "symmetric"
    quant_meta["rounding"] = "round_half_to_even"
    quant_meta["weight_bits"] = 8
    quant_meta["weight_range"] = [-127, 127]
    if quant_bias_ref is not None:
        quant_meta["bias_bits"] = 32

    return LinearOp(
        name=op.name,
        in_features=op.in_features,
        out_features=op.out_features,
        weight=quant_weight_ref,
        bias=quant_bias_ref,
        input_shape=op.input_shape,
        weight_scale=s_w,
        act_scale=op.act_scale,
        act_dtype=op.act_dtype,
        acc_dtype=op.acc_dtype,
        acc_bits_min=op.acc_bits_min,
        kind=op.kind,
        quant=quant_meta,
        hw=op.hw,
        source=op.source,
        is_quantized=True,
    )
