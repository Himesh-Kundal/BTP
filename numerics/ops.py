"""Pure Integer Operations Library and LUT Builders (Task UA.3).

Implements the single mathematical source of truth (Oracle) for all integer operations
defined in Part II Numeric Contract v2 (docs/part2/02_architecture_and_contracts.md):
- N2-2: rshift_rnd(x, s) with round-half-up toward +inf, s=0 identity
- N2-2: sat(x, min_v, max_v) with symmetric INT8 [-127, 127] default
- N2-3: requant(acc, M, s) dyadic scaling
- N2-10: build_exp2_fraction_lut (base-2 exponential decomposition)
- N2-6: build_rsqrt_mantissa_lut (reciprocal square-root for RMSNorm)
- N2-10: build_reciprocal_lut (reciprocal for Softmax sum normalization)
- N2-11: build_silu_lut (512-entry SiLU activation lookup table)

Strict Layering (Golden Rule 6):
Only Python stdlib and NumPy are used. Never import torch or transformers.
"""


import numpy as np

# ==============================================================================
# 1. Core Dyadic Arithmetic & Saturation (N2-2, N2-3)
# ==============================================================================


def rshift_rnd(x: int | np.ndarray, s: int) -> int | np.ndarray:
    """Arithmetic right shift with round-half-up toward +infinity.
    
    Formula: (x + (1 << (s - 1))) >> s for s > 0; identity for s == 0.
    """
    if s == 0:
        return x
    if s < 0:
        raise ValueError(f"Shift amount s must be non-negative, got {s}")

    offset = 1 << (s - 1)
    if isinstance(x, np.ndarray):
        # Force 64-bit signed integer to prevent intermediate overflow
        x_int64 = x.astype(np.int64)
        return np.right_shift(x_int64 + offset, s)
    else:
        return (int(x) + offset) >> s


def sat(
    x: int | np.ndarray, min_val: int = -127, max_val: int = 127
) -> int | np.ndarray:
    """Saturate integer or array to [min_val, max_val].
    
    Defaults to symmetric INT8 [-127, 127] per Numeric Contract N2-2.
    """
    if isinstance(x, np.ndarray):
        return np.clip(x, min_val, max_val)
    else:
        return max(min_val, min(max_val, int(x)))


def requant(
    acc: int | np.ndarray,
    M: int | np.ndarray,
    s: int,
    out_min: int = -127,
    out_max: int = 127,
) -> int | np.ndarray:
    """Dyadic requantization: y = sat(rshift_rnd(acc * M, s), out_min, out_max).
    
    acc: signed accumulator (typically INT32 or INT48)
    M: positive integer scale multiplier
    s: right shift amount
    """
    if isinstance(acc, np.ndarray):
        acc_int64 = acc.astype(np.int64)
        prod = acc_int64 * M
    else:
        prod = int(acc) * int(M)

    shifted = rshift_rnd(prod, s)
    return sat(shifted, out_min, out_max)


def float_to_dyadic_scale(scale: float, max_m_bits: int = 31) -> tuple[int, int]:
    """Convert a positive real float scale into an exact dyadic multiplier (M, s).
    
    Approximates: scale ≈ M * 2^(-s) with M >= 0 and s >= 0.
    """
    if scale <= 0.0:
        return 0, 0

    # Find shift s such that M fits within max_m_bits
    s = int(np.ceil(np.log2(1.0 / scale))) + (max_m_bits - 1) if scale < 1.0 else max_m_bits - 1
    s = max(0, min(62, s))
    M = int(np.round(scale * (2**s)))

    # Minimize bitwidth if trailing zeros exist
    while s > 0 and M % 2 == 0:
        M //= 2
        s -= 1

    return M, s


# ==============================================================================
# 2. Offline LUT Builders (Float64 with Explicit Rounding, N2-14)
# ==============================================================================


def build_exp2_fraction_lut(table_size: int = 256, out_bits: int = 16) -> np.ndarray:
    """Build lookup table for fractional base-2 exponent: 2^(k / table_size).
    
    Evaluates 2^f for f in [0, 1) in fixed-point format Q0.out_bits.
    """
    k = np.arange(table_size, dtype=np.float64)
    frac = k / float(table_size)
    values = np.exp2(frac)  # in [1.0, 2.0)
    scale = float(1 << out_bits)
    lut = np.round(values * scale).astype(np.int64)
    return lut


def build_rsqrt_mantissa_lut(table_size: int = 256, out_bits: int = 16) -> np.ndarray:
    """Build reciprocal square root mantissa LUT for RMSNorm.
    
    Normalized mantissa m in [1.0, 2.0), returns round(1 / sqrt(m) * 2^out_bits).
    """
    k = np.arange(table_size, dtype=np.float64)
    m = 1.0 + (k / float(table_size))
    values = 1.0 / np.sqrt(m)  # in (0.7071, 1.0]
    scale = float(1 << out_bits)
    lut = np.round(values * scale).astype(np.int64)
    return lut


def build_reciprocal_lut(table_size: int = 256, out_bits: int = 16) -> np.ndarray:
    """Build reciprocal mantissa LUT for Softmax normalizer sum.
    
    Normalized mantissa m in [1.0, 2.0), returns round((1.0 / m) * 2^out_bits).
    """
    k = np.arange(table_size, dtype=np.float64)
    m = 1.0 + (k / float(table_size))
    values = 1.0 / m  # in (0.5, 1.0]
    scale = float(1 << out_bits)
    lut = np.round(values * scale).astype(np.int64)
    return lut


def build_silu_lut(
    in_min: int = -256,
    in_max: int = 255,
    in_scale: float = 0.05,
    out_scale: float = 0.05,
    out_min: int = -127,
    out_max: int = 127,
) -> np.ndarray:
    """Build symmetric lookup table for SiLU(x) = x * sigmoid(x).
    
    Inputs are discrete integer indices from in_min to in_max.
    Real input x_real = x_int * in_scale
    y_real = SiLU(x_real)
    y_int = sat(round(y_real / out_scale), out_min, out_max)
    """
    indices = np.arange(in_min, in_max + 1, dtype=np.float64)
    x_real = indices * in_scale
    sigmoid = 1.0 / (1.0 + np.exp(-x_real))
    y_real = x_real * sigmoid
    y_int = np.round(y_real / out_scale).astype(np.int64)
    return np.clip(y_int, out_min, out_max).astype(np.int8)


def build_rope_cos_sin_tables(
    max_ctx: int = 256,
    head_dim: int = 64,
    theta: float = 100000.0,
    out_bits: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """Build fixed-point cos and sin tables for RoPE in Q1.out_bits format.
    
    Returns:
        cos_table: shape [max_ctx, head_dim // 2] int16
        sin_table: shape [max_ctx, head_dim // 2] int16
    """
    half_dim = head_dim // 2
    idx = np.arange(half_dim, dtype=np.float64)
    inv_freq = 1.0 / (theta ** (2.0 * idx / head_dim))

    positions = np.arange(max_ctx, dtype=np.float64)[:, np.newaxis]
    angles = positions * inv_freq  # [max_ctx, half_dim]

    scale = float(1 << out_bits)
    cos_q = np.round(np.cos(angles) * scale).astype(np.int16)
    sin_q = np.round(np.sin(angles) * scale).astype(np.int16)
    return cos_q, sin_q
