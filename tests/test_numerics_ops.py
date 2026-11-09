"""Exhaustive and Edge-Case Tests for Pure Integer Ops & LUT Builders (Task UA.3)."""

import math

import numpy as np
import pytest

from numerics.ops import (
    build_exp2_fraction_lut,
    build_reciprocal_lut,
    build_rope_cos_sin_tables,
    build_rsqrt_mantissa_lut,
    build_silu_lut,
    float_to_dyadic_scale,
    requant,
    rshift_rnd,
    sat,
)


def test_rshift_rnd_zero_shift():
    """Verify s=0 is exact identity for all inputs."""
    for x in [-128, -42, -1, 0, 1, 42, 127, 100000]:
        assert rshift_rnd(x, 0) == x

    arr = np.array([-100, 0, 100], dtype=np.int64)
    np.testing.assert_array_equal(rshift_rnd(arr, 0), arr)


def test_rshift_rnd_round_half_up_ties():
    """Verify round-half-up toward +infinity on exact half ties."""
    # x = 1, s = 1 -> 1 / 2 = 0.5 -> rounds up to 1
    assert rshift_rnd(1, 1) == 1
    # x = -1, s = 1 -> -1 / 2 = -0.5 -> rounds up to 0
    assert rshift_rnd(-1, 1) == 0
    # x = 3, s = 1 -> 3 / 2 = 1.5 -> rounds up to 2
    assert rshift_rnd(3, 1) == 2
    # x = -3, s = 1 -> -3 / 2 = -1.5 -> rounds up to -1
    assert rshift_rnd(-3, 1) == -1

    # Exact integers (no fraction) stay exact
    assert rshift_rnd(4, 1) == 2
    assert rshift_rnd(-4, 1) == -2
    assert rshift_rnd(0, 1) == 0


def test_rshift_rnd_exhaustive_small_widths():
    """Exhaustively verify rshift_rnd across all 16-bit values for s in [1..8]."""
    test_range = range(-512, 512)
    for s in range(1, 9):
        divisor = 2**s
        for x in test_range:
            # Mathematical round half up: floor(x / 2^s + 0.5)
            expected = math.floor(x / divisor + 0.5)
            actual = rshift_rnd(x, s)
            assert actual == expected, f"Mismatch at x={x}, s={s}: got {actual}, expected {expected}"


def test_rshift_rnd_invalid_negative_shift():
    """Verify negative shift raises ValueError."""
    with pytest.raises(ValueError, match="Shift amount s must be non-negative"):
        rshift_rnd(10, -1)


def test_sat_symmetric_int8():
    """Verify symmetric INT8 saturation in [-127, 127]."""
    assert sat(0) == 0
    assert sat(127) == 127
    assert sat(128) == 127
    assert sat(1000) == 127
    assert sat(-127) == -127
    assert sat(-128) == -127
    assert sat(-1000) == -127

    arr = np.array([-200, -127, 0, 127, 200])
    sat_arr = sat(arr)
    np.testing.assert_array_equal(sat_arr, np.array([-127, -127, 0, 127, 127]))


def test_sat_custom_bounds():
    """Verify saturation with custom bitwidth bounds."""
    assert sat(50, min_val=-32, max_val=31) == 31
    assert sat(-40, min_val=-32, max_val=31) == -32
    assert sat(10, min_val=-32, max_val=31) == 10


def test_requant_hand_calculated():
    """Verify dyadic requantization formula on hand-calculated cases."""
    # acc = 100, M = 5, s = 2 -> 500 >> 2 with rounding: (500 + 2) >> 2 = 125
    assert requant(100, M=5, s=2) == 125

    # Saturate on large accumulator
    assert requant(1000, M=1, s=0) == 127
    assert requant(-1000, M=1, s=0) == -127


def test_float_to_dyadic_scale():
    """Verify floating point scale conversion to integer dyadic (M, s)."""
    scale = 0.00390625  # exactly 2^-8
    M, s = float_to_dyadic_scale(scale)
    assert M * (2.0**-s) == pytest.approx(scale, rel=1e-6)

    # General real scale
    scale = 0.012345
    M, s = float_to_dyadic_scale(scale)
    approx = M * (2.0**-s)
    assert abs(approx - scale) / scale < 1e-6


def test_build_exp2_fraction_lut():
    """Verify properties of the base-2 fractional exponent LUT."""
    lut = build_exp2_fraction_lut(table_size=256, out_bits=16)
    assert len(lut) == 256
    # 2^0 * 2^16 = 65536
    assert lut[0] == 65536
    # 2^(255/256) * 2^16 < 131072
    assert 65536 <= lut[-1] < 131072
    # Monotonically increasing
    assert np.all(np.diff(lut) >= 0)


def test_build_rsqrt_mantissa_lut():
    """Verify properties of reciprocal square root mantissa LUT."""
    lut = build_rsqrt_mantissa_lut(table_size=256, out_bits=16)
    assert len(lut) == 256
    # 1 / sqrt(1.0) * 2^16 = 65536
    assert lut[0] == 65536
    # 1 / sqrt(2.0) * 2^16 ≈ 46341
    assert 46300 <= lut[-1] <= 46400
    # Monotonically decreasing
    assert np.all(np.diff(lut) <= 0)


def test_build_reciprocal_lut():
    """Verify properties of reciprocal mantissa LUT."""
    lut = build_reciprocal_lut(table_size=256, out_bits=16)
    assert len(lut) == 256
    # 1 / 1.0 * 2^16 = 65536
    assert lut[0] == 65536
    # 1 / 2.0 * 2^16 ≈ 32768
    assert 32700 <= lut[-1] <= 32900
    # Monotonically decreasing
    assert np.all(np.diff(lut) <= 0)


def test_build_silu_lut():
    """Verify SiLU lookup table properties."""
    lut = build_silu_lut(in_min=-256, in_max=255, in_scale=0.05, out_scale=0.05)
    assert len(lut) == 512
    # Zero input gives zero output
    # Index for 0 is at offset 256
    assert lut[256] == 0
    # Positive values are positive
    assert lut[256 + 20] > 0
    # Extreme values fit in signed INT8
    assert np.all(lut >= -127) and np.all(lut <= 127)


def test_build_rope_tables():
    """Verify RoPE cos and sin tables formatting and range."""
    cos_t, sin_t = build_rope_cos_sin_tables(max_ctx=32, head_dim=16, theta=10000.0)
    assert cos_t.shape == (32, 8)
    assert sin_t.shape == (32, 8)
    # At pos 0: cos(0) = 1.0 * 2^14 = 16384, sin(0) = 0
    np.testing.assert_array_equal(cos_t[0], 16384)
    np.testing.assert_array_equal(sin_t[0], 0)
    assert np.all(np.abs(cos_t) <= 16384)
    assert np.all(np.abs(sin_t) <= 16384)
