"""Tests for accumulator bitwidth analysis and overflow safety pass per rule N4 (T3.5)."""

import numpy as np
import pytest

from compiler.accumulator import (
    check_accumulator_bound,
    check_linear_op_accumulator,
    compute_acc_bits_min,
    compute_accumulator_bound,
)
from ir.linear import LinearOp


def test_accumulator_bound_basic():
    """Verify worst-case bound calculation: bound = K * 128 * 127 + |b_q|_max."""
    # K=1, no bias: 1 * 128 * 127 = 16256
    b1 = compute_accumulator_bound(k=1, max_bq=0)
    assert b1 == 16256
    bits1 = compute_acc_bits_min(b1)
    assert bits1 == 15  # 2^(15-1) - 1 = 16383 >= 16256

    # K=3, with bias 1000
    b3 = compute_accumulator_bound(k=3, max_bq=1000)
    assert b3 == 3 * 16256 + 1000
    bits3 = compute_acc_bits_min(b3)
    assert bits3 == 17


def test_smollm_layer_dimensions():
    """Verify accumulator bounds for SmolLM2-135M dimensions (K=576 and K=1536)."""
    # K=576 (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj)
    b_576 = compute_accumulator_bound(k=576)
    assert b_576 == 576 * 128 * 127
    assert b_576 == 9363456
    assert b_576 < 2**31
    bits_576 = check_accumulator_bound(k=576)
    assert bits_576 == 25

    # K=1536 (down_proj)
    b_1536 = compute_accumulator_bound(k=1536)
    assert b_1536 == 1536 * 128 * 127
    assert b_1536 == 24969216
    assert b_1536 < 2**31
    bits_1536 = check_accumulator_bound(k=1536)
    assert bits_1536 == 26


def test_overflow_rejection():
    """Rule N4: Reject layers if worst-case accumulation >= 2^31."""
    # Critical threshold: 2^31 // 16256 = 132104.
    # K = 132104: 132104 * 16256 = 2147482624 < 2^31 (passes)
    check_accumulator_bound(k=132104)

    # K = 132105: 132105 * 16256 = 2147498880 >= 2^31 (must reject)
    with pytest.raises(ValueError, match="exceeds signed 32-bit capacity"):
        check_accumulator_bound(k=132105)

    # Rejection triggered by large bias
    with pytest.raises(ValueError, match="exceeds signed 32-bit capacity"):
        check_accumulator_bound(k=576, max_bq=2**31)


def test_check_linear_op_accumulator_pass():
    """Verify check_linear_op_accumulator pass on LinearOp."""
    w = np.array([[3, 7, -2]], dtype=np.int8)
    b = np.array([100], dtype=np.int32)
    op = LinearOp(name="test_op", in_features=3, out_features=1, weight=w, bias=b)

    checked_op = check_linear_op_accumulator(op)
    assert checked_op.acc_bits_min == 17
    assert checked_op.acc_dtype == "int32"
    assert "accumulator_bound" in checked_op.quant


def test_check_linear_op_accumulator_rejection():
    """Verify check_linear_op_accumulator rejects invalid oversized layer."""
    # Huge dimension that overflows INT32
    w_huge = np.zeros((1, 200000), dtype=np.int8)
    op_huge = LinearOp(name="huge_op", in_features=200000, out_features=1, weight=w_huge)

    with pytest.raises(ValueError, match="exceeds signed 32-bit capacity"):
        check_linear_op_accumulator(op_huge)
