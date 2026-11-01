"""Accumulator bitwidth analysis and overflow safety pass per rule N4 and N8."""

import math

import numpy as np

from ir.linear import LinearOp

INT32_MAX_LIMIT: int = 2**31  # 2147483648


def compute_accumulator_bound(k: int, max_bq: int = 0) -> int:
    """Compute worst-case accumulation bound: bound = K * 128 * 127 + |b_q|_max.

    Each INT8 input activation has maximum magnitude 128 (at -128 per N7).
    Each INT8 weight has maximum magnitude 127 (within [-127, 127] per N2).
    Worst-case term product is 128 * 127 = 16256.
    """
    return k * 128 * 127 + abs(int(max_bq))


def compute_acc_bits_min(bound: int) -> int:
    """Compute minimum signed two's complement bitwidth required to hold bound without overflow.

    To represent range [-bound, bound], we need 2^(B-1) - 1 >= bound:
        B = ceil(log2(bound + 1)) + 1
    """
    if bound <= 0:
        return 1
    return math.ceil(math.log2(bound + 1)) + 1


def check_accumulator_bound(k: int, max_bq: int = 0) -> int:
    """Verify worst-case accumulator bound is strictly less than 2^31 per rule N4.

    Raises ValueError if bound >= 2^31.
    Returns acc_bits_min.
    """
    bound = compute_accumulator_bound(k=k, max_bq=max_bq)
    if bound >= INT32_MAX_LIMIT:
        raise ValueError(
            f"Worst-case accumulator bound {bound} exceeds signed 32-bit capacity "
            f"({INT32_MAX_LIMIT}) for K={k}, max_bq={max_bq}"
        )
    return compute_acc_bits_min(bound)


def check_linear_op_accumulator(op: LinearOp) -> LinearOp:
    """Compiler pass to verify accumulator safety and annotate minimal required bitwidth."""
    max_bq = 0
    if op.bias is not None and op.bias.data is not None:
        max_bq = int(np.max(np.abs(op.bias.data)))

    bound = compute_accumulator_bound(k=op.in_features, max_bq=max_bq)
    bits_min = check_accumulator_bound(k=op.in_features, max_bq=max_bq)

    op.acc_bits_min = bits_min
    op.acc_dtype = "int32"
    op.quant["accumulator_bound"] = bound
    op.quant["acc_bits_min"] = bits_min
    return op
