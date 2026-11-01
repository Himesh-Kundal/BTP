"""Constant weight specialization and port stripping for Design F (T4.1, Invariant I3).

Enforces compile-time weight specialization for fixed-weight hardware (Design F):
- Verifies that LinearOp weights are quantized and marked TensorKind.CONSTANT.
- Guarantees that no runtime weight input ports (e.g. w_we, w_addr, w_data, w_flat) exist.
- Formulates the exact port interfaces for combinational cores and registered wrappers.
"""

from __future__ import annotations

import re
from typing import Any

from ir.linear import LinearOp
from ir.tensor import TensorKind

# Forbidden port prefixes/patterns for Design F
FORBIDDEN_WEIGHT_PORT_PATTERNS = [
    re.compile(r"^w_(we|addr|data|flat|clk|en)$", re.IGNORECASE),
    re.compile(r"^w_.*", re.IGNORECASE),
    re.compile(r"^weight_.*", re.IGNORECASE),
]


def specialize_constant_weights(op: LinearOp) -> LinearOp:
    """Specialize a quantized LinearOp for Design F (fixed-weight architecture).

    Verifies Invariant I3: op.weight.kind must be TensorKind.CONSTANT.
    Marks op.hw["arch"] = "fixed" and ensures weights are treated strictly as
    compile-time numerical literals with zero runtime weight ports.

    Args:
        op: A quantized LinearOp to specialize.

    Returns:
        The specialized LinearOp.

    Raises:
        ValueError: If op is unquantized or weight is not TensorKind.CONSTANT.
    """
    if not op.is_quantized:
        raise ValueError(
            f"LinearOp '{op.name}' must be quantized before constant weight specialization."
        )

    if op.weight.kind != TensorKind.CONSTANT:
        raise ValueError(
            f"Design F requires constant weights (TensorKind.CONSTANT), "
            f"but found '{op.weight.kind}' for op '{op.name}'."
        )

    # Initialize or update hw field
    if op.hw is None:
        op.hw = {}
    op.hw["arch"] = "fixed"

    return op


def verify_design_f_ports(port_names: list[str]) -> bool:
    """Verify that a port list conforms to Design F by containing zero weight ports.

    Args:
        port_names: List of port names from generated RTL or module definition.

    Returns:
        True if all ports are valid Design F ports.

    Raises:
        ValueError: If any port matching forbidden weight patterns is detected.
    """
    for port in port_names:
        for pattern in FORBIDDEN_WEIGHT_PORT_PATTERNS:
            if pattern.match(port):
                raise ValueError(
                    f"Forbidden weight port detected for Design F: '{port}'. "
                    f"Design F must have zero weight ports and zero runtime weight signals."
                )
    return True


def get_design_f_interface_ports(
    in_features: int,
    out_features: int,
    is_wrapper: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return the canonical Design F port specification per docs/02 §6.

    Args:
        in_features: Number of input features K (or padded K').
        out_features: Number of output features N (or padded N').
        is_wrapper: If True, returns wrapper interface; otherwise combinational core.

    Returns:
        Dictionary mapping port name to port attributes (direction, width, signed).
    """
    if is_wrapper:
        return {
            "clk": {"direction": "input", "width": 1, "signed": False},
            "rst_n": {"direction": "input", "width": 1, "signed": False},
            "valid_in": {"direction": "input", "width": 1, "signed": False},
            "x_flat": {"direction": "input", "width": 8 * in_features, "signed": True},
            "valid_out": {"direction": "output", "width": 1, "signed": False},
            "y_flat": {"direction": "output", "width": 32 * out_features, "signed": True},
        }
    return {
        "x_flat": {"direction": "input", "width": 8 * in_features, "signed": True},
        "y_flat": {"direction": "output", "width": 32 * out_features, "signed": True},
    }
