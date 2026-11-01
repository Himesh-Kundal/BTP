"""Weight literal encoder and sparsity-aware term generation for Design F (T4.3).

Formats INT8 weights as signed Verilog-2001 literals, eliminates zero-weight
multiplication terms, handles +-1 pass-through and negation wires without
instantiating hardware multipliers, and generates sign-extended accumulation
expressions matching the architectural contracts in docs/02 §5, §6.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class EncodedTerm:
    """Representation of an encoded product wire in Verilog."""

    k: int
    w_val: int
    product_wire: str
    wire_declaration: str
    term_type: str  # "pass", "negate", "mult"


def format_weight_literal(val: int) -> str:
    """Format an integer in [-127, 127] as a Verilog-2001 signed literal.

    Args:
        val: Weight value in [-127, 127].

    Returns:
        Verilog literal string, e.g. "8'sd127" or "-8'sd42".

    Raises:
        ValueError: If val is outside the valid range [-127, 127].
    """
    if val < -127 or val > 127:
        raise ValueError(
            f"Weight literal {val} out of valid range [-127, 127] per Rule N2 and Invariant I2."
        )

    if val < 0:
        return f"-8'sd{abs(val)}"
    return f"8'sd{val}"


def encode_product_term(
    x_signal: str,
    w_val: int,
    product_signal: str,
    k: int = 0,
) -> EncodedTerm | None:
    """Generate a single MAC product term with sparsity optimizations.

    Args:
        x_signal: Name of the signed 8-bit input wire (e.g. "x_3").
        w_val: Integer weight value in [-127, 127].
        product_signal: Name for the 16-bit signed product wire (e.g. "p_0_3").
        k: Column index of the weight.

    Returns:
        EncodedTerm if non-zero, or None if w_val == 0 (sparsity omission).
    """
    if w_val == 0:
        return None

    if w_val == 1:
        # Pass-through: sign-extend 8-bit input to 16-bit product
        decl = f"wire signed [15:0] {product_signal} = {{{{8{{{x_signal}[7]}}}}, {x_signal}}};"
        return EncodedTerm(
            k=k,
            w_val=w_val,
            product_wire=product_signal,
            wire_declaration=decl,
            term_type="pass",
        )

    if w_val == -1:
        # Negation: negate sign-extended 8-bit input
        decl = f"wire signed [15:0] {product_signal} = -{{{{8{{{x_signal}[7]}}}}, {x_signal}}};"
        return EncodedTerm(
            k=k,
            w_val=w_val,
            product_wire=product_signal,
            wire_declaration=decl,
            term_type="negate",
        )

    # General multiplication: signed wire * signed literal
    literal = format_weight_literal(w_val)
    decl = f"wire signed [15:0] {product_signal} = {x_signal} * {literal};"
    return EncodedTerm(
        k=k,
        w_val=w_val,
        product_wire=product_signal,
        wire_declaration=decl,
        term_type="mult",
    )


def encode_row_accumulation(
    row_idx: int,
    weights_row: np.ndarray,
    x_prefix: str = "x",
    p_prefix: str = "p",
    acc_bits: int = 32,
) -> tuple[list[str], str]:
    """Generate product wire declarations and accumulation expression for one row.

    Args:
        row_idx: Row index n within the tile or module.
        weights_row: 1D array of signed INT8 weights for this row.
        x_prefix: Prefix for input wire names.
        p_prefix: Prefix for product wire names.
        acc_bits: Accumulator bit width (default 32).

    Returns:
        Tuple of (wire_declarations, sum_expression).
    """
    declarations: list[str] = []
    terms: list[str] = []
    pad_bits = acc_bits - 16

    for k, w in enumerate(weights_row):
        w_int = int(w)
        x_name = f"{x_prefix}_{k}"
        p_name = f"{p_prefix}_{k}"
        term = encode_product_term(x_name, w_int, p_name, k=k)
        if term is not None:
            declarations.append(term.wire_declaration)
            # Explicit sign-extension to accumulator width
            sign_ext_term = f"{{{{{pad_bits}{{{term.product_wire}[15]}}}}, {term.product_wire}}}"
            terms.append(sign_ext_term)

    if not terms:
        sum_expr = f"{acc_bits}'sd0"
    else:
        sum_expr = " + ".join(terms)

    return declarations, sum_expr
