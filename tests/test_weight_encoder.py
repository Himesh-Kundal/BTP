"""Tests for weight literal encoder and sparsity optimization (T4.3)."""

import numpy as np
import pytest

from backend.weight_encoder import (
    EncodedTerm,
    encode_product_term,
    encode_row_accumulation,
    format_weight_literal,
)


def test_format_weight_literal_valid():
    """Verify formatting of INT8 weights as signed Verilog-2001 literals."""
    assert format_weight_literal(0) == "8'sd0"
    assert format_weight_literal(1) == "8'sd1"
    assert format_weight_literal(127) == "8'sd127"
    assert format_weight_literal(-1) == "-8'sd1"
    assert format_weight_literal(-127) == "-8'sd127"
    assert format_weight_literal(42) == "8'sd42"
    assert format_weight_literal(-42) == "-8'sd42"


def test_format_weight_literal_out_of_range():
    """Verify out-of-range weights raise ValueError."""
    with pytest.raises(ValueError, match="Weight literal"):
        format_weight_literal(128)
    with pytest.raises(ValueError, match="Weight literal"):
        format_weight_literal(-128)


def test_encode_product_term_zero():
    """Verify zero weight terms are omitted completely (returns None)."""
    assert encode_product_term("x_0", 0, "p_0_0") is None


def test_encode_product_term_pass_and_negate():
    """Verify +-1 weights use pass-through or negation wires without multipliers."""
    # w = +1
    term_pos = encode_product_term("x_1", 1, "p_0_1")
    assert isinstance(term_pos, EncodedTerm)
    assert term_pos.term_type == "pass"
    assert "*" not in term_pos.wire_declaration
    assert "wire signed [15:0] p_0_1 = {{8{x_1[7]}}, x_1};" in term_pos.wire_declaration

    # w = -1
    term_neg = encode_product_term("x_2", -1, "p_0_2")
    assert isinstance(term_neg, EncodedTerm)
    assert term_neg.term_type == "negate"
    assert "*" not in term_neg.wire_declaration
    assert "wire signed [15:0] p_0_2 = -{{8{x_2[7]}}, x_2};" in term_neg.wire_declaration


def test_encode_product_term_multiplication():
    """Verify general non-zero weights use constant multiplication."""
    term_mult_pos = encode_product_term("x_3", 42, "p_0_3")
    assert isinstance(term_mult_pos, EncodedTerm)
    assert term_mult_pos.term_type == "mult"
    assert "wire signed [15:0] p_0_3 = x_3 * 8'sd42;" in term_mult_pos.wire_declaration

    term_mult_neg = encode_product_term("x_4", -127, "p_0_4")
    assert isinstance(term_mult_neg, EncodedTerm)
    assert term_mult_neg.term_type == "mult"
    assert "wire signed [15:0] p_0_4 = x_4 * -8'sd127;" in term_mult_neg.wire_declaration


def test_encode_row_accumulation_dense():
    """Verify row accumulation with mixed weights generates correct declarations and sum."""
    weights = np.array([3, -1, 0, 1, -7], dtype=np.int8)
    decls, sum_expr = encode_row_accumulation(row_idx=0, weights_row=weights, x_prefix="x", p_prefix="p_0")

    # 4 non-zero terms (0 is omitted)
    assert len(decls) == 4
    # All 4 products sign-extended to 32 bits in sum
    assert sum_expr.count("{{16{") == 4
    assert "p_0_0" in sum_expr
    assert "p_0_1" in sum_expr
    assert "p_0_2" not in sum_expr  # zero term omitted
    assert "p_0_3" in sum_expr
    assert "p_0_4" in sum_expr


def test_encode_row_accumulation_all_zeros():
    """Verify all-zero row generates no product wires and constant 0 sum."""
    weights = np.zeros(8, dtype=np.int8)
    decls, sum_expr = encode_row_accumulation(row_idx=0, weights_row=weights, x_prefix="x", p_prefix="p_0")

    assert len(decls) == 0
    assert sum_expr == "32'sd0"
