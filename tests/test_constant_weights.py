"""Tests for constant weight specialization and port stripping (T4.1, Invariant I3)."""

import numpy as np
import pytest

from backend.constant_weights import (
    get_design_f_interface_ports,
    specialize_constant_weights,
    verify_design_f_ports,
)
from ir.linear import LinearOp
from ir.tensor import TensorKind, TensorRef


def test_specialize_constant_weights_success():
    """Verify that a valid quantized constant LinearOp specializes cleanly."""
    w_q = np.array([[10, -20, 30], [5, 0, -15]], dtype=np.int8)
    op = LinearOp(
        name="test_linear",
        in_features=3,
        out_features=2,
        weight=w_q,
        is_quantized=True,
    )
    assert op.weight.kind == TensorKind.CONSTANT

    specialized = specialize_constant_weights(op)
    assert specialized.weight.kind == TensorKind.CONSTANT
    assert specialized.hw["arch"] == "fixed"
    assert specialized.is_quantized


def test_specialize_constant_weights_rejects_non_constant():
    """Negative control: verify that non-constant weights raise ValueError."""
    w_q = np.array([[10, -20, 30], [5, 0, -15]], dtype=np.int8)
    w_ref = TensorRef(
        name="weight",
        shape=(2, 3),
        dtype="int8",
        kind=TensorKind.VARIABLE,
        data=w_q,
    )
    op = LinearOp(
        name="test_linear",
        in_features=3,
        out_features=2,
        weight=w_ref,
        is_quantized=True,
    )

    with pytest.raises(ValueError, match="Design F requires constant weights"):
        specialize_constant_weights(op)

    # Also test INPUT kind
    w_ref_input = TensorRef(
        name="weight",
        shape=(2, 3),
        dtype="int8",
        kind=TensorKind.INPUT,
        data=w_q,
    )
    op_input = LinearOp(
        name="test_linear",
        in_features=3,
        out_features=2,
        weight=w_ref_input,
        is_quantized=True,
    )
    with pytest.raises(ValueError, match="Design F requires constant weights"):
        specialize_constant_weights(op_input)


def test_specialize_constant_weights_rejects_unquantized():
    """Negative control: verify unquantized op cannot be specialized for RTL."""
    w = np.array([[0.5, -0.2], [0.1, 0.4]], dtype=np.float32)
    op = LinearOp(
        name="test_unquantized",
        in_features=2,
        out_features=2,
        weight=w,
        is_quantized=False,
    )
    with pytest.raises(ValueError, match="must be quantized"):
        specialize_constant_weights(op)


def test_verify_design_f_ports_valid():
    """Verify that conforming Design F port lists pass verification."""
    core_ports = ["x_flat", "y_flat"]
    assert verify_design_f_ports(core_ports) is True

    wrapper_ports = ["clk", "rst_n", "valid_in", "x_flat", "valid_out", "y_flat"]
    assert verify_design_f_ports(wrapper_ports) is True


def test_verify_design_f_ports_rejects_weight_ports():
    """Negative control: verify that injecting runtime weight ports raises ValueError."""
    forbidden_ports = [
        ["clk", "rst_n", "valid_in", "x_flat", "w_data", "valid_out", "y_flat"],
        ["clk", "rst_n", "valid_in", "x_flat", "w_we", "valid_out", "y_flat"],
        ["clk", "rst_n", "valid_in", "x_flat", "w_addr", "valid_out", "y_flat"],
        ["clk", "rst_n", "valid_in", "x_flat", "w_flat", "valid_out", "y_flat"],
        ["x_flat", "weight_in", "y_flat"],
        ["x_flat", "w_en", "y_flat"],
    ]
    for ports in forbidden_ports:
        with pytest.raises(ValueError, match="Forbidden weight port"):
            verify_design_f_ports(ports)


def test_get_design_f_interface_ports():
    """Verify interface ports definitions match docs/02 §6."""
    core_spec = get_design_f_interface_ports(in_features=576, out_features=576, is_wrapper=False)
    assert set(core_spec.keys()) == {"x_flat", "y_flat"}
    assert core_spec["x_flat"]["width"] == 8 * 576
    assert core_spec["x_flat"]["direction"] == "input"
    assert core_spec["x_flat"]["signed"] is True
    assert core_spec["y_flat"]["width"] == 32 * 576
    assert core_spec["y_flat"]["direction"] == "output"
    assert core_spec["y_flat"]["signed"] is True

    wrapper_spec = get_design_f_interface_ports(in_features=16, out_features=8, is_wrapper=True)
    expected_wrapper_ports = {"clk", "rst_n", "valid_in", "x_flat", "valid_out", "y_flat"}
    assert set(wrapper_spec.keys()) == expected_wrapper_ports
    assert wrapper_spec["clk"]["width"] == 1
    assert wrapper_spec["rst_n"]["width"] == 1
    assert wrapper_spec["valid_in"]["width"] == 1
    assert wrapper_spec["x_flat"]["width"] == 8 * 16
    assert wrapper_spec["valid_out"]["width"] == 1
    assert wrapper_spec["y_flat"]["width"] == 32 * 8
