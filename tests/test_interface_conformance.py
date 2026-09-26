"""Tests for RTL interface conformance and weight port audit (T4.6)."""

from pathlib import Path

import numpy as np
import pytest

from backend.module_generator import generate_full_module
from backend.tiling import tile_weight_matrix
from ir.linear import LinearOp
from verification.weight_audit import audit_rtl_file, audit_rtl_source


def test_interface_conformance_valid_module():
    """Verify that conforming Design F RTL passes audit with zero violations."""
    w = np.array([[10, -20, 30], [5, 0, -15]], dtype=np.int8)  # 2x3
    op = LinearOp(name="conf_test", in_features=3, out_features=2, weight=w, is_quantized=True)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=3, layer_name=op.name)

    rtl = generate_full_module(op, manifest, tile_n=2, tile_k=3, latency=1)

    result = audit_rtl_source(rtl, expected_k=3, expected_n=2, is_wrapper=True)
    assert result["passed"] is True
    assert len(result["violations"]) == 0
    assert result["weight_ports_found"] == 0
    assert result["w_signals_found"] == 0
    assert result["storage_found"] == 0


def test_negative_control_injected_weight_port():
    """Negative control: verify that injecting runtime weight input ports triggers failure."""
    valid_rtl = """
module llmhw_test (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [23:0] x_flat,
    input  wire [7:0]  w_data,
    output reg         valid_out,
    output reg  [63:0] y_flat
);
endmodule
"""
    result = audit_rtl_source(valid_rtl, expected_k=3, expected_n=2, is_wrapper=True)
    assert result["passed"] is False
    assert any("weight port" in v.lower() for v in result["violations"])

    with pytest.raises(ValueError, match="Weight port found"):
        audit_rtl_source(valid_rtl, expected_k=3, expected_n=2, strict=True)


def test_negative_control_injected_w_signal():
    """Negative control: verify that internal w_* signals trigger failure."""
    valid_rtl = """
module llmhw_test (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [23:0] x_flat,
    output reg         valid_out,
    output reg  [63:0] y_flat
);
    wire signed [7:0] w_weight_wire;
endmodule
"""
    result = audit_rtl_source(valid_rtl, expected_k=3, expected_n=2, is_wrapper=True)
    assert result["passed"] is False
    assert any("w_*" in v.lower() or "forbidden" in v.lower() for v in result["violations"])


def test_negative_control_injected_storage():
    """Negative control: verify that memory arrays (weight registers) trigger failure."""
    valid_rtl = """
module llmhw_test (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [23:0] x_flat,
    output reg         valid_out,
    output reg  [63:0] y_flat
);
    reg [7:0] weight_storage [0:15];
endmodule
"""
    result = audit_rtl_source(valid_rtl, expected_k=3, expected_n=2, is_wrapper=True)
    assert result["passed"] is False
    assert any("storage" in v.lower() or "memory" in v.lower() for v in result["violations"])


def test_negative_control_injected_delays_and_display():
    """Negative control: verify that # delays and $display in synthesizable RTL trigger failure."""
    rtl_delay = """
module llmhw_test (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [23:0] x_flat,
    output reg         valid_out,
    output reg  [63:0] y_flat
);
    always @(posedge clk) #5 valid_out <= valid_in;
endmodule
"""
    result = audit_rtl_source(rtl_delay, is_wrapper=True)
    assert result["passed"] is False
    assert any("delay" in v.lower() for v in result["violations"])


def test_audit_rtl_file(tmp_path: Path):
    """Verify file-based audit on generated RTL."""
    w = np.zeros((4, 8), dtype=np.int8)
    op = LinearOp(name="file_test", in_features=8, out_features=4, weight=w, is_quantized=True)
    manifest, _ = tile_weight_matrix(w, tile_n=4, tile_k=8, layer_name=op.name)

    rtl = generate_full_module(op, manifest, tile_n=4, tile_k=8, latency=1)
    file_path = tmp_path / "layer.v"
    file_path.write_text(rtl)

    passed = audit_rtl_file(file_path, expected_k=8, expected_n=4, is_wrapper=True, strict=True)
    assert passed is True
