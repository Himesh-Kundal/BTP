"""Tests for parametric Verilog tile core code generator (T4.4)."""

import subprocess
from pathlib import Path

import numpy as np

from backend.verilog_codegen import generate_tile_core


def test_tile_core_signature_and_ports():
    """Verify module signature matches docs/02 §6 for various Nt x Kt shapes."""
    w = np.array([[3, 7, -2]], dtype=np.int8)  # 1x3
    rtl = generate_tile_core("tile_1x3_core", w, acc_bits=32, manifest_hash="deadbeef")

    assert "module tile_1x3_core (" in rtl
    assert "input  wire [23:0] x_flat," in rtl
    assert "output wire [31:0] y_flat" in rtl
    assert "Manifest SHA256: deadbeef" in rtl
    assert "w_" not in rtl  # Zero weight ports


def test_tile_core_golden_match():
    """Verify generated Verilog lines for a small 2x4 matrix with zero and unity weights."""
    w = np.array([[1, 0, -1, 5], [0, -127, 127, 0]], dtype=np.int8)
    rtl = generate_tile_core("tile_2x4_core", w, acc_bits=32)

    # Check unpacking
    assert "wire signed [7:0] x_0 = x_flat[7:0];" in rtl
    assert "wire signed [7:0] x_3 = x_flat[31:24];" in rtl

    # Row 0:
    # w=1 -> pass-through wire
    assert "wire signed [15:0] p_0_0 = {{8{x_0[7]}}, x_0};" in rtl
    # w=0 -> omitted
    assert "p_0_1" not in rtl
    # w=-1 -> negation wire
    assert "wire signed [15:0] p_0_2 = -{{8{x_2[7]}}, x_2};" in rtl
    # w=5 -> mult
    assert "wire signed [15:0] p_0_3 = x_3 * 8'sd5;" in rtl

    # Row 1:
    assert "wire signed [15:0] p_1_1 = x_1 * -8'sd127;" in rtl
    assert "wire signed [15:0] p_1_2 = x_2 * 8'sd127;" in rtl

    # Output packing
    assert "assign y_flat[31:0] = y_0;" in rtl
    assert "assign y_flat[63:32] = y_1;" in rtl


def test_tile_core_iverilog_and_verilator_lint(tmp_path: Path):
    """Verify that generated tile core compiles cleanly with iverilog and verilator."""
    rng = np.random.default_rng(42)
    # Standard 8x16 tile
    w = rng.integers(-127, 127, size=(8, 16), dtype=np.int8)
    # Inject some zeros and +-1
    w[0, 0] = 0
    w[1, 1] = 1
    w[2, 2] = -1

    rtl = generate_tile_core("llmhw_tile_8x16_core", w, acc_bits=32, manifest_hash="0123456789abcdef")
    rtl_file = tmp_path / "llmhw_tile_8x16_core.v"
    rtl_file.write_text(rtl)

    # 1. Icarus Verilog lint
    cmd_iverilog = ["iverilog", "-g2005", "-Wall", "-t", "null", str(rtl_file)]
    res_iv = subprocess.run(cmd_iverilog, capture_output=True, text=True)
    assert res_iv.returncode == 0, f"iverilog failed:\nSTDOUT: {res_iv.stdout}\nSTDERR: {res_iv.stderr}"

    # 2. Verilator lint
    cmd_verilator = ["verilator", "--lint-only", "-Wall", str(rtl_file)]
    res_vl = subprocess.run(cmd_verilator, capture_output=True, text=True)
    assert res_vl.returncode == 0, f"verilator failed:\nSTDOUT: {res_vl.stdout}\nSTDERR: {res_vl.stderr}"
