"""Tests for Design P baseline generator (T5.1).

Verifies:
- Design P tile core generation with generic signed multipliers
- Design P full module generation with W_reg register array, w_we, w_addr, w_data
- Design P_bus variant with primary-input w_flat bus
- Width conformance matching Rule N10 (x: 8, w: 8, prod: 16, acc: 32)
- Clean linting under iverilog, Verilator, and Yosys check -assert
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from backend.baseline_p import (
    generate_full_module_p,
    generate_full_module_p_bus,
    generate_tile_core_p,
)
from backend.tiling import tile_weight_matrix
from ir.linear import LinearOp


def test_tile_core_p_signature_and_widths():
    """Verify tile core P interface, signed multiplier widths, and accumulator packing."""
    nt, kt = 2, 3
    code = generate_tile_core_p("test_tile_p", nt=nt, kt=kt, acc_bits=32)

    # Port checks
    assert "input  wire [23:0] x_flat" in code  # 8*3 = 24
    assert "input  wire [47:0] w_flat" in code  # 8*2*3 = 48
    assert "output wire [63:0] y_flat" in code  # 32*2 = 64

    # Arithmetic checks: signed 16-bit products
    assert "wire signed [15:0] prod_0_0 = x_0 * w_0_0;" in code
    assert "wire signed [15:0] prod_1_2 = x_2 * w_1_2;" in code

    # Sign extension checks
    assert "$signed({{16{prod_0_0[15]}}, prod_0_0})" in code

    # Lint checks
    if shutil.which("iverilog"):
        with tempfile.NamedTemporaryFile("w", suffix=".v", delete=False) as f:
            f.write(code)
            fname = f.name
        try:
            res = subprocess.run(
                ["iverilog", "-g2001", "-Wall", "-t", "null", fname],
                capture_output=True,
                text=True,
                check=False,
            )
            assert res.returncode == 0, f"iverilog failed:\n{res.stderr}"
        finally:
            Path(fname).unlink(missing_ok=True)

    if shutil.which("verilator"):
        with tempfile.NamedTemporaryFile("w", suffix=".v", delete=False) as f:
            f.write(code)
            fname = f.name
        try:
            res = subprocess.run(
                [
                    "verilator",
                    "--lint-only",
                    "-Wall",
                    "-Wno-DECLFILENAME",
                    fname,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert res.returncode == 0, f"verilator failed:\n{res.stderr}"
        finally:
            Path(fname).unlink(missing_ok=True)


def test_full_module_p_signature_and_ports():
    """Verify full module P with W_reg array, write ports, and lint cleanliness."""
    w = np.zeros((4, 4), dtype=np.int8)
    op = LinearOp(name="proj_test_p", in_features=4, out_features=4, weight=w)
    manifest, _tiles = tile_weight_matrix(w, tile_n=2, tile_k=2)

    code = generate_full_module_p(op, manifest)

    # Check module ports
    assert "input  wire clk" in code
    assert "input  wire rst_n" in code
    assert "input  wire valid_in" in code
    assert "input  wire [31:0] x_flat" in code  # 8*4 = 32
    assert "input  wire w_we" in code
    assert "input  wire [3:0] w_addr" in code  # 4x4 = 16 words -> 4 bits
    assert "input  wire signed [7:0] w_data" in code
    assert "output reg  valid_out" in code
    assert "[127:0] y_flat" in code  # 32*4 = 128

    # Check W_reg array
    assert "reg signed [7:0] W_reg [0:15];" in code
    assert "W_reg[w_addr] <= w_data;" in code

    # Lint check under iverilog
    if shutil.which("iverilog"):
        with tempfile.NamedTemporaryFile("w", suffix=".v", delete=False) as f:
            f.write(code)
            fname = f.name
        try:
            res = subprocess.run(
                ["iverilog", "-g2001", "-Wall", "-t", "null", fname],
                capture_output=True,
                text=True,
                check=False,
            )
            assert res.returncode == 0, f"iverilog failed:\n{res.stderr}"
        finally:
            Path(fname).unlink(missing_ok=True)

    # Lint check under Verilator
    if shutil.which("verilator"):
        with tempfile.NamedTemporaryFile("w", suffix=".v", delete=False) as f:
            f.write(code)
            fname = f.name
        try:
            res = subprocess.run(
                [
                    "verilator",
                    "--lint-only",
                    "-Wall",
                    "-Wno-DECLFILENAME",
                    fname,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert res.returncode == 0, f"verilator failed:\n{res.stderr}"
        finally:
            Path(fname).unlink(missing_ok=True)

    # Yosys check -assert
    if shutil.which("yosys"):
        with tempfile.NamedTemporaryFile("w", suffix=".v", delete=False) as f:
            f.write(code)
            fname = f.name
        try:
            res = subprocess.run(
                ["yosys", "-p", f"read_verilog {fname}; prep; check -assert"],
                capture_output=True,
                text=True,
                check=False,
            )
            assert res.returncode == 0, f"yosys failed:\n{res.stderr}"
        finally:
            Path(fname).unlink(missing_ok=True)


def test_full_module_p_bus_signature_and_ports():
    """Verify full module P_bus logic-only baseline variant."""
    w = np.zeros((4, 4), dtype=np.int8)
    op = LinearOp(name="proj_test_p_bus", in_features=4, out_features=4, weight=w)
    manifest, _tiles = tile_weight_matrix(w, tile_n=2, tile_k=2)

    code = generate_full_module_p_bus(op, manifest)

    # Check ports: wide w_flat bus, no w_we or w_addr
    assert "input  wire clk" in code
    assert "input  wire rst_n" in code
    assert "input  wire valid_in" in code
    assert "input  wire [31:0] x_flat" in code
    assert "input  wire [127:0] w_flat" in code  # 8*16 = 128
    assert "w_we" not in code
    assert "w_addr" not in code
    assert "W_reg" not in code
    assert "output reg  valid_out" in code
    assert "[127:0] y_flat" in code

    # Lint check under iverilog
    if shutil.which("iverilog"):
        with tempfile.NamedTemporaryFile("w", suffix=".v", delete=False) as f:
            f.write(code)
            fname = f.name
        try:
            res = subprocess.run(
                ["iverilog", "-g2001", "-Wall", "-t", "null", fname],
                capture_output=True,
                text=True,
                check=False,
            )
            assert res.returncode == 0, f"iverilog failed:\n{res.stderr}"
        finally:
            Path(fname).unlink(missing_ok=True)

    # Yosys check -assert
    if shutil.which("yosys"):
        with tempfile.NamedTemporaryFile("w", suffix=".v", delete=False) as f:
            f.write(code)
            fname = f.name
        try:
            res = subprocess.run(
                ["yosys", "-p", f"read_verilog {fname}; prep; check -assert"],
                capture_output=True,
                text=True,
                check=False,
            )
            assert res.returncode == 0, f"yosys failed:\n{res.stderr}"
        finally:
            Path(fname).unlink(missing_ok=True)


def test_width_conformance_rule_n10():
    """Rule N10: Assert Design P and Design F use identical bit widths."""
    from backend.module_generator import generate_full_module

    w = np.ones((5, 11), dtype=np.int8)
    op = LinearOp(name="check_widths", in_features=11, out_features=5, weight=w)
    manifest, _tiles = tile_weight_matrix(w, tile_n=4, tile_k=4)

    code_f = generate_full_module(op, manifest)
    code_p = generate_full_module_p(op, manifest)

    # Both must have identical x_flat width
    # 11 padded to 12 -> 8*12 = 96
    assert "input  wire [95:0] x_flat" in code_f
    assert "input  wire [95:0] x_flat" in code_p

    # Both must have identical y_flat width
    # 8 padded outputs * 32 = 256 bits
    assert "[255:0]        y_flat" in code_f or "[255:0] y_flat" in code_f
    assert "[255:0] y_flat" in code_p
