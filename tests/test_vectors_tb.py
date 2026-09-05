"""Tests for vector serialization and testbench generator (backend/vectors.py, backend/testbench.py)."""

import subprocess
from pathlib import Path

import numpy as np

from backend.codegen import generate_fixed_core
from backend.testbench import generate_core_testbench
from backend.vectors import write_hex_vectors
from ir.linear import LinearOp
from verification.reference import linear_reference_int64


def test_write_hex_vectors_format(tmp_path: Path):
    """Verify x.hex (2 hex digits) and y_exp.hex (8 hex digits)."""
    # x has negative and positive int8 values
    x = np.array([[-1, 0, 127, -128]], dtype=np.int8)  # M=1, K=4
    # y has negative and positive int32 values
    y = np.array([[-1, 2147483647]], dtype=np.int64)  # M=1, N=2

    x_path, y_path = write_hex_vectors(x, y, tmp_path)

    x_lines = x_path.read_text().strip().splitlines()
    assert x_lines == ["ff", "00", "7f", "80"]

    y_lines = y_path.read_text().strip().splitlines()
    assert y_lines == ["ffffffff", "7fffffff"]


def test_testbench_execution_on_1x3_toy(tmp_path: Path):
    """Generate RTL, vectors, testbench, compile with iverilog, run with vvp, and check TB_RESULT."""
    w = np.array([[3, 7, -2]], dtype=np.int8)
    op = LinearOp(name="toy_1x3", in_features=3, out_features=1, weight=w)

    # 4 test vectors
    x = np.array([
        [10, -4, 5],    # 10*3 + (-4)*7 + 5*(-2) = -8
        [-1, 2, -3],    # -3 + 14 + 6 = 17
        [-128, 0, 127], # -128*3 + 127*(-2) = -384 - 254 = -638
        [0, 0, 0],      # 0
    ], dtype=np.int8)

    y_exp = linear_reference_int64(x, w)

    # Write files to tmp_path
    write_hex_vectors(x, y_exp, tmp_path)

    rtl_code = generate_fixed_core(op, module_name="llmhw_toy_1x3_core")
    (tmp_path / "dut.v").write_text(rtl_code, encoding="utf-8")

    tb_code = generate_core_testbench(op, num_vectors=len(x), module_name="llmhw_toy_1x3_core")
    (tmp_path / "tb.v").write_text(tb_code, encoding="utf-8")

    # Compile with iverilog
    sim_vvp = tmp_path / "sim.vvp"
    cmd_compile = [
        "iverilog", "-g2005", "-Wall",
        "-o", str(sim_vvp),
        str(tmp_path / "dut.v"),
        str(tmp_path / "tb.v")
    ]
    res_c = subprocess.run(cmd_compile, check=False, capture_output=True, text=True)
    assert res_c.returncode == 0, f"iverilog compile failed:\n{res_c.stderr}"

    # Run with vvp
    cmd_sim = ["vvp", str(sim_vvp)]
    res_s = subprocess.run(cmd_sim, check=False, capture_output=True, text=True, cwd=str(tmp_path))
    assert res_s.returncode == 0, f"vvp execution failed:\n{res_s.stderr}"

    stdout = res_s.stdout
    assert "TB_RESULT CHECKED=4 MISMATCH=0 TIMING_ERR=0" in stdout
