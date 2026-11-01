"""Tests for fixed-weight RTL code generator (backend/codegen.py)."""

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from backend.codegen import generate_fixed_core
from ir.linear import LinearOp

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "verification" / "golden"


@pytest.fixture(autouse=True)
def ensure_golden_dir():
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)


def test_codegen_1x3_toy():
    """Verify code generation and golden match for 1x3 toy."""
    w = np.array([[3, 7, -2]], dtype=np.int8)  # N=1, K=3
    op = LinearOp(name="toy_1x3", in_features=3, out_features=1, weight=w)

    rtl = generate_fixed_core(op, module_name="llmhw_toy_1x3_core")

    # Invariant I3: No weight input ports
    assert "input" in rtl
    assert "x_flat" in rtl
    assert "y_flat" in rtl
    assert not re.search(r"input\s+.*w_", rtl)
    assert not re.search(r"input\s+.*weight", rtl)

    # Check signedness conventions
    assert "8'sd3" in rtl
    assert "8'sd7" in rtl
    assert "-8'sd2" in rtl
    assert "wire signed [7:0] x_0" in rtl
    assert "wire signed [15:0]" in rtl
    assert "{{16{" in rtl

    # Golden comparison
    golden_path = GOLDEN_DIR / "toy_1x3_core.v"
    if not golden_path.exists():
        golden_path.write_text(rtl, encoding="utf-8")
    golden_rtl = golden_path.read_text(encoding="utf-8")
    assert rtl == golden_rtl


def test_codegen_2x4_toy_with_bias_and_zeros():
    """Verify 2x4 layer with zero-weights omitted and bias added."""
    # Row 0: [1, 0, -1, 5] (has 1, 0, -1, general constant)
    # Row 1: [0, 2, -3, 0] (has multiple zeros)
    w = np.array([
        [1, 0, -1, 5],
        [0, 2, -3, 0],
    ], dtype=np.int8)
    b = np.array([10, -20], dtype=np.int32)
    op = LinearOp(name="toy_2x4", in_features=4, out_features=2, weight=w, bias=b)

    rtl = generate_fixed_core(op, module_name="llmhw_toy_2x4_core")

    # Zero weights should NOT have product multipliers
    assert "prod_0_1" not in rtl  # W[0, 1] is 0
    assert "prod_1_0" not in rtl  # W[1, 0] is 0
    assert "prod_1_3" not in rtl  # W[1, 3] is 0

    # Non-zero terms present
    assert "prod_0_0" in rtl  # W[0, 0] is 1
    assert "prod_0_2" in rtl  # W[0, 2] is -1
    assert "prod_0_3" in rtl  # W[0, 3] is 5

    # Bias present
    assert "32'sd10" in rtl
    assert "- 32'sd20" in rtl

    # Golden comparison
    golden_path = GOLDEN_DIR / "toy_2x4_core.v"
    if not golden_path.exists():
        golden_path.write_text(rtl, encoding="utf-8")
    golden_rtl = golden_path.read_text(encoding="utf-8")
    assert rtl == golden_rtl


def test_codegen_iverilog_lint(tmp_path):
    """Verify that generated RTL passes iverilog -g2005 -Wall clean."""
    w = np.array([[3, 7, -2]], dtype=np.int8)
    op = LinearOp(name="toy_1x3", in_features=3, out_features=1, weight=w)
    rtl = generate_fixed_core(op, module_name="llmhw_toy_1x3_core")

    v_file = tmp_path / "core.v"
    v_file.write_text(rtl, encoding="utf-8")

    cmd = ["iverilog", "-g2005", "-Wall", "-tvvp", "-o", str(tmp_path / "out.vvp"), str(v_file)]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert res.returncode == 0, f"iverilog lint failed:\n{res.stderr}"
