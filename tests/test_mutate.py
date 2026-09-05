"""Tests for mutation testing suite (verification/mutate.py)."""

from pathlib import Path

import numpy as np

from backend.codegen import generate_fixed_core
from backend.testbench import generate_core_testbench
from backend.vectors import write_hex_vectors
from ir.linear import LinearOp
from verification.mutate import generate_mutants, run_mutation_testing
from verification.reference import linear_reference_int64


def test_generate_mutants_types():
    """Verify that M1, M2, and M3 mutants are produced from RTL."""
    w = np.array([[3, 7, -2]], dtype=np.int8)
    op = LinearOp(name="toy_1x3", in_features=3, out_features=1, weight=w)
    rtl = generate_fixed_core(op, module_name="llmhw_toy_1x3_core")

    mutants = generate_mutants(rtl)
    assert len(mutants) >= 3

    types = {m.mutant_type for m in mutants}
    assert "M1" in types  # Sign flip
    assert "M2" in types  # Weight +1
    assert "M3" in types  # Drop term


def test_mutation_suite_kills_all_toy_mutants(tmp_path: Path):
    """Run mutation testing on toy 1x3 layer and verify 100% kill rate."""
    w = np.array([[3, 7, -2]], dtype=np.int8)
    op = LinearOp(name="toy_1x3", in_features=3, out_features=1, weight=w)

    x = np.array([
        [10, -4, 5],
        [-1, 2, -3],
        [-128, 0, 127],
        [1, 0, 0],  # One-hot lane 0
        [0, 1, 0],  # One-hot lane 1
        [0, 0, 1],  # One-hot lane 2
    ], dtype=np.int8)

    y_exp = linear_reference_int64(x, w)

    write_hex_vectors(x, y_exp, tmp_path)

    rtl_code = generate_fixed_core(op, module_name="llmhw_toy_1x3_core")
    dut_path = tmp_path / "dut.v"
    dut_path.write_text(rtl_code, encoding="utf-8")

    tb_code = generate_core_testbench(op, num_vectors=len(x), module_name="llmhw_toy_1x3_core")
    tb_path = tmp_path / "tb.v"
    tb_path.write_text(tb_code, encoding="utf-8")

    report = run_mutation_testing(
        dut_path=dut_path,
        tb_path=tb_path,
        num_vectors=len(x),
        work_dir=tmp_path / "mutation_runs",
        rtl_code=rtl_code,
    )

    assert report["total"] >= 3
    assert report["killed"] == report["total"]
    assert report["survived"] == 0
    assert report["kill_rate"] == 1.0
