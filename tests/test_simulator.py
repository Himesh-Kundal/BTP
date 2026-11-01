"""Tests for simulation runner and negative control (verification/simulator.py)."""

import json
from pathlib import Path

import numpy as np

from backend.codegen import generate_fixed_core
from backend.testbench import generate_core_testbench
from backend.vectors import write_hex_vectors
from ir.linear import LinearOp
from verification.reference import linear_reference_int64
from verification.simulator import run_simulation


def setup_toy_simulation(work_dir: Path, corrupt_y: bool = False):
    """Helper to set up a toy 1x3 simulation environment."""
    w = np.array([[3, 7, -2]], dtype=np.int8)
    op = LinearOp(name="toy_1x3", in_features=3, out_features=1, weight=w)

    x = np.array([
        [10, -4, 5],
        [-1, 2, -3],
        [-128, 0, 127],
        [0, 0, 0],
    ], dtype=np.int8)

    y_exp = linear_reference_int64(x, w)

    if corrupt_y:
        # Deliberately corrupt expected output for lane 0 of vector 1
        y_exp[1, 0] += 999

    write_hex_vectors(x, y_exp, work_dir)

    rtl_code = generate_fixed_core(op, module_name="llmhw_toy_1x3_core")
    dut_path = work_dir / "dut.v"
    dut_path.write_text(rtl_code, encoding="utf-8")

    tb_code = generate_core_testbench(op, num_vectors=len(x), module_name="llmhw_toy_1x3_core")
    tb_path = work_dir / "tb.v"
    tb_path.write_text(tb_code, encoding="utf-8")

    return dut_path, tb_path, len(x)


def test_simulation_passes_clean_toy(tmp_path: Path):
    """Verify that a clean toy design passes simulation and produces valid report.json."""
    dut_path, tb_path, num_vecs = setup_toy_simulation(tmp_path, corrupt_y=False)
    report_file = tmp_path / "report.json"

    result = run_simulation(
        dut_path=dut_path,
        tb_path=tb_path,
        expected_checked=num_vecs,
        work_dir=tmp_path,
        report_file=report_file,
    )

    assert result.status == "PASS"
    assert result.checked == 4
    assert result.mismatch == 0
    assert result.timing_err == 0
    assert report_file.exists()

    with open(report_file, encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "PASS"
    assert data["checked"] == 4
    assert data["mismatch"] == 0


def test_simulation_negative_control_corrupted_vectors(tmp_path: Path):
    """Negative control: corrupted expected vector must be detected as FAIL."""
    dut_path, tb_path, num_vecs = setup_toy_simulation(tmp_path, corrupt_y=True)
    report_file = tmp_path / "report_fail.json"

    result = run_simulation(
        dut_path=dut_path,
        tb_path=tb_path,
        expected_checked=num_vecs,
        work_dir=tmp_path,
        report_file=report_file,
    )

    assert result.status == "FAIL"
    assert result.checked == 4
    assert result.mismatch >= 1


def test_simulation_zero_checked_triggers_failure(tmp_path: Path):
    """If checked count does not match expected_checked, must be FAIL."""
    dut_path, tb_path, _ = setup_toy_simulation(tmp_path, corrupt_y=False)

    # Claim we expected 10 vectors when only 4 were checked
    result = run_simulation(
        dut_path=dut_path,
        tb_path=tb_path,
        expected_checked=10,
        work_dir=tmp_path,
    )

    assert result.status == "FAIL"
