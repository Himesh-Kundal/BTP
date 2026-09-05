"""CLI script to run simulation on toy linear layer."""

import sys
from pathlib import Path

import numpy as np

from backend.codegen import generate_fixed_core
from backend.testbench import generate_core_testbench
from backend.vectors import write_hex_vectors
from ir.linear import LinearOp
from verification.reference import linear_reference_int64
from verification.simulator import run_simulation


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    work_dir = repo_root / "build" / "toy_sim"
    results_dir = repo_root / "results"
    work_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    w = np.array([[3, 7, -2]], dtype=np.int8)
    op = LinearOp(name="toy_1x3", in_features=3, out_features=1, weight=w)

    x = np.array([
        [10, -4, 5],
        [-1, 2, -3],
        [-128, 0, 127],
        [0, 0, 0],
    ], dtype=np.int8)

    y_exp = linear_reference_int64(x, w)

    write_hex_vectors(x, y_exp, work_dir)

    rtl_code = generate_fixed_core(op, module_name="llmhw_toy_1x3_core")
    dut_path = work_dir / "dut.v"
    dut_path.write_text(rtl_code, encoding="utf-8")

    tb_code = generate_core_testbench(op, num_vectors=len(x), module_name="llmhw_toy_1x3_core")
    tb_path = work_dir / "tb.v"
    tb_path.write_text(tb_code, encoding="utf-8")

    report_path = results_dir / "report.json"
    result = run_simulation(
        dut_path=dut_path,
        tb_path=tb_path,
        expected_checked=len(x),
        work_dir=work_dir,
        report_file=report_path,
    )

    print(f"Simulation {result.status}: checked {result.checked} vectors, {result.mismatch} mismatches.")
    print(f"Report written to: {report_path}")
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
