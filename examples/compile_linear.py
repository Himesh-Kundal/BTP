"""End-to-end toy compiler demo (walking skeleton for Phase 1).

Executes:
1. Builds LinearOp IR for toy linear layer (1x3: W=[3, 7, -2]).
2. Generates fixed-weight Verilog-2001 core (no weight ports).
3. Evaluates golden reference using NumPy int64 oracle.
4. Generates hex vectors and self-checking testbench.
5. Runs simulation in Icarus Verilog and asserts bit-exact match.
6. Injects text-level mutations (M1, M2, M3) and asserts 100% kill rate.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from backend.codegen import generate_fixed_core
from backend.testbench import generate_core_testbench
from backend.vectors import write_hex_vectors
from ir.linear import LinearOp
from verification.mutate import run_mutation_testing
from verification.reference import linear_reference_int64
from verification.simulator import run_simulation


def run_toy_demo(output_dir: Path | None = None) -> bool:
    repo_root = Path(__file__).resolve().parent.parent
    if output_dir is None:
        build_dir = repo_root / "build" / "demo_toy"
    else:
        build_dir = Path(output_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print(" LLMHW COMPILER — WALKING SKELETON DEMO (TOY LINEAR LAYER)")
    print("================================================================================")

    # 1. IR Definition
    w = np.array([[3, 7, -2]], dtype=np.int8)  # N=1, K=3
    op = LinearOp(name="toy_1x3", in_features=3, out_features=1, weight=w)
    print(f"[*] IR Model: LinearOp(name='{op.name}', in_features={op.in_features}, out_features={op.out_features})")
    print(f"[*] Weights:  {w.tolist()}")

    # 2. Golden Reference & Stimulus Generation
    # Includes normal vectors, extremes (-128, 127), and one-hot vectors
    x = np.array([
        [10, -4, 5],
        [-1, 2, -3],
        [-128, 0, 127],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ], dtype=np.int8)
    y_exp = linear_reference_int64(x, w)
    print(f"[*] Generated {len(x)} test vectors covering extremes and one-hot lanes.")

    # 3. Vector serialization
    write_hex_vectors(x, y_exp, build_dir)

    # 4. Verilog Code Generation
    rtl_code = generate_fixed_core(op, module_name="llmhw_toy_1x3_core")
    dut_path = build_dir / "dut.v"
    dut_path.write_text(rtl_code, encoding="utf-8")
    print(f"[*] Generated fixed-weight RTL core: {dut_path.name} ({len(rtl_code.splitlines())} lines)")

    # 5. Testbench Generation
    tb_code = generate_core_testbench(op, num_vectors=len(x), module_name="llmhw_toy_1x3_core")
    tb_path = build_dir / "tb.v"
    tb_path.write_text(tb_code, encoding="utf-8")
    print(f"[*] Generated self-checking testbench: {tb_path.name}")

    # 6. Simulation Execution
    print("\n--- Running Simulation in Icarus Verilog ---")
    sim_report_file = repo_root / "results" / "demo_toy_report.json"
    sim_res = run_simulation(
        dut_path=dut_path,
        tb_path=tb_path,
        expected_checked=len(x),
        work_dir=build_dir / "sim_run",
        dump_vcd=False,
        report_file=sim_report_file,
    )

    print(f"Status:       {sim_res.status}")
    print(f"Checked:      {sim_res.checked} vectors")
    print(f"Mismatches:   {sim_res.mismatch}")
    print(f"Elapsed:      {sim_res.elapsed_seconds * 1000:.2f} ms")

    if sim_res.status != "PASS":
        print(f"[-] SIMULATION FAILED:\n{sim_res.stderr}")
        return False

    # 7. Mutation Testing Suite (M1-M3)
    print("\n--- Running Mutation Testing Suite (M1-M3) ---")
    mut_report = run_mutation_testing(
        dut_path=dut_path,
        tb_path=tb_path,
        num_vectors=len(x),
        work_dir=build_dir / "mutants",
        rtl_code=rtl_code,
    )

    print(f"{'Mutant ID':<22} | {'Type':<6} | {'Status':<8} | {'Description'}")
    print("-" * 80)
    for m in mut_report["mutants"]:
        status_str = "KILLED" if m["killed"] else "SURVIVED"
        print(f"{m['id']:<22} | {m['type']:<6} | {status_str:<8} | {m['description']}")

    print("-" * 80)
    print(f"Total Mutants: {mut_report['total']}")
    print(f"Killed:        {mut_report['killed']}")
    print(f"Survived:      {mut_report['survived']}")
    print(f"Kill Rate:     {mut_report['kill_rate'] * 100:.1f}%")

    if mut_report["kill_rate"] < 1.0:
        print("[-] MUTATION TESTING FAILED: Some mutants survived!")
        return False

    print("\n================================================================================")
    print(" WALKING SKELETON: ALL CHECKS PASSED (BIT-EXACT + 100% MUTATIONS KILLED)")
    print("================================================================================")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and verify toy linear layer")
    parser.add_argument("--toy", action="store_true", help="Run toy walking skeleton")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    success = run_toy_demo(output_dir=Path(args.out) if args.out else None)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
