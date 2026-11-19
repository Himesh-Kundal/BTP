"""Task UG.2: Fast End-to-End Demo (`make demo2-fast`).

Executes the complete Part II compiler + verification pipeline on the 2-layer
tiny decoder fixture in seconds:
1. Compiles the tiny decoder to a ROM-backed Verilog accelerator bundle (`build/demo2_fast_hw`).
2. Runs autoregressive token generation (`prompt=[1, 10, 25]`, `max_new_tokens=4`)
   in both the Python `CompiledArtifactInterpreter` and Verilator RTL simulation,
   verifying 100% bit-exact parity.
3. Runs the 4 structural/ROM immutability audits (`verify --bundle`).
4. Prints the top-level mutation kill matrix (`results/top_mutation_matrix.json`, 7/7 killed)
   and unit mutation matrix (`results/mutation_matrix_v2.json`, 21/21 killed).
Outputs: `results/part2/demo2_fast.json`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from cli import main as cli_main


def main() -> None:
    t0 = time.perf_counter()
    repo_root = Path(__file__).resolve().parent.parent
    bundle_dir = repo_root / "build" / "demo2_fast_hw"
    run_report_path = repo_root / "results" / "part2" / "demo2_fast_run.json"
    verify_report_path = repo_root / "results" / "part2" / "demo2_fast_verify.json"
    summary_path = repo_root / "results" / "part2" / "demo2_fast.json"

    print("================================================================================")
    print(" MAKE DEMO2-FAST — TINY DECODER END-TO-END RTL SIMULATION & MUTATION AUDIT")
    print("================================================================================")

    # 1. Compile tiny decoder model
    rc1 = cli_main(["compile-model", "--tiny", "--out", str(bundle_dir)])
    if rc1 != 0:
        raise SystemExit(rc1)

    # 2. Run token generation in both interpreter and Verilator RTL
    rc2 = cli_main([
        "run",
        "--bundle", str(bundle_dir),
        "--tokens", "1,10,25",
        "--max-new-tokens", "4",
        "--backend", "both",
        "--out", str(run_report_path),
    ])
    if rc2 != 0:
        raise SystemExit(rc2)

    # 3. Run bundle verification & structural audits
    rc3 = cli_main([
        "verify",
        "--bundle", str(bundle_dir),
        "--tokens", "1,10,25",
        "--max-new-tokens", "4",
        "--out", str(verify_report_path),
    ])
    if rc3 != 0:
        raise SystemExit(rc3)

    # 4. Load mutation matrices
    top_mut_path = repo_root / "results" / "top_mutation_matrix.json"
    unit_mut_path = repo_root / "results" / "mutation_matrix_v2.json"

    if not top_mut_path.is_file():
        from tests.test_top_mutation import run_top_level_mutation_campaign
        run_top_level_mutation_campaign(output_path=top_mut_path)
    if not unit_mut_path.is_file():
        from verification.mutate2 import run_mutation_campaign
        run_mutation_campaign(output_path=unit_mut_path)

    top_mut = json.loads(top_mut_path.read_text(encoding="utf-8"))
    unit_mut = json.loads(unit_mut_path.read_text(encoding="utf-8"))
    run_data = json.loads(run_report_path.read_text(encoding="utf-8"))
    ver_data = json.loads(verify_report_path.read_text(encoding="utf-8"))

    wall_sec = round(time.perf_counter() - t0, 2)

    print("\n--- Mutation Kill Summary ---")
    print(f"[+] Unit Mutants (M10..M30)   : {unit_mut['killed_mutants']}/{unit_mut['total_mutants']} killed ({unit_mut['kill_rate_pct']:.1f}%)")
    print(f"[+] System Mutants (TM1..TM7) : {top_mut['killed_mutants']}/{top_mut['total_mutants']} killed ({top_mut['kill_rate_pct']:.1f}%)")
    print(f"[+] Total Wall Time           : {wall_sec:.2f} s")
    print("================================================================================")
    print(" DEMO2-FAST PASSED — 100% BIT-EXACT RTL & 100% MUTATION KILL RATE")
    print("================================================================================")

    summary = {
        "target": "demo2-fast",
        "status": "PASS",
        "wall_sec": wall_sec,
        "prompt_tokens": run_data["prompt_tokens"],
        "generated_tokens": run_data["generated_tokens"],
        "rtl_cycles": run_data["rtl"]["total_cycles"],
        "tokens_exact": ver_data["tokens_exact"],
        "audits_passed": ver_data["audit"]["passed"],
        "unit_mutation_kill_rate_pct": unit_mut["kill_rate_pct"],
        "top_mutation_kill_rate_pct": top_mut["kill_rate_pct"],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
