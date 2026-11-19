"""Task UG.2: Full SmolLM2-135M Short Generation Demo (`make demo2`).

Executes a short autoregressive generation on the complete 30-layer SmolLM2-135M
hardware accelerator (`build/smollm2_hw`) in Verilator RTL simulation:
1. If `build/smollm2_hw/top_accel.v` is not present (e.g. fresh clone), invokes
   `scripts/ue1_generate_full.py` to extract, calibrate, and compile the bundle.
2. Runs prompt `"The capital of France is Paris. The capital of Germany is"`
   for 4 new tokens through:
   - HuggingFace PyTorch FP32 greedy baseline
   - Python `CompiledArtifactInterpreter` (integer reference oracle)
   - Verilator RTL hardware simulation (`top_accel.v`)
3. Verifies 100% bit-exact token equality between Verilator RTL and the integer
   reference oracle, and reports agreement with FP32 greedy text.
Outputs: `results/part2/demo2.json`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from compiler.interpreter import CompiledArtifactInterpreter
from scripts.ue6_host_demo import run_live_rtl_demo


def main() -> None:
    t0 = time.perf_counter()
    repo_root = Path(__file__).resolve().parent.parent
    hw_dir = repo_root / "build" / "smollm2_hw"
    out_path = repo_root / "results" / "part2" / "demo2.json"

    print("================================================================================")
    print(" MAKE DEMO2 — FULL SMOLLM2-135M 30-LAYER VERILATOR RTL GENERATION DEMO")
    print("================================================================================")

    if not (hw_dir / "top_accel.v").is_file():
        print("[*] Hardware bundle build/smollm2_hw not found; generating via scripts/ue1_generate_full.py...")
        from scripts.ue1_generate_full import main as ue1_main
        ue1_main()

    prompt_text = "The capital of France is Paris. The capital of Germany is"
    max_new_tokens = 4

    # 1. Run live RTL simulation + FP32 greedy comparison
    live_res = run_live_rtl_demo(prompt_text=prompt_text, max_new_tokens=max_new_tokens)

    # 2. Verify CompiledArtifactInterpreter matches RTL bit-for-bit
    interp = CompiledArtifactInterpreter(hw_dir)
    interp.reset()
    prompt_ids = live_res["prompt_tokens"]
    interp_all: list[int] = []
    for pos in range(len(prompt_ids) + max_new_tokens):
        tok_in = prompt_ids[pos] if pos < len(prompt_ids) else interp_all[-1]
        _, tok_out = interp.decode_step(tok_in, pos=pos)
        interp_all.append(tok_out)
    interp_gen = interp_all[len(prompt_ids) - 1 : len(prompt_ids) - 1 + max_new_tokens]

    rtl_matches_oracle = live_res["rtl_gen_tokens"] == interp_gen
    wall_sec = round(time.perf_counter() - t0, 2)

    print(f"Oracle INT8: {interp_gen} (RTL == Oracle: {rtl_matches_oracle})")
    print(f"Total Wall : {wall_sec:.2f} s")
    if not rtl_matches_oracle:
        print("[-] ERROR: RTL tokens diverged from integer oracle!")
        raise SystemExit(1)

    print("================================================================================")
    print(" DEMO2 PASSED — SMOLLM2-135M RTL == INTEGER ORACLE (100% BIT-EXACT)")
    print("================================================================================")

    summary = {
        "target": "demo2",
        "status": "PASS",
        "wall_sec": wall_sec,
        **live_res,
        "oracle_gen_tokens": interp_gen,
        "rtl_matches_integer_oracle": rtl_matches_oracle,
    }
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
