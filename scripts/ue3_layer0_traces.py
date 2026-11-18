"""Task UE.3: Layer-0 trace equality on full SmolLM2-135M for first and later positions.

Runs both CompiledArtifactInterpreter and Verilator top_accel on a 12-position
sequence (8 prompt tokens + 4 generated tokens, pos=0..11), verifying bit-exact
Layer-0 trace equality and token equality at every position.
Outputs: results/part2/ue3_layer0_traces.json
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from compiler.interpreter import CompiledArtifactInterpreter
from sim.tb_generator import dump_interpreter_traces, generate_top_accel_tb_verilog
from verification.trace_diff import compare_trace_directories


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    hw_dir = repo_root / "build" / "smollm2_hw"
    ue1_path = repo_root / "results" / "part2" / "ue1_generation.json"
    out_json_path = repo_root / "results" / "part2" / "ue3_layer0_traces.json"

    with open(ue1_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)["config"]

    prompt = [1, 42, 128, 512, 1024, 2048, 4096, 8192]
    num_gen = 4
    total_steps = len(prompt) + num_gen

    golden_dir = repo_root / "build" / "ue3_golden_traces"
    rtl_dir = repo_root / "build" / "ue3_rtl_traces"
    shutil.rmtree(golden_dir, ignore_errors=True)
    shutil.rmtree(rtl_dir, ignore_errors=True)
    for s in range(total_steps):
        (rtl_dir / "seq0" / f"tok{s}" / "L0").mkdir(parents=True, exist_ok=True)

    # 1. Run Python CompiledArtifactInterpreter and dump Layer-0 traces for all positions
    print(f"[*] Running CompiledArtifactInterpreter for {total_steps} positions (pos=0..{total_steps-1})...")
    t0 = time.perf_counter()
    interp = CompiledArtifactInterpreter(hw_dir)
    interp.reset()

    golden_tokens: list[int] = []
    for pos in range(total_steps):
        tok_in = prompt[pos] if pos < len(prompt) else golden_tokens[-1]
        _, tok_out = interp.decode_step(tok_in, pos=pos)
        golden_tokens.append(tok_out)
        # Dump only L0 (plus final_norm & lm_head in L0)
        dump_interpreter_traces(interp, golden_dir, seq_id=0, tok_idx=pos, num_layers=1)

    interp_sec = time.perf_counter() - t0
    print(f"[+] Interpreter finished in {interp_sec:.2f}s. Tokens: {golden_tokens}")

    # 2. Generate testbench and compile with Verilator -O3
    ver_bin = shutil.which("verilator")
    if not ver_bin:
        raise RuntimeError("verilator not found")

    sim_dir = hw_dir / "verilator_ue3"
    sim_dir.mkdir(parents=True, exist_ok=True)
    tb_code = generate_top_accel_tb_verilog(
        prompt_tokens=prompt,
        num_generate=num_gen,
        seq_id=0,
        trace_dir=rtl_dir,
        expected_tokens=golden_tokens,
        timeout_cycles=250_000_000,
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        num_q_heads=cfg["num_attention_heads"],
        num_kv_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
    )
    tb_file = hw_dir / "tb_ue3.v"
    tb_file.write_text(tb_code, encoding="utf-8")

    rtl_lib = repo_root / "rtl" / "lib"
    all_rtl = sorted(rtl_lib.glob("*.v"))
    sources = [p for p in all_rtl if p.name != "top_accel.v"] + [hw_dir / "top_accel.v", tb_file]

    compile_cmd = [
        ver_bin,
        "--binary",
        "--timing",
        "-j",
        "0",
        "-CFLAGS",
        "-O3",
        "-Wall",
        "-Wno-BLKSEQ",
        "-Wno-WIDTHTRUNC",
        "-Wno-WIDTHEXPAND",
        "-Wno-UNUSEDSIGNAL",
        "-Wno-PINMISSING",
        "-Wno-UNUSEDPARAM",
        "-Wno-INITIALDLY",
        "-Wno-DECLFILENAME",
        "-Wno-CASEINCOMPLETE",
        "-Wno-SYNCASYNCNET",
        "-Wno-UNDRIVEN",
        f"-I{rtl_lib}",
        "--top-module",
        "tb_top_accel",
        "-Mdir",
        str(sim_dir),
    ] + [str(s) for s in sources]

    print("[*] Compiling UE.3 testbench with Verilator (-O3)...")
    t_b0 = time.perf_counter()
    res_c = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
    build_sec = time.perf_counter() - t_b0
    if res_c.returncode != 0:
        raise RuntimeError(f"Verilator build failed:\n{res_c.stderr}\n{res_c.stdout}")

    print(f"[*] Running {total_steps}-position RTL simulation...")
    t_s0 = time.perf_counter()
    res_r = subprocess.run([str(sim_dir / "Vtb_top_accel")], capture_output=True, text=True, check=False)
    sim_sec = time.perf_counter() - t_s0
    if res_r.returncode != 0:
        raise RuntimeError(f"Verilator run failed:\n{res_r.stderr}\n{res_r.stdout}")

    raw_out = res_r.stdout + res_r.stderr
    rtl_tokens = [int(m.group(2)) for m in re.finditer(r"STEP\s+(\d+)\s+TOK_OUT=(\d+)", raw_out)]
    m_cyc = re.search(r"TOTAL_CYCLES=(\d+)", raw_out)
    total_cycles = int(m_cyc.group(1)) if m_cyc else 0

    # 3. Compare Layer-0 traces across all 12 positions
    diff_res = compare_trace_directories(golden_dir, rtl_dir)

    report: dict[str, Any] = {
        "task": "UE.3",
        "prompt_tokens": prompt,
        "num_generate": num_gen,
        "total_positions": total_steps,
        "golden_tokens": golden_tokens,
        "rtl_tokens": rtl_tokens,
        "tokens_exact": golden_tokens == rtl_tokens,
        "trace_identical": diff_res.is_identical,
        "first_divergence": str(diff_res.first_divergence) if diff_res.first_divergence else None,
        "files_checked": diff_res.total_files_checked,
        "elements_checked": diff_res.total_elements_checked,
        "total_cycles": total_cycles,
        "build_sec": round(build_sec, 2),
        "sim_sec": round(sim_sec, 2),
        "sim_speed_kcycles_sec": round((total_cycles / sim_sec) / 1000.0, 2) if sim_sec > 0 else 0.0,
        "status": "PASS" if (golden_tokens == rtl_tokens and diff_res.is_identical) else "FAIL",
    }

    out_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Wrote {out_json_path}")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
