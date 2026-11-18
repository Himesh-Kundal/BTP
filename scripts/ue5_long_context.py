"""Task UE.5: Long-context verification up to MAX_CTX=256 on full SmolLM2-135M.

Runs a 256-token sequence (250 prompt tokens + 6 generated tokens, pos=0..255)
through both CompiledArtifactInterpreter and Verilator top_accel (-O3),
verifying 100% token exactness across all 256 positions and Layer-0 trace
equality at key positions including pos=254 and pos=255 (MAX_CTX-1).
Outputs: results/part2/ue5_long_context.json
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from compiler.interpreter import CompiledArtifactInterpreter
from eval.harness import get_tokenized_windows
from frontend.model_loader import DEFAULT_MODEL_ID, PINNED_REVISION
from sim.tb_generator import dump_interpreter_traces, generate_top_accel_tb_verilog
from verification.trace_diff import compare_trace_directories


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    hw_dir = repo_root / "build" / "smollm2_hw"
    ue1_path = repo_root / "results" / "part2" / "ue1_generation.json"
    out_json_path = repo_root / "results" / "part2" / "ue5_long_context.json"

    with open(ue1_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)["config"]

    max_ctx = 256
    prompt_len = 250
    num_gen = max_ctx - prompt_len  # 6 -> total 256 positions (0..255)

    # Build a deterministic 250-token prompt from WikiText-2 test windows
    windows = get_tokenized_windows(num_windows=5, window_length=64)
    long_prompt = (windows[0] + windows[1] + windows[2] + windows[3])[:prompt_len]
    assert len(long_prompt) == prompt_len

    golden_dir = repo_root / "build" / "ue5_golden_traces"
    rtl_dir = repo_root / "build" / "ue5_rtl_traces"
    shutil.rmtree(golden_dir, ignore_errors=True)
    shutil.rmtree(rtl_dir, ignore_errors=True)

    sample_positions = {0, 1, 127, 254, 255}
    for s in range(max_ctx):
        for l_idx in range(cfg["num_hidden_layers"]):
            (rtl_dir / "seq0" / f"tok{s}" / f"L{l_idx}").mkdir(parents=True, exist_ok=True)

    print(f"[*] Running CompiledArtifactInterpreter up to MAX_CTX={max_ctx}...")
    t0 = time.perf_counter()
    interp = CompiledArtifactInterpreter(hw_dir)
    interp.reset()
    golden_tokens: list[int] = []
    for pos in range(max_ctx):
        tok_in = long_prompt[pos] if pos < prompt_len else golden_tokens[-1]
        _, tok_out = interp.decode_step(tok_in, pos=pos)
        golden_tokens.append(tok_out)
        if pos in sample_positions:
            dump_interpreter_traces(interp, golden_dir, seq_id=0, tok_idx=pos, num_layers=1)

    interp_sec = time.perf_counter() - t0
    print(f"[+] Interpreter finished 256 positions in {interp_sec:.2f}s")

    ver_bin = shutil.which("verilator")
    if not ver_bin:
        raise RuntimeError("verilator not found")

    work_dir = repo_root / "build" / "ue5_sim"
    work_dir.mkdir(parents=True, exist_ok=True)
    sim_dir = work_dir / "obj_dir"

    # Generate testbench with trace dumping; we'll prune non-sampled positions/layers after run
    # To keep disk I/O fast during 256-step run, filter trace_done in tb to sample_positions and L0
    tb_code = generate_top_accel_tb_verilog(
        prompt_tokens=long_prompt,
        num_generate=num_gen,
        seq_id=0,
        trace_dir=rtl_dir,
        expected_tokens=golden_tokens,
        timeout_cycles=10_000_000_000,
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        num_q_heads=cfg["num_attention_heads"],
        num_kv_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
    )
    # Optimize trace dump condition in tb to only dump L0 at sample_positions
    tb_code = tb_code.replace(
        "if (sample_done) begin",
        "if (sample_done && sample_layer == 0 && (step_idx == 0 || step_idx == 1 || step_idx == 127 || step_idx == 254 || step_idx == 255)) begin",
    )

    tb_file = work_dir / "tb_ue5.v"
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

    print("[*] Compiling UE.5 long-context testbench with Verilator (-O3)...")
    res_c = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
    if res_c.returncode != 0:
        raise RuntimeError(f"Verilator compile failed:\n{res_c.stderr}\n{res_c.stdout}")

    print("[*] Running 256-position long-context RTL simulation...")
    t_s0 = time.perf_counter()
    res_r = subprocess.run([str(sim_dir / "Vtb_top_accel")], capture_output=True, text=True, check=False)
    sim_sec = time.perf_counter() - t_s0
    if res_r.returncode != 0:
        raise RuntimeError(f"Verilator sim failed:\n{res_r.stderr}\n{res_r.stdout}")

    raw_out = res_r.stdout + res_r.stderr
    rtl_tokens = [int(m.group(2)) for m in re.finditer(r"STEP\s+(\d+)\s+TOK_OUT=(\d+)", raw_out)]
    m_cyc = re.search(r"TOTAL_CYCLES=(\d+)", raw_out)
    total_cycles = int(m_cyc.group(1)) if m_cyc else 0

    # Remove empty non-sampled dirs from rtl_dir before compare_trace_directories
    for tok_dir in (rtl_dir / "seq0").glob("tok*"):
        pos_idx = int(tok_dir.name.replace("tok", ""))
        if pos_idx not in sample_positions:
            shutil.rmtree(tok_dir, ignore_errors=True)
        else:
            for l_dir in tok_dir.glob("L*"):
                if l_dir.name != "L0":
                    shutil.rmtree(l_dir, ignore_errors=True)

    diff_res = compare_trace_directories(golden_dir, rtl_dir)

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID, revision=PINNED_REVISION)
    gen_tokens = rtl_tokens[prompt_len - 1 :]

    report: dict[str, Any] = {
        "task": "UE.5",
        "max_ctx": max_ctx,
        "prompt_len": prompt_len,
        "num_generate": num_gen,
        "total_positions": len(rtl_tokens),
        "tokens_exact": rtl_tokens == golden_tokens,
        "sample_positions_traced": sorted(sample_positions),
        "trace_identical": diff_res.is_identical,
        "files_checked": diff_res.total_files_checked,
        "elements_checked": diff_res.total_elements_checked,
        "first_divergence": str(diff_res.first_divergence) if diff_res.first_divergence else None,
        "total_cycles": total_cycles,
        "cycles_per_token": total_cycles // max(1, len(rtl_tokens)),
        "sim_sec": round(sim_sec, 2),
        "sim_speed_kcycles_sec": round((total_cycles / sim_sec) / 1000.0, 2) if sim_sec > 0 else 0.0,
        "generated_tokens": gen_tokens,
        "generated_text": tokenizer.decode(gen_tokens),
        "status": "PASS" if (rtl_tokens == golden_tokens and diff_res.is_identical) else "FAIL",
    }

    out_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Wrote {out_json_path}")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
