"""Task UE.4: Full 30-layer SmolLM2-135M Decode Verification.

Runs >= 10 real prompts (p01..p10 from eval/prompts.json) x 16 new tokens
through both CompiledArtifactInterpreter and Verilator top_accel (-O3),
verifying 100% token exactness, sampled-layer (L0, L14, L29) trace equality,
and measuring cycles/token.
Outputs: results/part2/full_decode.json
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from compiler.interpreter import CompiledArtifactInterpreter
from frontend.model_loader import DEFAULT_MODEL_ID, PINNED_REVISION
from sim.tb_generator import dump_interpreter_traces, generate_top_accel_tb_verilog
from verification.trace_diff import compare_trace_directories


def _run_prompt_verilator(task: dict[str, Any]) -> dict[str, Any]:
    """Compile and run a single prompt simulation in an isolated working directory."""
    idx = task["idx"]
    prompt_id = task["id"]
    prompt_tokens = task["prompt_tokens"]
    num_gen = task["num_gen"]
    golden_tokens = task["golden_tokens"]
    cfg = task["config"]
    hw_dir = Path(task["hw_dir"])
    rtl_lib = Path(task["rtl_lib"])
    work_dir = Path(task["work_dir"])
    trace_dir = Path(task["trace_dir"]) if task["trace_dir"] else None

    work_dir.mkdir(parents=True, exist_ok=True)
    if trace_dir is not None:
        shutil.rmtree(trace_dir, ignore_errors=True)
        total_steps = len(prompt_tokens) + num_gen
        for s in range(total_steps):
            for l_idx in range(cfg["num_hidden_layers"]):
                (trace_dir / f"seq{idx}" / f"tok{s}" / f"L{l_idx}").mkdir(parents=True, exist_ok=True)

    tb_code = generate_top_accel_tb_verilog(
        prompt_tokens=prompt_tokens,
        num_generate=num_gen,
        seq_id=idx,
        trace_dir=trace_dir,
        expected_tokens=golden_tokens,
        timeout_cycles=500_000_000,
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        num_q_heads=cfg["num_attention_heads"],
        num_kv_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
    )
    tb_file = work_dir / f"tb_{prompt_id}.v"
    tb_file.write_text(tb_code, encoding="utf-8")

    ver_bin = shutil.which("verilator")
    all_rtl = sorted(rtl_lib.glob("*.v"))
    sources = [p for p in all_rtl if p.name != "top_accel.v"] + [hw_dir / "top_accel.v", tb_file]

    sim_dir = work_dir / "obj_dir"
    compile_cmd = [
        ver_bin,
        "--binary",
        "--timing",
        "-j",
        "4",
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

    res_c = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
    if res_c.returncode != 0:
        raise RuntimeError(f"Verilator compile failed for {prompt_id}:\n{res_c.stderr}\n{res_c.stdout}")

    t_s0 = time.perf_counter()
    res_r = subprocess.run([str(sim_dir / "Vtb_top_accel")], capture_output=True, text=True, check=False)
    sim_sec = time.perf_counter() - t_s0
    if res_r.returncode != 0:
        raise RuntimeError(f"Verilator sim failed for {prompt_id}:\n{res_r.stderr}\n{res_r.stdout}")

    raw_out = res_r.stdout + res_r.stderr
    rtl_tokens = [int(m.group(2)) for m in re.finditer(r"STEP\s+(\d+)\s+TOK_OUT=(\d+)", raw_out)]
    m_cyc = re.search(r"TOTAL_CYCLES=(\d+)", raw_out)
    total_cycles = int(m_cyc.group(1)) if m_cyc else 0

    return {
        "id": prompt_id,
        "rtl_tokens": rtl_tokens,
        "total_cycles": total_cycles,
        "sim_sec": round(sim_sec, 2),
        "tokens_exact": rtl_tokens == golden_tokens,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    hw_dir = repo_root / "build" / "smollm2_hw"
    ue1_path = repo_root / "results" / "part2" / "ue1_generation.json"
    prompts_path = repo_root / "eval" / "prompts.json"
    out_json_path = repo_root / "results" / "part2" / "full_decode.json"

    with open(ue1_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)["config"]
    with open(prompts_path, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)[:10]  # p01..p10

    num_gen = 16
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID, revision=PINNED_REVISION)

    print("[*] Initializing CompiledArtifactInterpreter for full SmolLM2-135M...")
    interp = CompiledArtifactInterpreter(hw_dir)

    golden_trace_dir = repo_root / "build" / "ue4_golden_traces"
    rtl_trace_dir = repo_root / "build" / "ue4_rtl_traces"
    shutil.rmtree(golden_trace_dir, ignore_errors=True)
    shutil.rmtree(rtl_trace_dir, ignore_errors=True)

    tasks: list[dict[str, Any]] = []
    prompt_meta: list[dict[str, Any]] = []

    print(f"[*] Running interpreter on {len(all_prompts)} prompts x {num_gen} new tokens...")
    t_interp0 = time.perf_counter()
    for idx, p_item in enumerate(all_prompts):
        p_ids = tokenizer.encode(p_item["text"], add_special_tokens=False)
        interp.reset()
        golden_tokens: list[int] = []
        total_steps = len(p_ids) + num_gen
        for pos in range(total_steps):
            tok_in = p_ids[pos] if pos < len(p_ids) else golden_tokens[-1]
            _, tok_out = interp.decode_step(tok_in, pos=pos)
            golden_tokens.append(tok_out)
            if idx == 0:
                dump_interpreter_traces(interp, golden_trace_dir, seq_id=0, tok_idx=pos, num_layers=cfg["num_hidden_layers"])

        gen_only_tokens = golden_tokens[len(p_ids) - 1 : len(p_ids) - 1 + num_gen]
        prompt_meta.append({
            "idx": idx,
            "id": p_item["id"],
            "name": p_item["name"],
            "text": p_item["text"],
            "prompt_tokens": p_ids,
            "golden_all_tokens": golden_tokens,
            "golden_gen_tokens": gen_only_tokens,
            "decoded_text": tokenizer.decode(gen_only_tokens),
        })
        tasks.append({
            "idx": idx,
            "id": p_item["id"],
            "prompt_tokens": p_ids,
            "num_gen": num_gen,
            "golden_tokens": golden_tokens,
            "config": cfg,
            "hw_dir": str(hw_dir),
            "rtl_lib": str(repo_root / "rtl" / "lib"),
            "work_dir": str(repo_root / "build" / "ue4_sim" / p_item["id"]),
            "trace_dir": str(rtl_trace_dir) if idx == 0 else None,
        })
    interp_sec = time.perf_counter() - t_interp0
    print(f"[+] Interpreter finished all 10 prompts in {interp_sec:.2f}s")

    print("[*] Running 10 Verilator simulations in parallel (4 workers)...")
    t_sim0 = time.perf_counter()
    sim_results_by_id: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for res in pool.map(_run_prompt_verilator, tasks):
            sim_results_by_id[res["id"]] = res
            print(f"    [+] {res['id']}: exact={res['tokens_exact']}, cycles={res['total_cycles']}, sim_sec={res['sim_sec']}s")
    total_wall_sec = time.perf_counter() - t_sim0

    # Sampled-layer (L0, L14, L29) trace check on p01
    # Remove non-sampled layers from both trees to compare sampled layers cleanly, or compare directly:
    sampled_layers = {"L0", "L14", "L29"}
    for base in (golden_trace_dir, rtl_trace_dir):
        for tok_dir in (base / "seq0").glob("tok*"):
            for l_dir in tok_dir.glob("L*"):
                if l_dir.name not in sampled_layers:
                    shutil.rmtree(l_dir, ignore_errors=True)

    diff_res = compare_trace_directories(golden_trace_dir, rtl_trace_dir)

    total_steps_all = 0
    total_cycles_all = 0
    all_exact = True
    prompt_records: list[dict[str, Any]] = []

    for meta in prompt_meta:
        sr = sim_results_by_id[meta["id"]]
        n_steps = len(meta["prompt_tokens"]) + num_gen
        total_steps_all += n_steps
        total_cycles_all += sr["total_cycles"]
        if not sr["tokens_exact"]:
            all_exact = False
        prompt_records.append({
            "id": meta["id"],
            "name": meta["name"],
            "prompt_len": len(meta["prompt_tokens"]),
            "new_tokens": num_gen,
            "total_steps": n_steps,
            "tokens_exact": sr["tokens_exact"],
            "total_cycles": sr["total_cycles"],
            "cycles_per_token": sr["total_cycles"] // n_steps,
            "sim_sec": sr["sim_sec"],
            "golden_gen_tokens": meta["golden_gen_tokens"],
            "decoded_text": meta["decoded_text"],
        })

    report: dict[str, Any] = {
        "task": "UE.4",
        "num_prompts": len(prompt_records),
        "new_tokens_per_prompt": num_gen,
        "total_decode_steps": total_steps_all,
        "all_prompts_token_exact": all_exact,
        "sampled_layers_checked": sorted(sampled_layers),
        "sampled_trace_identical": diff_res.is_identical,
        "sampled_trace_files_checked": diff_res.total_files_checked,
        "sampled_trace_elements_checked": diff_res.total_elements_checked,
        "first_divergence": str(diff_res.first_divergence) if diff_res.first_divergence else None,
        "total_cycles": total_cycles_all,
        "mean_cycles_per_token": total_cycles_all // max(1, total_steps_all),
        "parallel_sim_wall_sec": round(total_wall_sec, 2),
        "prompts": prompt_records,
        "status": "PASS" if (all_exact and diff_res.is_identical) else "FAIL",
    }

    out_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Wrote {out_json_path}")
    print(f"[+] Status: {report['status']}, Mean cycles/token: {report['mean_cycles_per_token']}, Wall: {total_wall_sec:.2f}s")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
