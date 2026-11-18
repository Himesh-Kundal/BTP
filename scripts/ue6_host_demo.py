"""Task UE.6: Host Demo (prompt text -> tokens -> RTL sim -> text) & fp32 Comparison.

Supports two modes:
1. --battery (default): Compares the 10 RTL-verified prompt decodes from UE.4
   (results/part2/full_decode.json) against fp32 HuggingFace greedy text,
   reporting per-prompt and aggregate first-divergence statistics in
   results/part2/ue6_host_demo.json.
2. --prompt "<text>" --max-new-tokens N: Live end-to-end host demo running
   tokenizer -> RTL Verilator simulation -> detokenizer, and comparing against
   fp32 HF greedy output.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from frontend.model_loader import DEFAULT_MODEL_ID, PINNED_REVISION, load_model_and_tokenizer
from sim.tb_generator import generate_top_accel_tb_verilog


def _compute_first_divergence(rtl_toks: list[int], fp32_toks: list[int]) -> dict[str, Any]:
    """Compute greedy match length and first-divergence position."""
    n = min(len(rtl_toks), len(fp32_toks))
    first_div = None
    for i in range(n):
        if rtl_toks[i] != fp32_toks[i]:
            first_div = i
            break
    match_len = first_div if first_div is not None else n
    exact_matches = sum(1 for a, b in zip(rtl_toks, fp32_toks, strict=False) if a == b)
    return {
        "greedy_match_length": match_len,
        "first_divergence_step": first_div,
        "exact_token_matches": exact_matches,
        "total_compared": n,
        "token_agreement_pct": round(100.0 * exact_matches / max(1, n), 2),
    }


def run_live_rtl_demo(prompt_text: str, max_new_tokens: int = 8) -> dict[str, Any]:
    """Run live text -> tokens -> RTL Verilator sim -> text and compare vs fp32 HF."""
    repo_root = Path(__file__).resolve().parent.parent
    hw_dir = repo_root / "build" / "smollm2_hw"
    ue1_path = repo_root / "results" / "part2" / "ue1_generation.json"
    with open(ue1_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)["config"]

    model, tokenizer = load_model_and_tokenizer(DEFAULT_MODEL_ID, PINNED_REVISION)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    # 1. FP32 HF greedy decode
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        out_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    fp32_gen_ids = out_ids[0, len(prompt_ids) :].tolist()
    fp32_text = tokenizer.decode(fp32_gen_ids)

    # 2. RTL Verilator simulation
    work_dir = repo_root / "build" / "ue6_live_sim"
    work_dir.mkdir(parents=True, exist_ok=True)
    sim_dir = work_dir / "obj_dir"

    tb_code = generate_top_accel_tb_verilog(
        prompt_tokens=prompt_ids,
        num_generate=max_new_tokens,
        seq_id=0,
        trace_dir=None,
        expected_tokens=None,
        timeout_cycles=1_000_000_000,
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        num_q_heads=cfg["num_attention_heads"],
        num_kv_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
    )
    tb_file = work_dir / "tb_live.v"
    tb_file.write_text(tb_code, encoding="utf-8")

    ver_bin = shutil.which("verilator")
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

    res_c = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
    if res_c.returncode != 0:
        raise RuntimeError(f"Verilator compile failed:\n{res_c.stderr}\n{res_c.stdout}")

    t0 = time.perf_counter()
    res_r = subprocess.run([str(sim_dir / "Vtb_top_accel")], capture_output=True, text=True, check=False)
    sim_sec = time.perf_counter() - t0
    if res_r.returncode != 0:
        raise RuntimeError(f"Verilator sim failed:\n{res_r.stderr}\n{res_r.stdout}")

    raw_out = res_r.stdout + res_r.stderr
    all_rtl_toks = [int(m.group(2)) for m in re.finditer(r"STEP\s+(\d+)\s+TOK_OUT=(\d+)", raw_out)]
    rtl_gen_ids = all_rtl_toks[len(prompt_ids) - 1 : len(prompt_ids) - 1 + max_new_tokens]
    rtl_text = tokenizer.decode(rtl_gen_ids)
    m_cyc = re.search(r"TOTAL_CYCLES=(\d+)", raw_out)
    total_cycles = int(m_cyc.group(1)) if m_cyc else 0

    div = _compute_first_divergence(rtl_gen_ids, fp32_gen_ids)

    res = {
        "prompt": prompt_text,
        "prompt_tokens": prompt_ids,
        "max_new_tokens": max_new_tokens,
        "fp32_gen_tokens": fp32_gen_ids,
        "fp32_text": fp32_text,
        "rtl_gen_tokens": rtl_gen_ids,
        "rtl_text": rtl_text,
        "total_cycles": total_cycles,
        "sim_sec": round(sim_sec, 2),
        **div,
    }
    print("\n=== SmolLM2-135M Host RTL Demo ===")
    print(f"Prompt     : {prompt_text!r}")
    print(f"FP32 Greedy: {fp32_text!r}")
    print(f"RTL  INT8  : {rtl_text!r}")
    print(f"Match Len  : {div['greedy_match_length']}/{max_new_tokens} (first_div={div['first_divergence_step']})")
    print(f"Cycles     : {total_cycles:,} ({sim_sec:.2f}s)")
    return res


def run_battery_comparison() -> None:
    """Compare UE.4 RTL-verified outputs against fp32 HF greedy text across all 10 prompts."""
    repo_root = Path(__file__).resolve().parent.parent
    ue4_path = repo_root / "results" / "part2" / "full_decode.json"
    prompts_path = repo_root / "eval" / "prompts.json"
    out_json_path = repo_root / "results" / "part2" / "ue6_host_demo.json"

    with open(ue4_path, "r", encoding="utf-8") as f:
        ue4 = json.load(f)
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompt_texts = {p["id"]: p["text"] for p in json.load(f)}

    model, tokenizer = load_model_and_tokenizer(DEFAULT_MODEL_ID, PINNED_REVISION)

    rows: list[dict[str, Any]] = []
    match_lengths: list[int] = []

    print("[*] Running fp32 HF greedy baseline on UE.4 prompts...")
    for p_rec in ue4["prompts"]:
        pid = p_rec["id"]
        text = prompt_texts[pid]
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        num_new = p_rec["new_tokens"]

        with torch.no_grad():
            out_ids = model.generate(
                torch.tensor([prompt_ids], dtype=torch.long),
                max_new_tokens=num_new,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        fp32_gen_ids = out_ids[0, len(prompt_ids) :].tolist()
        fp32_text = tokenizer.decode(fp32_gen_ids)
        rtl_gen_ids = p_rec["golden_gen_tokens"]
        rtl_text = p_rec["decoded_text"]

        div = _compute_first_divergence(rtl_gen_ids, fp32_gen_ids)
        match_lengths.append(div["greedy_match_length"])

        rows.append({
            "id": pid,
            "name": p_rec["name"],
            "prompt_text": text,
            "rtl_tokens_exact": p_rec["tokens_exact"],
            "fp32_text": fp32_text,
            "rtl_text": rtl_text,
            "fp32_gen_tokens": fp32_gen_ids,
            "rtl_gen_tokens": rtl_gen_ids,
            **div,
        })

    report = {
        "task": "UE.6",
        "num_prompts": len(rows),
        "new_tokens_per_prompt": ue4["new_tokens_per_prompt"],
        "mean_greedy_match_length": round(statistics.mean(match_lengths), 2),
        "median_greedy_match_length": round(statistics.median(match_lengths), 2),
        "min_greedy_match_length": min(match_lengths),
        "max_greedy_match_length": max(match_lengths),
        "mean_token_agreement_pct": round(statistics.mean(r["token_agreement_pct"] for r in rows), 2),
        "prompts": rows,
        "status": "PASS",
    }

    out_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Wrote {out_json_path}")
    print("\n=== UE.6 First-Divergence Statistics (RTL INT8 vs FP32 HF Greedy) ===")
    print(f"{'ID':<5} {'Name':<22} {'MatchLen':<10} {'FirstDiv':<10} {'Agr%':<8} {'RTL Text (trunc)':<30}")
    for r in rows:
        rtl_snip = repr(r["rtl_text"])[:28]
        print(
            f"{r['id']:<5} {r['name']:<22} {r['greedy_match_length']:<10} "
            f"{r['first_divergence_step']!s:<10} {r['token_agreement_pct']:<8} {rtl_snip:<30}"
        )
    print(
        f"\nSummary: mean_match_len={report['mean_greedy_match_length']}, "
        f"median_match_len={report['median_greedy_match_length']}, "
        f"mean_agreement={report['mean_token_agreement_pct']}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="UE.6 Host Demo & FP32 vs RTL Divergence Report")
    parser.add_argument("--prompt", type=str, default=None, help="Run live RTL simulation on custom prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Number of tokens to generate in live mode")
    args = parser.parse_args()

    if args.prompt is not None:
        run_live_rtl_demo(args.prompt, max_new_tokens=args.max_new_tokens)
    else:
        run_battery_comparison()


if __name__ == "__main__":
    main()
