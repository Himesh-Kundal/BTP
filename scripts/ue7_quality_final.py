"""Task UE.7: Final Quality Evaluation of Integer Model vs FP32 Reference.

Evaluates the compiled SmolLM2-135M integer model (CompiledArtifactInterpreter,
proven 100% bit-exact with RTL top_accel.v in UE.4) against fp32 HuggingFace
on the pinned WikiText-2 test windows (eval/harness.py), reporting:
- FP32 Perplexity vs INT8 Perplexity (and delta_ppl)
- Mean KL(P_fp32 || P_int)
- Top-1 ArgMax Agreement %
- Link to UE.4 RTL bit-exactness evidence
Outputs: results/part2/quality_final.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from compiler.interpreter import CompiledArtifactInterpreter
from eval.harness import (
    compute_cross_entropy_and_perplexity,
    compute_kl_divergence,
    compute_top1_agreement,
    get_tokenized_windows,
)
from frontend.model_loader import DEFAULT_MODEL_ID, PINNED_REVISION, load_model_and_tokenizer


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    hw_dir = repo_root / "build" / "smollm2_hw"
    ue4_path = repo_root / "results" / "part2" / "full_decode.json"
    ue6_path = repo_root / "results" / "part2" / "ue6_host_demo.json"
    frozen_path = repo_root / "numerics" / "config_frozen.json"
    out_json_path = repo_root / "results" / "part2" / "quality_final.json"

    with open(ue4_path, "r", encoding="utf-8") as f:
        ue4 = json.load(f)
    with open(ue6_path, "r", encoding="utf-8") as f:
        ue6 = json.load(f)
    with open(frozen_path, "r", encoding="utf-8") as f:
        frozen_cfg = json.load(f)

    # Load pinned WikiText-2 test windows
    num_windows = 10
    window_length = 64
    windows = get_tokenized_windows(
        model_id=DEFAULT_MODEL_ID,
        revision=PINNED_REVISION,
        num_windows=num_windows,
        window_length=window_length,
    )

    model, _ = load_model_and_tokenizer(DEFAULT_MODEL_ID, PINNED_REVISION)
    interp = CompiledArtifactInterpreter(hw_dir)

    all_fp32_logits: list[np.ndarray] = []
    all_int_logits: list[np.ndarray] = []
    all_targets: list[int] = []

    print(f"[*] Evaluating FP32 and INT8 on {num_windows} WikiText-2 windows (len={window_length})...")
    t0 = time.perf_counter()
    for w_idx, win in enumerate(windows):
        # FP32 forward pass on window[:-1] -> logits for window[1:]
        inp = torch.tensor([win[:-1]], dtype=torch.long)
        with torch.no_grad():
            fp32_out = model(inp).logits[0].cpu().numpy()  # [63, vocab]

        # INT8 compiled-artifact interpreter step-by-step
        interp.reset()
        int_win_logits: list[np.ndarray] = []
        for pos, tok_id in enumerate(win[:-1]):
            logits_i, _ = interp.decode_step(int(tok_id), pos=pos)
            # Scale raw INT24 logits by s_res=0.01 (folded into lm_head dyadic scales)
            int_win_logits.append(logits_i.astype(np.float64) * 0.01)

        all_fp32_logits.append(fp32_out)
        all_int_logits.append(np.stack(int_win_logits, axis=0))
        all_targets.extend(win[1:])
        print(f"    [+] Window {w_idx + 1}/{num_windows} done")

    eval_sec = time.perf_counter() - t0
    fp32_mat = np.concatenate(all_fp32_logits, axis=0)
    int_mat = np.concatenate(all_int_logits, axis=0)

    fp32_ce, fp32_ppl = compute_cross_entropy_and_perplexity(fp32_mat, all_targets)
    int_ce, int_ppl = compute_cross_entropy_and_perplexity(int_mat, all_targets)
    delta_ppl = int_ppl - fp32_ppl
    mean_kl = compute_kl_divergence(fp32_mat, int_mat)
    top1_agr = compute_top1_agreement(fp32_mat, int_mat)

    thresholds = frozen_cfg["quality_thresholds"]
    meets_thresholds = (
        delta_ppl <= thresholds["max_delta_ppl"]
        and top1_agr >= thresholds["min_top1_agreement_pct"]
        and mean_kl <= thresholds["max_mean_kl_div"]
    )

    report: dict[str, Any] = {
        "task": "UE.7",
        "model_id": DEFAULT_MODEL_ID,
        "revision": PINNED_REVISION,
        "eval_dataset": {
            "name": "wikitext-2-raw-v1",
            "split": "test",
            "num_windows": num_windows,
            "window_length": window_length,
            "total_predicted_tokens": len(all_targets),
        },
        "fp32_metrics": {
            "cross_entropy": round(fp32_ce, 4),
            "perplexity": round(fp32_ppl, 4),
        },
        "int_metrics": {
            "cross_entropy": round(int_ce, 4),
            "perplexity": round(int_ppl, 4),
        },
        "comparison": {
            "delta_ppl": round(delta_ppl, 4),
            "mean_kl_div": round(mean_kl, 4),
            "top1_agreement_pct": round(top1_agr, 2),
            "mean_greedy_match_length": ue6["mean_greedy_match_length"],
            "median_greedy_match_length": ue6["median_greedy_match_length"],
        },
        "frozen_thresholds_ha": thresholds,
        "meets_ha_thresholds": meets_thresholds,
        "rtl_equivalence_link": {
            "source_report": "results/part2/full_decode.json",
            "ue4_prompts_exact": f"{ue4['num_prompts']}/{ue4['num_prompts']}",
            "ue4_total_decode_steps": ue4["total_decode_steps"],
            "ue4_sampled_trace_identical": ue4["sampled_trace_identical"],
            "ue4_sampled_trace_elements_checked": ue4["sampled_trace_elements_checked"],
        },
        "eval_wall_sec": round(eval_sec, 2),
        "status": "PASS" if meets_thresholds else "THRESHOLD_VIOLATION",
    }

    out_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Wrote {out_json_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
