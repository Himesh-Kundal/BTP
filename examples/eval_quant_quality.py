"""Evaluate quantization numerical quality (Claim C2) across 21 layer configurations."""

import json
import sys
from pathlib import Path

import numpy as np

from compiler.quality import evaluate_layer_c2

PROJECTION_TYPES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
EVAL_LAYERS = [0, 14, 29]
MODEL_DIR = Path("artifacts") / "SmolLM2-135M@93efa2f097d5"
OUTPUT_PATH = Path("results") / "quant_quality.json"


def run_c2_evaluation() -> dict[str, dict[str, float]]:
    """Run Claim C2 evaluation across 21 configurations and save results/quant_quality.json."""
    if not MODEL_DIR.exists():
        print(f"Error: Artifact directory {MODEL_DIR} not found.")
        sys.exit(1)

    print("=" * 80)
    print(" LLMHW COMPILER — CLAIM C2 QUANTIZATION QUALITY EVALUATION")
    print(" (Integer Reference vs Float Reference)")
    print("=" * 80)
    print(f"{'Layer / Proj':<22} | {'MaxAbsErr':<10} | {'Rel-L2':<10} | {'Cos-Sim':<10} | {'SQNR (dB)':<10}")
    print("-" * 80)

    results: dict[str, dict[str, float]] = {}

    for layer_idx in EVAL_LAYERS:
        for proj in PROJECTION_TYPES:
            cfg_name = f"layer_{layer_idx}_{proj}"
            layer_dir = MODEL_DIR / f"layer_{layer_idx}" / proj
            w_path = layer_dir / "weight.npy"
            x_path = layer_dir / "calib" / "x_sample.npy"

            if not w_path.exists() or not x_path.exists():
                print(f"Warning: Missing files for {cfg_name}, skipping.")
                continue

            w_float = np.load(w_path)
            x_float = np.load(x_path)

            metrics = evaluate_layer_c2(w_float, x_float, calib_method="absmax")
            results[cfg_name] = metrics

            print(
                f"{cfg_name:<22} | "
                f"{metrics['max_abs_error']:<10.4f} | "
                f"{metrics['relative_l2']:<10.4e} | "
                f"{metrics['cosine_similarity']:<10.6f} | "
                f"{metrics['sqnr_db']:<10.2f}"
            )

    print("=" * 80)

    # Compute summary statistics across all evaluated configurations
    all_sqnr = [m["sqnr_db"] for m in results.values()]
    all_cos = [m["cosine_similarity"] for m in results.values()]
    all_rel_l2 = [m["relative_l2"] for m in results.values()]
    all_max_abs = [m["max_abs_error"] for m in results.values()]

    summary = {
        "num_configurations": len(results),
        "sqnr_db_min": float(np.min(all_sqnr)),
        "sqnr_db_mean": float(np.mean(all_sqnr)),
        "sqnr_db_max": float(np.max(all_sqnr)),
        "cosine_sim_min": float(np.min(all_cos)),
        "cosine_sim_mean": float(np.mean(all_cos)),
        "rel_l2_max": float(np.max(all_rel_l2)),
        "max_abs_err_max": float(np.max(all_max_abs)),
    }

    output_data = {
        "summary": summary,
        "configurations": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"[*] Summary: min SQNR = {summary['sqnr_db_min']:.2f} dB, "
          f"min Cosine Sim = {summary['cosine_sim_min']:.6f}, "
          f"max Rel-L2 = {summary['rel_l2_max']:.4e}")
    print(f"[*] Saved quality report to {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    run_c2_evaluation()
