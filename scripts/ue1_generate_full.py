"""Task UE.1: Generate full SmolLM2-135M ROM images, program, and top_accel.v.

Measures generation wall time, peak RAM, and exact file sizes on disk,
verifying compliance with U0.4 feasibility budgets (results/part2/budgets.json).
Outputs: results/part2/ue1_generation.json
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path
from typing import Any

from backend.top_generator import emit_hardware_bundle
from eval.harness import get_tokenized_windows
from frontend.importer_v2 import (
    build_graph_v2_from_artifacts,
    calibrate_and_migrate_artifacts_v2,
    extract_and_dump_model_v2,
    load_model_artifacts_v2,
)
from frontend.model_loader import DEFAULT_MODEL_ID, PINNED_REVISION, load_model_and_tokenizer


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    artifacts_dir = repo_root / "build" / "smollm2_artifacts_v2"
    hw_out_dir = repo_root / "build" / "smollm2_hw"
    frozen_cfg_path = repo_root / "numerics" / "config_frozen.json"
    budgets_path = repo_root / "results" / "part2" / "budgets.json"
    out_json_path = repo_root / "results" / "part2" / "ue1_generation.json"

    with open(frozen_cfg_path, "r", encoding="utf-8") as f:
        frozen_config = json.load(f)
    with open(budgets_path, "r", encoding="utf-8") as f:
        budgets = json.load(f)

    t0 = time.perf_counter()

    # Step 1: Extract and calibrate model artifacts if not already present with ADR-2-2 calibration
    calib_file = artifacts_dir / "calib_scales.json"
    needs_extract = not (artifacts_dir / "manifest.json").exists() or not calib_file.exists()
    if not needs_extract:
        with open(calib_file, "r", encoding="utf-8") as f:
            existing_calib = json.load(f)
        if "__s_gate__" not in existing_calib:
            needs_extract = True

    if needs_extract:
        print(f"[*] Loading {DEFAULT_MODEL_ID} ({PINNED_REVISION}) and running ADR-2-2 calibration...")
        model, _ = load_model_and_tokenizer(DEFAULT_MODEL_ID, PINNED_REVISION)
        extract_and_dump_model_v2(
            model=model,
            config=model.config,
            target_dir=artifacts_dir,
            model_id=DEFAULT_MODEL_ID,
            revision=PINNED_REVISION,
        )
        calib_windows = get_tokenized_windows(
            model_id=DEFAULT_MODEL_ID,
            revision=PINNED_REVISION,
            num_windows=4,
            window_length=64,
        )
        calibrate_and_migrate_artifacts_v2(
            model=model,
            calibration_windows=calib_windows,
            target_dir=artifacts_dir,
            s_gate=0.10,
            swig_shift=6,
            score_shift=1,
        )
        del model
    t_extract = time.perf_counter()

    # Step 2: Load artifacts & build GraphV2
    print("[*] Loading v2 artifacts and building GraphV2...")
    artifacts = load_model_artifacts_v2(artifacts_dir)
    graph = build_graph_v2_from_artifacts(artifacts_dir)
    t_graph = time.perf_counter()

    # Step 3: Emit hardware bundle (formats, ROM packing, schedule, 128b weights, top_accel.v)
    print(f"[*] Emitting full SmolLM2-135M hardware bundle to {hw_out_dir}...")
    bundle_res = emit_hardware_bundle(
        graph=graph,
        weights_dict=artifacts["tensors"],
        config=artifacts["config"],
        frozen_config=frozen_config,
        out_dir=hw_out_dir,
        tile_r=8,
        tile_c=16,
        max_ctx=256,
        calib_scales=artifacts.get("calib_scales"),
    )
    t_emit = time.perf_counter()

    # Step 4: Measure file sizes and peak memory
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_gb = peak_rss_kb / (1024.0 * 1024.0)

    file_sizes_bytes: dict[str, int] = {}
    total_hw_bytes = 0
    for p in sorted(hw_out_dir.iterdir()):
        if p.is_file():
            sz = p.stat().st_size
            file_sizes_bytes[p.name] = sz
            total_hw_bytes += sz

    addr_map = bundle_res["address_map"]
    w_words = addr_map["weights_image"]["total_words"]
    p_words = addr_map["requant_image"]["total_words"]

    host_ram_avail_gb = budgets["host"]["ram_available_gb"]
    disk_free_gb = budgets["host"]["disk_free_gb"]
    total_hw_gb = total_hw_bytes / (1024.0**3)

    within_ram_budget = peak_rss_gb < host_ram_avail_gb
    within_disk_budget = total_hw_gb < disk_free_gb

    report: dict[str, Any] = {
        "task": "UE.1",
        "model_id": DEFAULT_MODEL_ID,
        "revision": PINNED_REVISION,
        "config": artifacts["config"],
        "timings_sec": {
            "extract_sec": round(t_extract - t0, 3),
            "graph_build_sec": round(t_graph - t_extract, 3),
            "emit_bundle_sec": round(t_emit - t_graph, 3),
            "total_sec": round(t_emit - t0, 3),
        },
        "peak_rss_gb": round(peak_rss_gb, 3),
        "rom_words": {
            "weights_8b_words": w_words,
            "weights_128b_words": (w_words + 15) // 16,
            "requant_params_64b_words": p_words,
            "program_128b_words": len(bundle_res["microprogram"].instructions),
        },
        "file_sizes_bytes": file_sizes_bytes,
        "total_hw_bundle_mb": round(total_hw_bytes / (1024.0**2), 2),
        "budget_check": {
            "host_ram_available_gb": host_ram_avail_gb,
            "within_ram_budget": within_ram_budget,
            "host_disk_free_gb": disk_free_gb,
            "within_disk_budget": within_disk_budget,
        },
        "status": "PASS" if (within_ram_budget and within_disk_budget) else "FAIL",
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Wrote {out_json_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
