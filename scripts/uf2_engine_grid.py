"""UF.2 / EA2: Engine Grid (R, C) Design Exploration.

Explores linear_engine grid configurations (R, C) in {(4,4), (8,4), (8,8), (8,16), (16,16), (16,32)}:
1. Synthesizes linear_engine at each grid for 'generic' and 'ice40' targets.
2. Extracts area (cells, LUT4, flip-flops) and topological critical path depth (ltp).
3. Computes analytical cycle estimates per token for SmolLM2-135M.
4. Validates analytical model against actual simulated cycles from UE.4 (14,100,262 cycles/token).
5. Writes results/part2/EA2.csv and results/part2/EA2.json.
"""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from synth.yosys_runner import classify_cells, parse_yosys_ltp, parse_yosys_stat_json

REPO_ROOT = Path(__file__).resolve().parent.parent
RTL_LIB = REPO_ROOT / "rtl" / "lib"
RESULTS_DIR = REPO_ROOT / "results" / "part2"

GRIDS = [
    (4, 4),
    (8, 4),
    (8, 8),
    (8, 16),
    (16, 16),
    (16, 32),
]

ACTUAL_SIM_CYCLES_UE4 = 14100262


def compute_analytic_cycles(R: int, C: int, pos: int = 16) -> int:
    """Accurate analytical cycle model for SmolLM2-135M decoder step on (R, C) grid."""
    total = 0
    # Embedding gather
    total += 576 + 2

    # 30 Transformer layers
    for _ in range(30):
        # 1. Attn norm: sum-of-squares (576) + rsqrt + normalize (576) + latency (12)
        total += 2 * 576 + 12

        # 2. Q, K, V Projections
        for n, k in [(576, 576), (192, 576), (192, 576)]:
            tn = (n + R - 1) // R
            tk = (k + C - 1) // C
            total += tn * (tk * (R + 4) + R + 2) + 2

        # 3. RoPE
        total += 576 // 2 + 4

        # 4. GQA Attention datapath (9 Q heads, 3 KV heads, dim 64)
        att_cyc = 128 + 9 * (64 + (pos + 1) * 129 + 15)
        total += att_cyc

        # 5. Output Projection
        tn = (576 + R - 1) // R
        tk = (576 + C - 1) // C
        total += tn * (tk * (R + 4) + R + 2) + 2

        # 6. Residual Add 1
        total += 576 + 4

        # 7. MLP norm
        total += 2 * 576 + 12

        # 8. Gate and Up Projections
        for n, k in [(1536, 576), (1536, 576)]:
            tn = (n + R - 1) // R
            tk = (k + C - 1) // C
            total += tn * (tk * (R + 4) + R + 2) + 2

        # 9. SwiGLU
        total += 1536 + 8

        # 10. Down Projection
        tn = (576 + R - 1) // R
        tk = (1536 + C - 1) // C
        total += tn * (tk * (R + 4) + R + 2) + 2

        # 11. Residual Add 2
        total += 576 + 4

    # Final RMSNorm
    total += 2 * 576 + 12

    # LM Head Projection (Vocab = 49152, Hidden = 576)
    tn = (49152 + R - 1) // R
    tk = (576 + C - 1) // C
    total += tn * (tk * (R + 4) + R + 2) + 2

    # Argmax
    total += 49152 + 4

    return total


def synthesize_grid(R: int, C: int, target: str) -> dict[str, Any]:
    files = [
        (RTL_LIB / "requant.v").resolve(),
        (RTL_LIB / "linear_engine.v").resolve(),
    ]
    with tempfile.TemporaryDirectory(prefix=f"synth_grid_{R}x{C}_") as td:
        run_dir = Path(td)
        ys_lines = [f"read_verilog -I{RTL_LIB.resolve()} {' '.join(str(f) for f in files)}"]
        ys_lines.append(f"chparam -set R {R} -set C {C} linear_engine")
        ys_lines.append("hierarchy -check -top linear_engine")
        if target == "generic":
            ys_lines.append("synth -top linear_engine -flatten")
        elif target == "ice40":
            ys_lines.append("synth_ice40 -top linear_engine -flatten")
        ys_lines.append("stat -json")
        ys_lines.append("ltp")

        script_file = run_dir / "synth.ys"
        script_file.write_text("\n".join(ys_lines) + "\n", encoding="utf-8")

        t0 = time.perf_counter()
        proc = subprocess.run(
            ["yosys", "-s", str(script_file)],
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=600.0,
        )
        wall_s = time.perf_counter() - t0

        if proc.returncode != 0:
            raise RuntimeError(f"Yosys failed on {R}x{C} ({target}):\n{proc.stderr}")

        stat = parse_yosys_stat_json(proc.stdout)
        ltp = parse_yosys_ltp(proc.stdout)
        mod_stat = stat["modules"]["\\linear_engine"]
        seq, comb, lut4 = classify_cells(mod_stat["num_cells_by_type"])

        return {
            "R": R,
            "C": C,
            "lanes": R * C,
            "target": target,
            "cells_total": mod_stat["num_cells"],
            "cells_comb": comb,
            "cells_seq": seq,
            "num_lut4": lut4,
            "ltp_depth": ltp,
            "wall_s": round(wall_s, 2),
        }


def run_ea2() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "EA2.csv"
    json_path = RESULTS_DIR / "EA2.json"

    print("[*] Starting EA2: Engine Grid Exploration across 6 configurations...")
    records: list[dict[str, Any]] = []

    for R, C in GRIDS:
        lanes = R * C
        analytic_cyc = compute_analytic_cycles(R, C, pos=16)
        err_pct = round(((analytic_cyc - ACTUAL_SIM_CYCLES_UE4) / ACTUAL_SIM_CYCLES_UE4) * 100.0, 2)
        print(f"\n[+] Grid (R={R}, C={C}) -> {lanes} MAC lanes | Analytic cycles: {analytic_cyc:,} (err vs UE.4: {err_pct}%)")

        for target in ["generic", "ice40"]:
            print(f"    Synthesizing {target}...")
            res = synthesize_grid(R, C, target)
            res["analytic_cycles_per_token"] = analytic_cyc
            res["simulated_cycles_per_token_baseline"] = ACTUAL_SIM_CYCLES_UE4 if (R == 8 and C == 16) else None
            res["model_error_pct"] = err_pct if (R == 8 and C == 16) else None
            records.append(res)
            print(
                f"      -> cells={res['cells_total']}, seq={res['cells_seq']}, "
                f"lut4={res['num_lut4']}, ltp={res['ltp_depth']} in {res['wall_s']}s"
            )

    fieldnames = [
        "R",
        "C",
        "lanes",
        "target",
        "cells_total",
        "cells_comb",
        "cells_seq",
        "num_lut4",
        "ltp_depth",
        "analytic_cycles_per_token",
        "simulated_cycles_per_token_baseline",
        "model_error_pct",
        "wall_s",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k) for k in fieldnames})
    print(f"\n[+] Wrote {csv_path}")

    summary = {
        "task": "UF.2",
        "experiment": "EA2",
        "num_grids": len(GRIDS),
        "actual_sim_cycles_ue4": ACTUAL_SIM_CYCLES_UE4,
        "records": records,
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Wrote {json_path}")


if __name__ == "__main__":
    run_ea2()
