"""UF.1 / EA1: Unit-level and Engine Logic Synthesis Study (Yosys).

Synthesizes each parameterized unit and the linear engine using Yosys:
1. "generic" standard logic gates.
2. "ice40" FPGA LUT-4 cell mapping proxy.
Extracts cell breakdown, flip-flops, LUTs, and longest topological path (ltp) depth.
Emits results/part2/EA1.csv and results/part2/EA1.json.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from synth.yosys_runner import run_yosys_synthesis

REPO_ROOT = Path(__file__).resolve().parent.parent
RTL_LIB = REPO_ROOT / "rtl" / "lib"
RESULTS_DIR = REPO_ROOT / "results" / "part2"

UNITS_SPEC = [
    {
        "unit": "requant",
        "module": "requant",
        "sources": [RTL_LIB / "requant.v"],
    },
    {
        "unit": "residual_add",
        "module": "residual_add",
        "sources": [RTL_LIB / "residual_add.v"],
    },
    {
        "unit": "argmax",
        "module": "argmax",
        "sources": [RTL_LIB / "argmax.v"],
    },
    {
        "unit": "kv_addr_gen",
        "module": "kv_addr_gen",
        "sources": [RTL_LIB / "kv_addr_gen.v"],
    },
    {
        "unit": "embed_gather",
        "module": "embed_gather",
        "sources": [RTL_LIB / "embed_gather.v"],
    },
    {
        "unit": "rope",
        "module": "rope",
        "sources": [RTL_LIB / "rom_sync.v", RTL_LIB / "rope.v"],
    },
    {
        "unit": "swiglu",
        "module": "swiglu",
        "sources": [RTL_LIB / "rom_sync.v", RTL_LIB / "swiglu.v"],
    },
    {
        "unit": "sequencer",
        "module": "sequencer",
        "sources": [RTL_LIB / "rom_sync.v", RTL_LIB / "sequencer.v"],
    },
    {
        "unit": "linear_engine",
        "module": "linear_engine",
        "sources": [RTL_LIB / "requant.v", RTL_LIB / "linear_engine.v"],
    },
    {
        "unit": "rmsnorm",
        "module": "rmsnorm",
        "sources": [RTL_LIB / "rom_sync.v", RTL_LIB / "rmsnorm.v"],
    },
    {
        "unit": "softmax",
        "module": "softmax",
        "sources": [RTL_LIB / "rom_sync.v", RTL_LIB / "softmax.v"],
    },
    {
        "unit": "attention",
        "module": "attention",
        "sources": [RTL_LIB / "rom_sync.v", RTL_LIB / "softmax.v", RTL_LIB / "attention.v"],
    },
]


def run_ea1() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "EA1.csv"
    json_path = RESULTS_DIR / "EA1.json"

    records: list[dict[str, Any]] = []

    print("[*] Starting EA1: Unit & Engine Synthesis Study across 12 units...")
    t0_all = time.perf_counter()

    for item in UNITS_SPEC:
        unit_name = item["unit"]
        mod_name = item["module"]
        sources = item["sources"]

        for target in ["generic", "ice40"]:
            print(f"  [+] Synthesizing {unit_name} ({target})...")
            t_s = time.perf_counter()
            res = run_yosys_synthesis(
                verilog_paths=sources,
                top_module=mod_name,
                target=target,
                flatten=True,
                timeout_seconds=600.0,
            )
            wall_s = time.perf_counter() - t_s

            if res.returncode != 0:
                print(f"[-] ERROR synthesizing {unit_name} ({target}):\n{res.stderr}")
                raise RuntimeError(f"Synthesis failed for {unit_name} ({target})")

            rec = {
                "unit": unit_name,
                "module": mod_name,
                "target": target,
                "cells_total": res.num_cells_total,
                "cells_comb": res.num_cells_comb,
                "cells_seq": res.num_cells_seq,
                "num_lut4": res.num_lut4,
                "num_wires": res.num_wires,
                "num_wire_bits": res.num_wire_bits,
                "ltp_depth": res.ltp_depth,
                "wall_time_s": round(wall_s, 2),
                "cell_breakdown": res.cell_counts,
            }
            records.append(rec)
            print(
                f"      -> total={res.num_cells_total}, seq={res.num_cells_seq}, "
                f"lut4={res.num_lut4}, ltp={res.ltp_depth} in {wall_s:.2f}s"
            )

    total_wall = time.perf_counter() - t0_all
    print(f"[+] All 12 units synthesized for generic and ice40 in {total_wall:.2f}s")

    # Write CSV
    fieldnames = [
        "unit",
        "module",
        "target",
        "cells_total",
        "cells_comb",
        "cells_seq",
        "num_lut4",
        "num_wires",
        "num_wire_bits",
        "ltp_depth",
        "wall_time_s",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"[+] Wrote {csv_path}")

    # Write JSON with detailed cell breakdown
    summary = {
        "task": "UF.1",
        "experiment": "EA1",
        "num_units": len(UNITS_SPEC),
        "total_runs": len(records),
        "total_wall_s": round(total_wall, 2),
        "records": records,
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Wrote {json_path}")


if __name__ == "__main__":
    run_ea1()
