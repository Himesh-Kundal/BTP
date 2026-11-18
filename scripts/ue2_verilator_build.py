"""Task UE.2: Verilator build of the full SmolLM2-135M top_accel.

Records Verilator build time, peak build RAM, binary size, simulation wall time,
peak simulation RAM, cycles executed, and simulation throughput (cycles/s).
Outputs: results/part2/ue2_verilator_build.json
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from sim.tb_generator import generate_top_accel_tb_verilog


def _parse_max_rss_kb(stderr_text: str) -> int | None:
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr_text)
    return int(m.group(1)) if m else None


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    hw_dir = repo_root / "build" / "smollm2_hw"
    ue1_path = repo_root / "results" / "part2" / "ue1_generation.json"
    out_json_path = repo_root / "results" / "part2" / "ue2_verilator_build.json"

    if not (hw_dir / "top_accel.v").exists():
        raise FileNotFoundError(f"Missing {hw_dir / 'top_accel.v'}; run scripts/ue1_generate_full.py first.")

    with open(ue1_path, "r", encoding="utf-8") as f:
        ue1 = json.load(f)
    cfg = ue1["config"]

    ver_bin = shutil.which("verilator")
    if not ver_bin:
        raise RuntimeError("verilator not found in PATH")
    time_bin = shutil.which("time")  # /usr/bin/time for accurate child RSS

    sim_dir = hw_dir / "verilator_tb"
    sim_dir.mkdir(parents=True, exist_ok=True)

    # Generate testbench for 1 prompt token + 1 generated token (2 steps)
    tb_code = generate_top_accel_tb_verilog(
        prompt_tokens=[1],
        num_generate=1,
        seq_id=0,
        trace_dir=None,
        expected_tokens=None,
        timeout_cycles=50_000_000,
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        num_q_heads=cfg["num_attention_heads"],
        num_kv_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
    )
    tb_file = hw_dir / "tb_top_accel.v"
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

    import resource

    if time_bin:
        compile_cmd = [time_bin, "-v"] + compile_cmd

    print("[*] Compiling full SmolLM2-135M top_accel with Verilator...")
    t0 = time.perf_counter()
    res_c = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
    build_time_sec = time.perf_counter() - t0

    if res_c.returncode != 0:
        raise RuntimeError(f"Verilator build failed (rc={res_c.returncode}):\n{res_c.stderr}\n{res_c.stdout}")

    build_rss_kb = _parse_max_rss_kb(res_c.stderr) or resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    sim_bin = sim_dir / "Vtb_top_accel"
    bin_size_mb = sim_bin.stat().st_size / (1024.0**2)
    print(f"[+] Verilator build succeeded in {build_time_sec:.2f}s (binary={bin_size_mb:.2f} MB)")

    # Run compiled binary (1 prompt + 1 gen = 2 steps)
    run_cmd = [str(sim_bin)]
    if time_bin:
        run_cmd = [time_bin, "-v"] + run_cmd

    print("[*] Running 2-step full SmolLM2-135M simulation...")
    t_run0 = time.perf_counter()
    res_r = subprocess.run(run_cmd, capture_output=True, text=True, check=False)
    sim_time_sec = time.perf_counter() - t_run0

    if res_r.returncode != 0:
        raise RuntimeError(f"Verilator simulation failed (rc={res_r.returncode}):\n{res_r.stderr}\n{res_r.stdout}")

    raw_out = res_r.stdout + res_r.stderr
    m_alloc = re.search(r"alloced\s+(\d+)\s+MB", raw_out)
    sim_rss_kb = _parse_max_rss_kb(res_r.stderr) or (int(m_alloc.group(1)) * 1024 if m_alloc else resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)

    m_cyc = re.search(r"TOTAL_CYCLES=(\d+)", raw_out)
    total_cycles = int(m_cyc.group(1)) if m_cyc else 0
    tokens = [int(m.group(2)) for m in re.finditer(r"STEP\s+(\d+)\s+TOK_OUT=(\d+)", raw_out)]
    cycles_per_sec = total_cycles / sim_time_sec if sim_time_sec > 0 else 0.0

    report: dict[str, Any] = {
        "task": "UE.2",
        "build_success": True,
        "build_time_sec": round(build_time_sec, 3),
        "build_peak_rss_gb": round(build_rss_kb / (1024.0**2), 3) if build_rss_kb else None,
        "binary_size_mb": round(bin_size_mb, 2),
        "sim_steps": len(tokens),
        "tokens_out": tokens,
        "sim_wall_time_sec": round(sim_time_sec, 3),
        "sim_peak_rss_gb": round(sim_rss_kb / (1024.0**2), 3) if sim_rss_kb else None,
        "total_cycles": total_cycles,
        "cycles_per_token": total_cycles // max(1, len(tokens)),
        "sim_speed_cycles_per_sec": round(cycles_per_sec, 1),
        "sim_speed_kcycles_per_sec": round(cycles_per_sec / 1000.0, 2),
        "status": "PASS" if (total_cycles > 0 and len(tokens) == 2) else "FAIL",
    }

    out_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Wrote {out_json_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
