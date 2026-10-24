"""Yosys Synthesis Runner and Metric Parser (T7.1, docs/06 §B2, §B3).

Automates logic synthesis and metric extraction for Design F, Design P, and Design P_bus:
- Targets:
  1. "generic": Standard gate-level logic synthesis (`synth -top <top>; stat -json; ltp`).
  2. "ice40": Lattice iCE40 FPGA LUT-4 cell mapping (`synth_ice40 -top <top>; stat -json; ltp`).
- Metrics extracted:
  * Total cells, combinational cells, sequential cells (flip-flops).
  * LUT-4 count (for iCE40 proxy).
  * Wire count and port bit counts.
  * Longest topological path depth (`ltp`) as a depth proxy.
  * Synthesis wall time.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend.baseline_p import generate_full_module_p, generate_full_module_p_bus
from backend.module_generator import generate_full_module
from backend.tiling import TileManifest
from ir.linear import LinearOp


@dataclass
class SynthesisResult:
    """Structured metrics produced by a Yosys synthesis run."""

    top_module: str
    target: str  # "generic" or "ice40"
    flatten: bool
    returncode: int
    wall_time_seconds: float
    num_wires: int
    num_wire_bits: int
    num_ports: int
    num_port_bits: int
    num_cells_total: int
    num_cells_comb: int
    num_cells_seq: int  # Flip-flops
    num_lut4: int  # LUT4 count for FPGA targets
    ltp_depth: int  # Longest topological path length
    cell_counts: dict[str, int]
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return asdict(self)


def parse_yosys_stat_json(stdout: str) -> dict[str, Any]:
    """Extract and parse the JSON block emitted by Yosys 'stat -json'."""
    marker = '"invocation": "stat -json'
    idx = stdout.find(marker)
    if idx == -1:
        marker = '"modules":'
        idx = stdout.find(marker)
    if idx == -1:
        raise ValueError("Could not find Yosys 'stat -json' output block in stdout.")

    # Backtrack to the opening brace '{'
    start = stdout.rfind("{", 0, idx)
    if start == -1:
        raise ValueError("Could not find opening brace for Yosys 'stat -json'.")

    # Track balanced braces to find the matching closing brace
    brace_depth = 0
    end = -1
    for i in range(start, len(stdout)):
        if stdout[i] == "{":
            brace_depth += 1
        elif stdout[i] == "}":
            brace_depth -= 1
            if brace_depth == 0:
                end = i + 1
                break

    if end == -1:
        raise ValueError("Unterminated JSON block in Yosys stdout.")

    json_str = stdout[start:end]
    return json.loads(json_str)


def parse_yosys_ltp(stdout: str) -> int:
    """Extract longest topological path depth from Yosys 'ltp' command output."""
    m = re.search(r"Longest topological path in \S+ \(length=(\d+)\):", stdout)
    if m:
        return int(m.group(1))
    return 0


def classify_cells(cell_counts: dict[str, int]) -> tuple[int, int, int]:
    """Classify cell counts into (sequential_flops, combinational_cells, lut4_cells)."""
    seq_cells = 0
    lut4_cells = 0
    comb_cells = 0

    for cell_type, count in cell_counts.items():
        if cell_type == "$scopeinfo":
            continue
        upper = cell_type.upper()
        if "DFF" in upper:
            seq_cells += count
        elif "LUT4" in upper:
            lut4_cells += count
            comb_cells += count
        else:
            comb_cells += count

    return seq_cells, comb_cells, lut4_cells


def run_yosys_synthesis(
    verilog_paths: list[Path] | Path,
    top_module: str,
    target: str = "generic",
    flatten: bool = True,
    work_dir: Path | None = None,
    timeout_seconds: float = 300.0,
) -> SynthesisResult:
    """Execute non-interactive Yosys synthesis script and extract structured metrics."""
    if isinstance(verilog_paths, Path):
        files = [verilog_paths]
    else:
        files = list(verilog_paths)

    if target not in {"generic", "ice40"}:
        raise ValueError(f"Unsupported synthesis target: '{target}'. Must be 'generic' or 'ice40'.")

    # Build Yosys synthesis script
    script_lines = []
    for f in files:
        script_lines.append(f"read_verilog {f.resolve()}")

    script_lines.append(f"hierarchy -check -top {top_module}")

    flatten_flag = " -flatten" if flatten else ""
    if target == "generic":
        script_lines.append(f"synth -top {top_module}{flatten_flag}")
    elif target == "ice40":
        script_lines.append(f"synth_ice40 -top {top_module}{flatten_flag}")

    script_lines.append("stat -json")
    script_lines.append("ltp")

    script = "\n".join(script_lines) + "\n"

    # Temporary directory if work_dir not supplied
    ctx: Any
    if work_dir is None:
        ctx = tempfile.TemporaryDirectory(prefix="synth_")
        run_dir = Path(ctx.name)
    else:
        ctx = None
        run_dir = work_dir
        run_dir.mkdir(parents=True, exist_ok=True)

    script_file = run_dir / "synth.ys"
    script_file.write_text(script, encoding="utf-8")

    start_t = time.perf_counter()
    try:
        proc = subprocess.run(
            ["yosys", "-s", str(script_file)],
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        elapsed = time.perf_counter() - start_t
    except subprocess.TimeoutExpired:
        if ctx is not None:
            ctx.cleanup()
        raise TimeoutError(f"Yosys synthesis timed out after {timeout_seconds}s on {top_module}")

    if proc.returncode != 0:
        if ctx is not None:
            ctx.cleanup()
        return SynthesisResult(
            top_module=top_module,
            target=target,
            flatten=flatten,
            returncode=proc.returncode,
            wall_time_seconds=elapsed,
            num_wires=0,
            num_wire_bits=0,
            num_ports=0,
            num_port_bits=0,
            num_cells_total=0,
            num_cells_comb=0,
            num_cells_seq=0,
            num_lut4=0,
            ltp_depth=0,
            cell_counts={},
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

    # Parse JSON and LTP metrics
    stat_data = parse_yosys_stat_json(proc.stdout)
    design_stat = stat_data.get("design", {})
    cell_counts = design_stat.get("num_cells_by_type", {})
    ltp_depth = parse_yosys_ltp(proc.stdout)

    seq_cells, comb_cells, lut4_cells = classify_cells(cell_counts)

    res = SynthesisResult(
        top_module=top_module,
        target=target,
        flatten=flatten,
        returncode=proc.returncode,
        wall_time_seconds=elapsed,
        num_wires=design_stat.get("num_wires", 0),
        num_wire_bits=design_stat.get("num_wire_bits", 0),
        num_ports=design_stat.get("num_ports", 0),
        num_port_bits=design_stat.get("num_port_bits", 0),
        num_cells_total=design_stat.get("num_cells", 0),
        num_cells_comb=comb_cells,
        num_cells_seq=seq_cells,
        num_lut4=lut4_cells,
        ltp_depth=ltp_depth,
        cell_counts=cell_counts,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )

    if ctx is not None:
        ctx.cleanup()

    return res


def synthesize_design(
    op: LinearOp,
    manifest: TileManifest,
    design_type: str = "F",  # "F", "P", or "P_bus"
    target: str = "generic",  # "generic" or "ice40"
    flatten: bool = True,
    latency: int = 1,
    work_dir: Path | None = None,
    timeout_seconds: float = 300.0,
) -> SynthesisResult:
    """Generate RTL for specified design type and execute Yosys synthesis flow."""
    if design_type not in {"F", "P", "P_bus"}:
        raise ValueError(f"Unknown design_type: {design_type}. Must be 'F', 'P', or 'P_bus'.")

    ctx: Any
    if work_dir is None:
        ctx = tempfile.TemporaryDirectory(prefix="synth_design_")
        run_dir = Path(ctx.name)
    else:
        ctx = None
        run_dir = work_dir
        run_dir.mkdir(parents=True, exist_ok=True)

    if design_type == "F":
        top_module = f"llmhw_{op.name}"
        rtl = generate_full_module(op, manifest, latency=latency)
    elif design_type == "P":
        top_module = f"llmhw_{op.name}_p"
        rtl = generate_full_module_p(op, manifest)
    else:  # "P_bus"
        top_module = f"llmhw_{op.name}_p_bus"
        rtl = generate_full_module_p_bus(op, manifest)

    dut_file = run_dir / f"{top_module}.v"
    dut_file.write_text(rtl, encoding="utf-8")

    result = run_yosys_synthesis(
        verilog_paths=dut_file,
        top_module=top_module,
        target=target,
        flatten=flatten,
        work_dir=run_dir,
        timeout_seconds=timeout_seconds,
    )

    if ctx is not None:
        ctx.cleanup()

    return result


def main() -> int:
    """CLI demo entry point for synthesis runner."""
    import argparse

    import numpy as np

    from backend.tiling import tile_weight_matrix

    parser = argparse.ArgumentParser(description="Yosys Synthesis Runner for LLMHW")
    parser.add_argument("--design", choices=["F", "P", "P_bus"], default="F", help="Design variant")
    parser.add_argument("--target", choices=["generic", "ice40"], default="generic", help="Target techmap")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    # Demo on 4x8 tile
    w = np.array(
        [
            [12, -7, 0, 15, -25, 35, 0, -10],
            [-3, 8, 22, -1, 10, -20, 30, -40],
            [5, -12, 3, 0, 15, -25, 0, 50],
            [0, 14, -8, 2, -18, 24, -30, 0],
        ],
        dtype=np.int8,
    )
    op = LinearOp(name="synth_demo", in_features=8, out_features=4, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=4, tile_k=8)

    print(f"Synthesizing demo tile {w.shape} for Design {args.design} on target '{args.target}'...")
    res = synthesize_design(op, manifest, design_type=args.design, target=args.target)

    if res.returncode != 0:
        print(f"Synthesis FAILED (code {res.returncode}):\n{res.stderr}")
        return 1

    print("Synthesis PASSED:")
    print(f"  Wall time        : {res.wall_time_seconds:.2f}s")
    print(f"  Total cells      : {res.num_cells_total}")
    print(f"  Combinational    : {res.num_cells_comb}")
    print(f"  Sequential (DFF) : {res.num_cells_seq}")
    if args.target == "ice40":
        print(f"  LUT4 cells       : {res.num_lut4}")
    print(f"  LTP depth        : {res.ltp_depth}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res.to_dict(), f, indent=2)
        print(f"Saved metrics to: {args.out}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
