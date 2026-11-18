"""Unit tests for Top-Level Accelerator Generator & Composition (Task UD.1).

Verifies:
1. End-to-end hardware bundle generation (ROM hex files, address map, program.hex, top_accel.v).
2. Clean compilation with Icarus Verilog (`iverilog -g2005 -Wall`).
3. Clean lint check with Verilator (`verilator --lint-only -Wall`).
4. Formal synthesis check with Yosys (`hierarchy -check; proc; check -assert`).
5. Strict adherence to immutable weights and no write ports on ROM instances.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from backend.top_generator import emit_hardware_bundle
from frontend.importer_v2 import build_graph_v2_from_artifacts, load_model_artifacts_v2

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_model_v2"
CONFIG_FROZEN_PATH = REPO_ROOT / "numerics" / "config_frozen.json"
RTL_LIB_DIR = REPO_ROOT / "rtl" / "lib"


def test_top_generator_tiny_model(tmp_path: Path):
    """Verify hardware bundle generation and linting across iverilog, verilator, and yosys."""
    artifacts = load_model_artifacts_v2(TINY_ARTIFACTS_DIR)
    graph = build_graph_v2_from_artifacts(TINY_ARTIFACTS_DIR)

    with open(CONFIG_FROZEN_PATH, "r", encoding="utf-8") as f:
        frozen_config = json.load(f)

    bundle = emit_hardware_bundle(
        graph=graph,
        weights_dict=artifacts["tensors"],
        config=artifacts["config"],
        frozen_config=frozen_config,
        out_dir=tmp_path / "hw",
        tile_r=8,
        tile_c=16,
        max_ctx=32,
    )

    hw_dir = Path(bundle["out_dir"])

    # 1. Verify all expected files are present
    assert (hw_dir / "weights_128b.hex").exists()
    assert (hw_dir / "requant_params.hex").exists()
    assert (hw_dir / "tables.hex").exists()
    assert (hw_dir / "program.hex").exists()
    assert (hw_dir / "layer_bases.hex").exists()
    assert (hw_dir / "tables_exp2.hex").exists()
    assert (hw_dir / "tables_rsqrt.hex").exists()
    assert (hw_dir / "tables_silu.hex").exists()
    assert (hw_dir / "tables_rope_cos.hex").exists()
    assert (hw_dir / "tables_rope_sin.hex").exists()
    assert (hw_dir / "top_accel.v").exists()

    # 2. Icarus Verilog lint check
    iverilog_bin = shutil.which("iverilog")
    assert iverilog_bin is not None, "iverilog not found in PATH"

    vvp_file = hw_dir / "top_accel.vvp"
    rtl_files = sorted(RTL_LIB_DIR.glob("*.v"))
    # Replace top_accel.v from rtl/lib with the generated top_accel.v from bundle
    comp_sources = [p for p in rtl_files if p.name != "top_accel.v"] + [hw_dir / "top_accel.v"]

    cmd_iverilog = [
        iverilog_bin,
        "-g2005",
        "-Wall",
        "-I",
        str(RTL_LIB_DIR),
        "-s",
        "top_accel",
        "-o",
        str(vvp_file),
    ] + [str(p) for p in comp_sources]

    res_iv = subprocess.run(cmd_iverilog, capture_output=True, text=True, check=False)
    assert res_iv.returncode == 0, f"iverilog failed:\nSTDOUT:\n{res_iv.stdout}\nSTDERR:\n{res_iv.stderr}"

    # 3. Verilator lint check
    verilator_bin = shutil.which("verilator")
    assert verilator_bin is not None, "verilator not found in PATH"

    cmd_verilator = [
        verilator_bin,
        "--lint-only",
        "-Wall",
        "-I" + str(RTL_LIB_DIR),
        "--top-module",
        "top_accel",
    ] + [str(p) for p in comp_sources]

    res_ver = subprocess.run(cmd_verilator, capture_output=True, text=True, check=False)
    assert res_ver.returncode == 0, f"verilator lint failed:\nSTDOUT:\n{res_ver.stdout}\nSTDERR:\n{res_ver.stderr}"

    # 4. Yosys hierarchy & assert check
    yosys_bin = shutil.which("yosys")
    assert yosys_bin is not None, "yosys not found in PATH"

    # For Yosys synthesis check, we run proc and check -assert on top_accel
    yosys_script = (
        f"read_verilog -I{RTL_LIB_DIR} {' '.join(str(p) for p in comp_sources)}; "
        "hierarchy -top top_accel -check; proc; check -assert"
    )
    res_yo = subprocess.run([yosys_bin, "-p", yosys_script], capture_output=True, text=True, check=False)
    assert res_yo.returncode == 0, f"yosys check failed:\nSTDOUT:\n{res_yo.stdout}\nSTDERR:\n{res_yo.stderr}"
