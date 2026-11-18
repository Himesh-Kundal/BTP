"""Cross-Simulator Equality Verification (Task UD.4).

Verifies:
1. Exact trace equality between Icarus Verilog (`iverilog`/`vvp`) and Verilator (`--binary --timing`).
2. Generated token equality between both simulation engines.
3. Hierarchical trace diff across all decoded tokens, layers, and operations is 100% empty
   under standard POSIX `diff -r` as well as `compare_trace_directories`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from backend.top_generator import emit_hardware_bundle
from frontend.importer_v2 import build_graph_v2_from_artifacts, load_model_artifacts_v2
from sim.tb_generator import run_top_accel_simulation
from verification.trace_diff import compare_trace_directories

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_model_v2"
CONFIG_FROZEN_PATH = REPO_ROOT / "numerics" / "config_frozen.json"


@pytest.fixture(scope="module")
def tiny_hardware_bundle(tmp_path_factory) -> Path:
    """Generate hardware bundle for tiny model once per test module."""
    bundle_dir = tmp_path_factory.mktemp("tiny_hw_cross_sim")
    artifacts = load_model_artifacts_v2(TINY_ARTIFACTS_DIR)
    graph = build_graph_v2_from_artifacts(TINY_ARTIFACTS_DIR)
    with open(CONFIG_FROZEN_PATH, "r", encoding="utf-8") as f:
        frozen_config = json.load(f)

    emit_hardware_bundle(
        graph=graph,
        weights_dict=artifacts["tensors"],
        config=artifacts["config"],
        frozen_config=frozen_config,
        out_dir=bundle_dir,
        tile_r=8,
        tile_c=16,
        max_ctx=32,
    )
    return bundle_dir


def test_cross_simulator_trace_and_token_equality(tiny_hardware_bundle: Path, tmp_path: Path):
    """Run prompt through Icarus Verilog and Verilator and assert empty diff."""
    bdir = tiny_hardware_bundle
    prompt = [42, 17]
    num_gen = 1

    trace_iverilog = tmp_path / "traces_iverilog"
    trace_verilator = tmp_path / "traces_verilator"

    # 1. Simulate using Icarus Verilog
    res_iv = run_top_accel_simulation(
        bundle_dir=bdir,
        prompt_tokens=prompt,
        num_generate=num_gen,
        trace_dir=trace_iverilog,
        simulator="iverilog",
    )
    assert res_iv.success, f"Icarus simulation failed: {res_iv.raw_output}"

    # 2. Simulate using Verilator
    res_ver = run_top_accel_simulation(
        bundle_dir=bdir,
        prompt_tokens=prompt,
        num_generate=num_gen,
        trace_dir=trace_verilator,
        simulator="verilator",
    )
    assert res_ver.success, f"Verilator simulation failed: {res_ver.raw_output}"

    # 3. Assert token equality
    assert res_iv.tokens == res_ver.tokens, f"Tokens mismatch: Icarus={res_iv.tokens} vs Verilator={res_ver.tokens}"

    # 4. Compare dumped trace trees using verification/trace_diff.py
    diff_res = compare_trace_directories(trace_iverilog, trace_verilator)
    assert diff_res.is_identical, f"Trace mismatch between Icarus and Verilator: {diff_res.first_divergence}"
    assert diff_res.total_elements_checked > 0, "No trace elements were checked"

    # 5. POSIX diff -r verification
    diff_proc = subprocess.run(
        ["diff", "-r", str(trace_iverilog), str(trace_verilator)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert diff_proc.returncode == 0, f"POSIX diff -r failed with code {diff_proc.returncode}"
    assert diff_proc.stdout == "", f"POSIX diff -r output non-empty:\n{diff_proc.stdout}"
