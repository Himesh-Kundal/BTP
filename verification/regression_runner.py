"""Standalone Regression Test Runner (T6.6, docs/06 §A3, §A8).

Executes the full verification hardening matrix across all levels and suites:
- Suite L0: Toy models (1x3 combinational, 2x4 registered wrapper)
- Suite L1: Synthetic matrix (N, K, tile configs, 8 weight classes, 6 input classes, biases)
- Suite L2: Real layer tiles (8x16 slices across 7 projection types at 3 depths)
- Suite L3: Real multi-tile slices (16x32 2x2 grid across 3 depths)
- Suite Cross-Sim: Verilator lint clean, Verilator vs Icarus vs Reference bit-exact equality
- Suite Weight-Audit: RTL constant parser vs W_q, manifest sha256, port audit, negative controls
- Suite Mutation: M1-M9 fault injection campaign asserting 100% kill rate (0 survivors)

Produces structured execution metrics in results/regression.json.
Conforms strictly to Golden Rule 5 (no model-specific hardcoding, no torch imports).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.baseline_p import generate_full_module_p, generate_full_module_p_bus
from backend.codegen import generate_fixed_core
from backend.module_generator import generate_full_module
from backend.testbench import generate_core_testbench, generate_wrapper_testbench
from backend.tiling import pad_activations, tile_weight_matrix
from backend.vectors import write_hex_vectors
from compiler.calibration import (
    compute_activation_scale,
    compute_activation_stat,
    quantize_activations,
)
from compiler.quantization import quantize_weights
from ir.linear import LinearOp
from verification.mutate import run_mutation_campaign
from verification.reference import linear_reference_int64
from verification.simulator import lint_verilator, run_simulation, run_verilator_simulation
from verification.test_matrix import (
    generate_inputs,
    get_l1_test_matrix,
    run_matrix_case,
)
from verification.weight_audit import audit_weights_match


@dataclass
class TestCaseResult:
    """Result of an individual test case in the regression matrix."""

    name: str
    status: str  # "PASS" or "FAIL"
    checked: int = 0
    mismatches: int = 0
    timing_errors: int = 0
    duration_seconds: float = 0.0
    details: str = ""


@dataclass
class TestSuiteResult:
    """Result of a test suite comprising multiple test cases."""

    name: str
    description: str
    status: str = "PASS"
    tests_run: int = 0
    passed: int = 0
    failed: int = 0
    vectors_checked: int = 0
    duration_seconds: float = 0.0
    cases: list[TestCaseResult] = field(default_factory=list)


def get_git_info(repo_root: Path) -> dict[str, str]:
    """Retrieve current git commit hash and branch."""
    commit = "unknown"
    branch = "unknown"
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            commit = res.stdout.strip()
        res_b = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if res_b.returncode == 0:
            branch = res_b.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        pass
    return {"commit": commit, "branch": branch}


def get_tool_versions(repo_root: Path) -> dict[str, str]:
    """Query versions of Verilog and synthesis tools."""
    tools: dict[str, str] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }

    # iverilog
    try:
        res = subprocess.run(["iverilog", "-V"], capture_output=True, text=True, check=False)
        first_line = res.stdout.splitlines()[0] if res.stdout else "unknown"
        tools["iverilog"] = first_line.strip()
    except (subprocess.SubprocessError, OSError):
        tools["iverilog"] = "unavailable"

    # verilator
    try:
        res = subprocess.run(["verilator", "--version"], capture_output=True, text=True, check=False)
        tools["verilator"] = res.stdout.strip() if res.stdout else "unknown"
    except (subprocess.SubprocessError, OSError):
        tools["verilator"] = "unavailable"

    # yosys
    try:
        res = subprocess.run(["yosys", "-V"], capture_output=True, text=True, check=False)
        tools["yosys"] = res.stdout.strip() if res.stdout else "unknown"
    except (subprocess.SubprocessError, OSError):
        tools["yosys"] = "unavailable"

    return tools


def find_model_artifact_dir(repo_root: Path) -> Path | None:
    """Locate the real model weight artifact directory without hardcoding model names (Rule 5)."""
    art_root = repo_root / "artifacts"
    if not art_root.exists():
        return None
    for p in sorted(art_root.iterdir()):
        if p.is_dir() and (p / "layer_0").exists():
            return p
    return None


# -----------------------------------------------------------------------------
# Suite L0: Toy Models
# -----------------------------------------------------------------------------
def run_suite_l0(tmp_dir: Path) -> TestSuiteResult:
    """Level L0: Toy 1x3 combinational core and 2x4 registered wrapper."""
    suite = TestSuiteResult(
        name="L0_toy",
        description="Level L0 toy combinational and registered cores",
    )
    t0 = time.time()

    # Case 1: Toy 1x3 combinational core
    c1_t0 = time.time()
    w1 = np.array([[10, -5, 1]], dtype=np.int8)
    op1 = LinearOp(name="toy_1x3", in_features=3, out_features=1, weight=w1)
    x1 = np.array([[12, -7, 3], [-10, 20, -30], [127, -128, 0], [0, 0, 0]], dtype=np.int8)
    y1_exp = linear_reference_int64(x1, w1)
    c1_dir = tmp_dir / "l0_c1"
    c1_dir.mkdir(parents=True, exist_ok=True)
    x1_hex, y1_hex = write_hex_vectors(x1, y1_exp, c1_dir)
    dut1_file = c1_dir / "toy_core.v"
    dut1_file.write_text(generate_fixed_core(op1), encoding="utf-8")
    tb1_file = c1_dir / "tb_toy.v"
    tb1_file.write_text(
        generate_core_testbench(
            op1, num_vectors=len(x1), x_hex_path=str(x1_hex), y_hex_path=str(y1_hex)
        ),
        encoding="utf-8",
    )
    res1 = run_simulation(dut1_file, tb1_file, expected_checked=len(x1), work_dir=c1_dir)
    case1 = TestCaseResult(
        name="toy_1x3_combinational",
        status=res1.status,
        checked=res1.checked,
        mismatches=res1.mismatch,
        timing_errors=res1.timing_err,
        duration_seconds=time.time() - c1_t0,
    )
    suite.cases.append(case1)

    # Case 2: 2x4 registered wrapper core
    c2_t0 = time.time()
    w2 = np.array([[12, -7, 0, 15], [-3, 8, 22, -1]], dtype=np.int8)
    op2 = LinearOp(name="reg_2x4", in_features=4, out_features=2, weight=w2)
    manifest2, _ = tile_weight_matrix(w2, tile_n=2, tile_k=2)
    x2 = np.array([[5, -10, 15, -20], [-128, 127, -128, 127], [1, 2, 3, 4]], dtype=np.int8)
    y2_exp = linear_reference_int64(x2, w2)
    c2_dir = tmp_dir / "l0_c2"
    c2_dir.mkdir(parents=True, exist_ok=True)
    x2_hex, y2_hex = write_hex_vectors(x2, y2_exp, c2_dir)
    dut2_file = c2_dir / "reg_core.v"
    dut2_file.write_text(generate_full_module(op2, manifest2, latency=1), encoding="utf-8")
    tb2_file = c2_dir / "tb_reg.v"
    tb2_file.write_text(
        generate_wrapper_testbench(
            op2,
            num_vectors=len(x2),
            x_hex_path=str(x2_hex),
            y_hex_path=str(y2_hex),
            module_name="llmhw_reg_2x4",
            manifest=manifest2,
            latency=1,
        ),
        encoding="utf-8",
    )
    res2 = run_simulation(dut2_file, tb2_file, expected_checked=len(x2), work_dir=c2_dir)
    case2 = TestCaseResult(
        name="reg_2x4_wrapper",
        status=res2.status,
        checked=res2.checked,
        mismatches=res2.mismatch,
        timing_errors=res2.timing_err,
        duration_seconds=time.time() - c2_t0,
    )
    suite.cases.append(case2)

    suite.duration_seconds = time.time() - t0
    suite.tests_run = len(suite.cases)
    suite.passed = sum(1 for c in suite.cases if c.status == "PASS")
    suite.failed = suite.tests_run - suite.passed
    suite.vectors_checked = sum(c.checked for c in suite.cases)
    if suite.failed > 0:
        suite.status = "FAIL"
    return suite


# -----------------------------------------------------------------------------
# Suite L1: Synthetic Matrix
# -----------------------------------------------------------------------------
def run_suite_l1(tmp_dir: Path, fast_mode: bool = False) -> TestSuiteResult:
    """Level L1: Synthetic shapes across dimensions, tiles, weight and input classes."""
    suite = TestSuiteResult(
        name="L1_synthetic",
        description="Level L1 synthetic dimensions, tile grids, weight and input classes",
    )
    t0 = time.time()
    configs = get_l1_test_matrix(fast_mode=fast_mode)

    for cfg in configs:
        c_t0 = time.time()
        c_dir = tmp_dir / f"l1_{cfg.name}"
        c_dir.mkdir(parents=True, exist_ok=True)
        try:
            report, _ = run_matrix_case(cfg, c_dir, num_vectors=5)
            case = TestCaseResult(
                name=cfg.name,
                status=report.status,
                checked=report.checked,
                mismatches=report.mismatch,
                timing_errors=report.timing_err,
                duration_seconds=time.time() - c_t0,
            )
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as e:
            case = TestCaseResult(
                name=cfg.name,
                status="FAIL",
                duration_seconds=time.time() - c_t0,
                details=str(e),
            )
        suite.cases.append(case)

    suite.duration_seconds = time.time() - t0
    suite.tests_run = len(suite.cases)
    suite.passed = sum(1 for c in suite.cases if c.status == "PASS")
    suite.failed = suite.tests_run - suite.passed
    suite.vectors_checked = sum(c.checked for c in suite.cases)
    if suite.failed > 0:
        suite.status = "FAIL"
    return suite


# -----------------------------------------------------------------------------
# Suite L2: Real Model Tiles
# -----------------------------------------------------------------------------
def run_suite_l2(tmp_dir: Path, model_dir: Path) -> TestSuiteResult:
    """Level L2: 8x16 real model tile slices across all 7 projection types at 3 depths."""
    suite = TestSuiteResult(
        name="L2_real_tiles",
        description="Level L2 8x16 real model tiles across 7 projection types and 3 depths",
    )
    t0 = time.time()
    layers = [0, 14, 29]
    projections = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    for layer in layers:
        for proj in projections:
            c_t0 = time.time()
            case_name = f"L{layer}_{proj}"
            p_dir = model_dir / f"layer_{layer}" / proj
            if not p_dir.exists():
                suite.cases.append(
                    TestCaseResult(
                        name=case_name,
                        status="FAIL",
                        details=f"Missing artifact dir {p_dir}",
                    )
                )
                continue

            w_float = np.load(p_dir / "weight.npy")
            x_float = np.load(p_dir / "calib" / "x_sample.npy")
            w_q, _ = quantize_weights(w_float)
            stat = compute_activation_stat(x_float, method="absmax")
            scale = compute_activation_scale(stat)
            x_q = quantize_activations(x_float, scale)

            slice_n, slice_k = 8, 16
            num_vectors = 5
            w_slice = w_q[:slice_n, :slice_k].copy()
            x_stim = x_q[:num_vectors, :slice_k].copy()
            y_exp = linear_reference_int64(x_stim, w_slice)

            op_name = f"l2_L{layer}_{proj}"
            op = LinearOp(
                name=op_name,
                in_features=slice_k,
                out_features=slice_n,
                weight=w_slice,
                is_quantized=True,
            )
            manifest, _ = tile_weight_matrix(w_slice, tile_n=slice_n, tile_k=slice_k)

            c_dir = tmp_dir / op_name
            c_dir.mkdir(parents=True, exist_ok=True)
            x_hex, y_hex = write_hex_vectors(x_stim, y_exp, c_dir)
            dut_file = c_dir / f"{op_name}.v"
            dut_file.write_text(generate_full_module(op, manifest, latency=1), encoding="utf-8")
            tb_file = c_dir / f"tb_{op_name}.v"
            tb_file.write_text(
                generate_wrapper_testbench(
                    op=op,
                    num_vectors=num_vectors,
                    x_hex_path=str(x_hex),
                    y_hex_path=str(y_hex),
                    module_name=f"llmhw_{op_name}",
                    manifest=manifest,
                    latency=1,
                ),
                encoding="utf-8",
            )
            report = run_simulation(
                dut_path=dut_file,
                tb_path=tb_file,
                expected_checked=num_vectors,
                work_dir=c_dir,
            )
            suite.cases.append(
                TestCaseResult(
                    name=case_name,
                    status=report.status,
                    checked=report.checked,
                    mismatches=report.mismatch,
                    timing_errors=report.timing_err,
                    duration_seconds=time.time() - c_t0,
                )
            )

    suite.duration_seconds = time.time() - t0
    suite.tests_run = len(suite.cases)
    suite.passed = sum(1 for c in suite.cases if c.status == "PASS")
    suite.failed = suite.tests_run - suite.passed
    suite.vectors_checked = sum(c.checked for c in suite.cases)
    if suite.failed > 0:
        suite.status = "FAIL"
    return suite


# -----------------------------------------------------------------------------
# Suite L3: Real Multi-Tile Slices
# -----------------------------------------------------------------------------
def run_suite_l3(tmp_dir: Path, model_dir: Path) -> TestSuiteResult:
    """Level L3: 16x32 multi-tile slices (2x2 grid) with real weights and activations."""
    suite = TestSuiteResult(
        name="L3_real_multi_tile",
        description="Level L3 16x32 multi-tile slices (2x2 grid) across 3 depths",
    )
    t0 = time.time()
    targets = [(0, "q_proj"), (14, "gate_proj"), (29, "down_proj")]

    for layer, proj in targets:
        c_t0 = time.time()
        case_name = f"L{layer}_{proj}_multi"
        p_dir = model_dir / f"layer_{layer}" / proj
        if not p_dir.exists():
            suite.cases.append(
                TestCaseResult(
                    name=case_name,
                    status="FAIL",
                    details=f"Missing artifact dir {p_dir}",
                )
            )
            continue

        w_float = np.load(p_dir / "weight.npy")
        x_float = np.load(p_dir / "calib" / "x_sample.npy")
        w_q, _ = quantize_weights(w_float)
        stat = compute_activation_stat(x_float, method="absmax")
        scale = compute_activation_scale(stat)
        x_q = quantize_activations(x_float, scale)

        slice_n, slice_k = 16, 32
        tile_n, tile_k = 8, 16
        num_vectors = 5
        w_slice = w_q[:slice_n, :slice_k].copy()
        x_stim = x_q[:num_vectors, :slice_k].copy()
        y_exp = linear_reference_int64(x_stim, w_slice)

        op_name = f"l3_L{layer}_{proj}"
        op = LinearOp(
            name=op_name,
            in_features=slice_k,
            out_features=slice_n,
            weight=w_slice,
            is_quantized=True,
        )
        manifest, _ = tile_weight_matrix(w_slice, tile_n=tile_n, tile_k=tile_k)

        c_dir = tmp_dir / op_name
        c_dir.mkdir(parents=True, exist_ok=True)
        x_hex, y_hex = write_hex_vectors(x_stim, y_exp, c_dir)
        dut_file = c_dir / f"{op_name}.v"
        dut_file.write_text(generate_full_module(op, manifest, latency=1), encoding="utf-8")
        tb_file = c_dir / f"tb_{op_name}.v"
        tb_file.write_text(
            generate_wrapper_testbench(
                op=op,
                num_vectors=num_vectors,
                x_hex_path=str(x_hex),
                y_hex_path=str(y_hex),
                module_name=f"llmhw_{op_name}",
                manifest=manifest,
                latency=1,
            ),
            encoding="utf-8",
        )
        report = run_simulation(
            dut_path=dut_file,
            tb_path=tb_file,
            expected_checked=num_vectors,
            work_dir=c_dir,
        )
        suite.cases.append(
            TestCaseResult(
                name=case_name,
                status=report.status,
                checked=report.checked,
                mismatches=report.mismatch,
                timing_errors=report.timing_err,
                duration_seconds=time.time() - c_t0,
            )
        )

    suite.duration_seconds = time.time() - t0
    suite.tests_run = len(suite.cases)
    suite.passed = sum(1 for c in suite.cases if c.status == "PASS")
    suite.failed = suite.tests_run - suite.passed
    suite.vectors_checked = sum(c.checked for c in suite.cases)
    if suite.failed > 0:
        suite.status = "FAIL"
    return suite


# -----------------------------------------------------------------------------
# Suite Cross-Sim: Verilator vs Icarus Equivalence
# -----------------------------------------------------------------------------
def run_suite_cross_sim(tmp_dir: Path) -> TestSuiteResult:
    """Cross-Simulator verification: Verilator lint, Icarus vs Verilator bit-exact equality."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suite = TestSuiteResult(
        name="Cross_Simulator",
        description="Verilator lint clean and Icarus vs Verilator cross-simulator equality",
    )
    t0 = time.time()

    # 1. Verilator lint check on Design F, Design P, and Design P_bus
    lint_t0 = time.time()
    w_lint = np.array([[12, -7, 0, 15], [-3, 8, 22, -1], [5, -12, 3, 0]], dtype=np.int8)
    b_lint = np.array([100, -200, 50], dtype=np.int32)
    op_lint = LinearOp(name="lint_test", in_features=4, out_features=3, weight=w_lint, bias=b_lint)
    m_lint, _ = tile_weight_matrix(w_lint, tile_n=4, tile_k=4)

    f_v = tmp_dir / "dut_f.v"
    f_v.write_text(generate_full_module(op_lint, m_lint), encoding="utf-8")
    p_v = tmp_dir / "dut_p.v"
    p_v.write_text(generate_full_module_p(op_lint, m_lint), encoding="utf-8")
    p_bus_v = tmp_dir / "dut_p_bus.v"
    p_bus_v.write_text(generate_full_module_p_bus(op_lint, m_lint), encoding="utf-8")

    res_f = lint_verilator(f_v)
    res_p = lint_verilator(p_v)
    res_p_bus = lint_verilator(p_bus_v)

    lint_pass = (res_f.returncode == 0) and (res_p.returncode == 0) and (res_p_bus.returncode == 0)
    suite.cases.append(
        TestCaseResult(
            name="verilator_lint_wall_clean",
            status="PASS" if lint_pass else "FAIL",
            duration_seconds=time.time() - lint_t0,
            details="Design F, P, P_bus pass verilator --lint-only -Wall",
        )
    )

    # 2. Cross-simulator equality across representative geometries
    shapes = [
        ("square_4x4", 4, 4, 2, 2),
        ("non_multiple_3x5", 3, 5, 2, 4),
    ]

    for name, n, k, tile_n, tile_k in shapes:
        c_t0 = time.time()
        rng = np.random.default_rng(101)
        w = rng.integers(-127, 128, size=(n, k), dtype=np.int8)
        op = LinearOp(name=f"xsim_{name}", in_features=k, out_features=n, weight=w)
        manifest, _ = tile_weight_matrix(w, tile_n=tile_n, tile_k=tile_k)

        x = generate_inputs(k, w, "class1_random_int8", num_vectors=5, seed=42)
        y_exp = linear_reference_int64(x, w)

        if k != manifest.k_padded:
            x_stim, _ = pad_activations(x, manifest.tile_k)
        else:
            x_stim = x

        c_dir = tmp_dir / f"xsim_{name}"
        c_dir.mkdir(parents=True, exist_ok=True)
        x_hex, y_hex = write_hex_vectors(x_stim, y_exp, c_dir)
        dut_file = c_dir / f"dut_{name}.v"
        dut_file.write_text(generate_full_module(op, manifest, latency=1), encoding="utf-8")
        tb_file = c_dir / f"tb_{name}.v"
        tb_file.write_text(
            generate_wrapper_testbench(
                op=op,
                num_vectors=len(x),
                x_hex_path=str(x_hex),
                y_hex_path=str(y_hex),
                module_name=f"llmhw_xsim_{name}",
                manifest=manifest,
                latency=1,
            ),
            encoding="utf-8",
        )

        res_iv = run_simulation(dut_file, tb_file, expected_checked=len(x), work_dir=c_dir)
        res_vl = run_verilator_simulation(
            dut_file, tb_file, expected_checked=len(x), work_dir=c_dir
        )

        match = (
            res_iv.status == "PASS"
            and res_vl.status == "PASS"
            and res_iv.checked == len(x)
            and res_vl.checked == len(x)
            and res_iv.mismatch == 0
            and res_vl.mismatch == 0
            and res_iv.timing_err == 0
            and res_vl.timing_err == 0
        )

        suite.cases.append(
            TestCaseResult(
                name=f"cross_sim_{name}",
                status="PASS" if match else "FAIL",
                checked=len(x),
                mismatches=res_iv.mismatch + res_vl.mismatch,
                timing_errors=res_iv.timing_err + res_vl.timing_err,
                duration_seconds=time.time() - c_t0,
                details=f"iverilog={res_iv.status}, verilator={res_vl.status}",
            )
        )

    # 3. Negative control on Verilator
    neg_t0 = time.time()
    neg_dir = tmp_dir / "verilator_neg_control"
    neg_dir.mkdir(parents=True, exist_ok=True)
    c_base = tmp_dir / "xsim_square_4x4"
    corrupt_y = neg_dir / "y_corrupt.hex"
    with open(c_base / "y_exp.hex") as f_in, open(corrupt_y, "w") as f_out:
        lines = f_in.readlines()
        lines[0] = "ffffffff\n"
        f_out.writelines(lines)
    neg_tb = neg_dir / "tb_neg.v"
    neg_tb.write_text(
        (c_base / "tb_square_4x4.v").read_text().replace("y_exp.hex", str(corrupt_y)),
        encoding="utf-8",
    )
    res_neg = run_verilator_simulation(
        c_base / "dut_square_4x4.v", neg_tb, expected_checked=5, work_dir=neg_dir
    )
    neg_pass = (res_neg.status == "FAIL") and (res_neg.mismatch > 0)
    suite.cases.append(
        TestCaseResult(
            name="verilator_negative_control",
            status="PASS" if neg_pass else "FAIL",
            duration_seconds=time.time() - neg_t0,
            details=f"Negative control correctly failed (mismatches={res_neg.mismatch})",
        )
    )

    suite.duration_seconds = time.time() - t0
    suite.tests_run = len(suite.cases)
    suite.passed = sum(1 for c in suite.cases if c.status == "PASS")
    suite.failed = suite.tests_run - suite.passed
    suite.vectors_checked = sum(c.checked for c in suite.cases)
    if suite.failed > 0:
        suite.status = "FAIL"
    return suite


# -----------------------------------------------------------------------------
# Suite Weight-Audit: Static RTL Weight Parser and Integrity
# -----------------------------------------------------------------------------
def run_suite_weight_audit(tmp_dir: Path, model_dir: Path | None) -> TestSuiteResult:
    """Static weight audit: parses constants from RTL and proves exact equality to W_q."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suite = TestSuiteResult(
        name="Weight_Audit",
        description="Static RTL constant parser vs W_q, manifest hash, and negative controls",
    )
    t0 = time.time()

    # 1. Toy core audit
    c1_t0 = time.time()
    w1 = np.array([[10, -5, 1]], dtype=np.int8)
    op1 = LinearOp(name="toy_audit", in_features=3, out_features=1, weight=w1)
    rtl1 = generate_fixed_core(op1)
    rep1 = audit_weights_match(rtl1, w1, strict=True)
    suite.cases.append(
        TestCaseResult(
            name="audit_toy_1x3",
            status="PASS" if rep1["passed"] else "FAIL",
            checked=rep1["total_weights_checked"],
            duration_seconds=time.time() - c1_t0,
            details="Bit-exact match on 1x3 toy core",
        )
    )

    # 2. Multi-tile synthetic audit
    c2_t0 = time.time()
    rng = np.random.default_rng(42)
    w2 = rng.integers(-127, 128, size=(4, 8), dtype=np.int8)
    w2[0, 0] = 0
    w2[0, 1] = 1
    w2[0, 2] = -1
    op2 = LinearOp(name="multi_audit", in_features=8, out_features=4, weight=w2)
    manifest2, _ = tile_weight_matrix(w2, tile_n=2, tile_k=4)
    rtl2 = generate_full_module(op2, manifest2)
    rep2 = audit_weights_match(rtl2, w2, manifest=manifest2, strict=True)
    suite.cases.append(
        TestCaseResult(
            name="audit_multi_tile_4x8",
            status="PASS" if rep2["passed"] else "FAIL",
            checked=rep2["total_weights_checked"],
            duration_seconds=time.time() - c2_t0,
            details="Bit-exact match and manifest SHA256 verified",
        )
    )

    # 3. Real layer tile audit
    if model_dir is not None:
        c3_t0 = time.time()
        p_dir = model_dir / "layer_0" / "q_proj"
        if p_dir.exists():
            w_float = np.load(p_dir / "weight.npy")
            w_q, _ = quantize_weights(w_float)
            w_slice = w_q[:8, :16].copy()
            op3 = LinearOp(name="real_audit", in_features=16, out_features=8, weight=w_slice)
            manifest3, _ = tile_weight_matrix(w_slice, tile_n=8, tile_k=16)
            rtl3 = generate_full_module(op3, manifest3)
            rep3 = audit_weights_match(rtl3, w_slice, manifest=manifest3, strict=True)
            suite.cases.append(
                TestCaseResult(
                    name="audit_real_layer0_q_proj_tile",
                    status="PASS" if rep3["passed"] else "FAIL",
                    checked=rep3["total_weights_checked"],
                    duration_seconds=time.time() - c3_t0,
                    details="Real layer slice matched bit-exact with SHA256 integrity",
                )
            )

    # 4. Negative controls: mutated constant, flipped sign, dropped term
    neg_t0 = time.time()
    neg_passed = True
    # Mutated constant
    mut_rtl = rtl1.replace("8'sd10", "8'sd11")
    if audit_weights_match(mut_rtl, w1, strict=False)["passed"]:
        neg_passed = False
    # Flipped sign
    mut_rtl2 = rtl1.replace("-8'sd5", "8'sd5")
    if audit_weights_match(mut_rtl2, w1, strict=False)["passed"]:
        neg_passed = False
    # Dropped term
    mut_rtl3 = "\n".join([line for line in rtl1.splitlines() if not ("prod_0_0" in line and "wire" in line)])
    if audit_weights_match(mut_rtl3, w1, strict=False)["passed"]:
        neg_passed = False

    suite.cases.append(
        TestCaseResult(
            name="audit_negative_controls",
            status="PASS" if neg_passed else "FAIL",
            duration_seconds=time.time() - neg_t0,
            details="100% of injected weight corruptions detected by auditor",
        )
    )

    suite.duration_seconds = time.time() - t0
    suite.tests_run = len(suite.cases)
    suite.passed = sum(1 for c in suite.cases if c.status == "PASS")
    suite.failed = suite.tests_run - suite.passed
    suite.vectors_checked = sum(c.checked for c in suite.cases)
    if suite.failed > 0:
        suite.status = "FAIL"
    return suite


# -----------------------------------------------------------------------------
# Suite Mutation: M1–M9 Fault Injection Campaign
# -----------------------------------------------------------------------------
def run_suite_mutation(tmp_dir: Path) -> TestSuiteResult:
    """Full mutation testing campaign across M1–M9 asserting 100% kill rate."""
    suite = TestSuiteResult(
        name="Mutation_Campaign",
        description="M1-M9 fault injection campaign asserting 100% kill rate (0 survivors)",
    )
    t0 = time.time()

    w = np.array([[15, -25, 35, -45], [10, -20, 30, -40]], dtype=np.int8)
    op = LinearOp(name="mut_runner", in_features=4, out_features=2, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=2)

    rng = np.random.default_rng(42)
    x_rand = rng.integers(-128, 128, size=(5, 4), dtype=np.int8)
    x_one_hot = np.eye(4, dtype=np.int8)
    x_ext = np.array(
        [
            [-128, -128, -128, -128],
            [127, 127, 127, 127],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    x_worst = np.array(
        [
            [127, -128, 127, -128],
            [-128, 127, -128, 127],
        ],
        dtype=np.int8,
    )
    x_all = np.vstack([x_rand, x_one_hot, x_ext, x_worst])
    y_exp = linear_reference_int64(x_all, w)

    c_dir = tmp_dir / "mut_campaign"
    c_dir.mkdir(parents=True, exist_ok=True)
    x_hex, y_hex = write_hex_vectors(x_all, y_exp, c_dir)
    rtl = generate_full_module(op, manifest, latency=1)
    dut_file = c_dir / "dut.v"
    dut_file.write_text(rtl, encoding="utf-8")
    tb_file = c_dir / "tb.v"
    tb_file.write_text(
        generate_wrapper_testbench(
            op=op,
            num_vectors=len(x_all),
            x_hex_path=str(x_hex),
            y_hex_path=str(y_hex),
            module_name="llmhw_mut_runner",
            manifest=manifest,
            latency=1,
        ),
        encoding="utf-8",
    )

    campaign = run_mutation_campaign(
        dut_path=dut_file,
        tb_path=tb_file,
        num_vectors=len(x_all),
        work_dir=c_dir / "campaign",
        rtl_code=rtl,
    )

    for entry in campaign["mutants"]:
        case_name = f"{entry['type']}_{entry['id']}"
        suite.cases.append(
            TestCaseResult(
                name=case_name,
                status="PASS" if entry["killed"] else "FAIL",
                checked=len(x_all),
                mismatches=entry["mismatch_count"],
                timing_errors=entry["timing_err_count"],
                details=f"{entry['description']} (status: {entry['status']})",
            )
        )

    suite.duration_seconds = time.time() - t0
    suite.tests_run = len(suite.cases)
    suite.passed = sum(1 for c in suite.cases if c.status == "PASS")
    suite.failed = suite.tests_run - suite.passed
    suite.vectors_checked = len(x_all)
    if suite.failed > 0:
        suite.status = "FAIL"
    return suite


# -----------------------------------------------------------------------------
# Main Regression Runner Orchestrator
# -----------------------------------------------------------------------------
def run_regression(
    repo_root: Path,
    output_path: Path,
    fast_mode: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Execute all regression test suites, collect structured metrics, and write JSON."""
    wall_t0 = time.time()
    git_info = get_git_info(repo_root)
    tool_versions = get_tool_versions(repo_root)
    model_dir = find_model_artifact_dir(repo_root)

    if verbose:
        print("=" * 80)
        print("  LLM-HARDWARE COMPILER — COMPREHENSIVE VERIFICATION REGRESSION")
        print("=" * 80)
        print(f"Git commit : {git_info['commit']} (branch: {git_info['branch']})")
        print(f"Timestamp  : {datetime.now(timezone.utc).isoformat()}")
        print(f"Platform   : {platform.platform()}")
        print(f"Tools      : iverilog={tool_versions['iverilog']}")
        print(f"             verilator={tool_versions['verilator']}")
        print(f"Artifacts  : {model_dir if model_dir else 'None'}")
        print("-" * 80)

    suites: list[TestSuiteResult] = []

    with tempfile.TemporaryDirectory(prefix="reg_") as td:
        tmp_dir = Path(td)

        # 1. Suite L0
        if verbose:
            print("[1/7] Running Suite L0 (Toy combinational and registered cores)...", end=" ", flush=True)
        s0 = run_suite_l0(tmp_dir / "s0")
        suites.append(s0)
        if verbose:
            print(f"{s0.status} ({s0.passed}/{s0.tests_run} passed, {s0.duration_seconds:.2f}s)")

        # 2. Suite L1
        if verbose:
            print(f"[2/7] Running Suite L1 (Synthetic matrix, fast={fast_mode})...", end=" ", flush=True)
        s1 = run_suite_l1(tmp_dir / "s1", fast_mode=fast_mode)
        suites.append(s1)
        if verbose:
            print(f"{s1.status} ({s1.passed}/{s1.tests_run} passed, {s1.duration_seconds:.2f}s)")

        # 3. Suite L2
        if verbose:
            print("[3/7] Running Suite L2 (Real layer 8x16 tiles across 21 configs)...", end=" ", flush=True)
        if model_dir is not None:
            s2 = run_suite_l2(tmp_dir / "s2", model_dir)
        else:
            s2 = TestSuiteResult(name="L2_real_tiles", description="Artifacts missing", status="FAIL")
        suites.append(s2)
        if verbose:
            print(f"{s2.status} ({s2.passed}/{s2.tests_run} passed, {s2.duration_seconds:.2f}s)")

        # 4. Suite L3
        if verbose:
            print("[4/7] Running Suite L3 (Real multi-tile slices across 3 depths)...", end=" ", flush=True)
        if model_dir is not None:
            s3 = run_suite_l3(tmp_dir / "s3", model_dir)
        else:
            s3 = TestSuiteResult(name="L3_real_multi_tile", description="Artifacts missing", status="FAIL")
        suites.append(s3)
        if verbose:
            print(f"{s3.status} ({s3.passed}/{s3.tests_run} passed, {s3.duration_seconds:.2f}s)")

        # 5. Suite Cross-Sim
        if verbose:
            print("[5/7] Running Suite Cross-Sim (Verilator lint & Icarus vs Verilator)...", end=" ", flush=True)
        s4 = run_suite_cross_sim(tmp_dir / "s4")
        suites.append(s4)
        if verbose:
            print(f"{s4.status} ({s4.passed}/{s4.tests_run} passed, {s4.duration_seconds:.2f}s)")

        # 6. Suite Weight-Audit
        if verbose:
            print("[6/7] Running Suite Weight-Audit (Static RTL constant parser)...", end=" ", flush=True)
        s5 = run_suite_weight_audit(tmp_dir / "s5", model_dir)
        suites.append(s5)
        if verbose:
            print(f"{s5.status} ({s5.passed}/{s5.tests_run} passed, {s5.duration_seconds:.2f}s)")

        # 7. Suite Mutation
        if verbose:
            print("[7/7] Running Suite Mutation (M1-M9 fault injection campaign)...", end=" ", flush=True)
        s6 = run_suite_mutation(tmp_dir / "s6")
        suites.append(s6)
        if verbose:
            print(f"{s6.status} ({s6.passed}/{s6.tests_run} passed, {s6.duration_seconds:.2f}s)")

    total_wall_time = time.time() - wall_t0
    total_tests = sum(s.tests_run for s in suites)
    total_passed = sum(s.passed for s in suites)
    total_failed = sum(s.failed for s in suites)
    total_vectors = sum(s.vectors_checked for s in suites)
    overall_status = "PASS" if total_failed == 0 else "FAIL"

    report_data: dict[str, Any] = {
        "metadata": {
            "git_commit": git_info["commit"],
            "git_branch": git_info["branch"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "tools": tool_versions,
            "fast_mode": fast_mode,
        },
        "summary": {
            "status": overall_status,
            "total_suites": len(suites),
            "total_tests": total_tests,
            "passed_tests": total_passed,
            "failed_tests": total_failed,
            "total_vectors_checked": total_vectors,
            "wall_time_seconds": round(total_wall_time, 2),
        },
        "suites": [asdict(s) for s in suites],
    }

    # Write results/regression.json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    if verbose:
        print("=" * 80)
        print("REGRESSION SUMMARY:")
        print(f"{'Suite Name':<24} | {'Status':<6} | {'Tests':<8} | {'Passed':<8} | {'Duration (s)':<12}")
        print("-" * 80)
        for s in suites:
            print(
                f"{s.name:<24} | {s.status:<6} | {s.tests_run:<8} | {s.passed:<8} | {s.duration_seconds:<12.2f}"
            )
        print("-" * 80)
        print(
            f"OVERALL STATUS: {overall_status} ({total_passed}/{total_tests} passed in {total_wall_time:.2f}s, {total_vectors} vectors checked)"
        )
        print(f"Results recorded at: {output_path}")
        print("=" * 80)

    return report_data


def main() -> int:
    """CLI entry point for regression test runner."""
    parser = argparse.ArgumentParser(description="LLM-Hardware Comprehensive Regression Runner")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results/regression.json"),
        help="Destination path for regression results JSON (default: results/regression.json)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast mode subset for quick sanity checking",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose console printing",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    report = run_regression(
        repo_root=repo_root,
        output_path=args.output,
        fast_mode=args.quick,
        verbose=not args.quiet,
    )
    return 0 if report["summary"]["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
