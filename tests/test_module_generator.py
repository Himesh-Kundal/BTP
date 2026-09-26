"""Tests for full module generator, adder trees, registered wrapper, and budgets (T4.5)."""

import subprocess
import time
from pathlib import Path

import numpy as np

from backend.module_generator import generate_full_module
from backend.testbench import generate_wrapper_testbench
from backend.tiling import tile_weight_matrix
from compiler.quantization import quantize_weights
from ir.linear import LinearOp
from verification.reference import linear_reference_int64
from verification.simulator import run_simulation
from verification.vectors import write_hex_vectors


def test_module_generator_multi_tile_structure(tmp_path: Path):
    """Verify tile instantiation, reduction adder trees over j, and wrapper for 4x8 matrix."""
    rng = np.random.default_rng(123)
    # N=4, K=8 with tile (2, 4) -> 2 tiles in N, 2 tiles in K -> 4 tiles total
    w = rng.integers(-100, 100, size=(4, 8), dtype=np.int8)
    b = np.array([5, -10, 15, -20], dtype=np.int32)

    op = LinearOp(name="multi_tile_4x8", in_features=8, out_features=4, weight=w, bias=b, is_quantized=True)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=4, layer_name=op.name)

    rtl = generate_full_module(op, manifest, tile_n=2, tile_k=4, latency=1)

    # Check tile core instantiations
    assert "u_tile_0_0" in rtl
    assert "u_tile_0_1" in rtl
    assert "u_tile_1_0" in rtl
    assert "u_tile_1_1" in rtl

    # Check adder tree reduction over j
    assert "y_row_0" in rtl
    assert "term_0_0 + term_0_1" in rtl
    # Bias is added
    assert "32'sd5" in rtl

    # Check registered wrapper
    assert "module llmhw_multi_tile_4x8 (" in rtl
    assert "input  wire                    clk," in rtl
    assert "input  wire                    rst_n," in rtl
    assert "input  wire                    valid_in," in rtl
    assert "output reg                     valid_out," in rtl
    assert "output reg  [127:0] y_flat" in rtl

    # Lint check
    v_file = tmp_path / "top_4x8.v"
    v_file.write_text(rtl)

    cmd_iv = ["iverilog", "-g2005", "-Wall", "-t", "null", str(v_file)]
    res_iv = subprocess.run(cmd_iv, check=False, capture_output=True, text=True)
    assert res_iv.returncode == 0, f"iverilog failed:\n{res_iv.stderr}"

    cmd_vl = ["verilator", "--lint-only", "-Wall", str(v_file)]
    res_vl = subprocess.run(cmd_vl, check=False, capture_output=True, text=True)
    assert res_vl.returncode == 0, f"verilator failed:\n{res_vl.stderr}"


def test_module_generator_real_smollm_tile_simulation(tmp_path: Path):
    """Simulate a real 8x16 tile extracted from SmolLM2 Layer 0 Q-proj with registered wrapper."""
    repo_root = Path(__file__).resolve().parent.parent
    weight_path = repo_root / "artifacts" / "smollm2-135m@93efa2f" / "layer_0" / "q_proj" / "weight.npy"

    if weight_path.exists():
        raw_w = np.load(weight_path)
        sub_w_float = raw_w[:8, :16]
        sub_w_q, _ = quantize_weights(sub_w_float)
    else:
        rng = np.random.default_rng(42)
        sub_w_q = rng.integers(-127, 127, size=(8, 16), dtype=np.int8)

    op = LinearOp(
        name="smollm_tile_8x16",
        in_features=16,
        out_features=8,
        weight=sub_w_q,
        is_quantized=True,
    )
    manifest, _ = tile_weight_matrix(sub_w_q, tile_n=8, tile_k=16, layer_name=op.name)

    rtl = generate_full_module(op, manifest, tile_n=8, tile_k=16, latency=1)
    dut_path = tmp_path / "dut.v"
    dut_path.write_text(rtl)

    # Create random test vectors (10 vectors)
    rng = np.random.default_rng(2026)
    x = rng.integers(-128, 128, size=(10, 16), dtype=np.int8)
    y_exp = linear_reference_int64(x, sub_w_q).astype(np.int32)

    x_hex = tmp_path / "x.hex"
    y_hex = tmp_path / "y_exp.hex"
    write_hex_vectors(x, y_exp, x_hex, y_hex)

    tb_code = generate_wrapper_testbench(
        op=op,
        num_vectors=10,
        x_hex_path=str(x_hex),
        y_hex_path=str(y_hex),
        module_name=f"llmhw_{op.name}",
        latency=1,
    )
    tb_path = tmp_path / "tb.v"
    tb_path.write_text(tb_code)

    # Run simulation
    report = run_simulation(dut_path, tb_path, run_dir=tmp_path)
    assert report.passed, f"Simulation failed: {report.error_msg}"
    assert report.checked == 10
    assert report.mismatch == 0
    assert report.timing_err == 0


def test_module_generator_full_576x576_layer_benchmark(tmp_path: Path):
    """Benchmark full 576x576 layer generation time and file size, and verify iverilog compilation."""
    rng = np.random.default_rng(999)
    w = rng.integers(-127, 127, size=(576, 576), dtype=np.int8)

    op = LinearOp(
        name="layer0_q_proj_full",
        in_features=576,
        out_features=576,
        weight=w,
        is_quantized=True,
    )
    manifest, _ = tile_weight_matrix(w, tile_n=8, tile_k=16, layer_name=op.name)

    t0 = time.perf_counter()
    rtl = generate_full_module(op, manifest, tile_n=8, tile_k=16, latency=1)
    gen_time = time.perf_counter() - t0

    out_file = tmp_path / "full_576x576.v"
    out_file.write_text(rtl)
    file_size_mb = out_file.stat().st_size / (1024 * 1024)

    # Budget assertions
    assert gen_time < 60.0, f"Generation too slow: {gen_time:.2f}s"
    assert file_size_mb < 50.0, f"File size too large: {file_size_mb:.2f}MB"

    # Icarus syntax check
    cmd = ["iverilog", "-g2005", "-Wall", "-t", "null", str(out_file)]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert res.returncode == 0, f"iverilog failed on full layer:\n{res.stderr}"
