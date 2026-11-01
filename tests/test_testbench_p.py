"""Tests for Design P testbench generation and simulation (T5.2).

Verifies:
- `write_weight_hex_vectors` format (row-major, 2 hex digits per INT8)
- `generate_wrapper_testbench_p` compilation and simulation in Icarus
- Bit-exact simulation match on toy module
- Bit-exact simulation match on real SmolLM2-135M tile
- Negative control: corrupted w.hex produces simulation mismatch
"""

from pathlib import Path

import numpy as np
import pytest

from backend.baseline_p import generate_full_module_p
from backend.testbench import generate_wrapper_testbench_p
from backend.tiling import tile_weight_matrix
from backend.vectors import write_hex_vectors, write_weight_hex_vectors
from ir.linear import LinearOp
from verification.reference import linear_reference_int64
from verification.simulator import run_simulation


def test_write_weight_hex_vectors_format(tmp_path: Path):
    """Verify w.hex generation format: 2 hex digits per int8 in two's complement."""
    w = np.array([[0, 127], [-128, -1]], dtype=np.int8)
    w_hex = write_weight_hex_vectors(w, tmp_path)

    assert w_hex.exists()
    lines = [line.strip() for line in w_hex.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 4
    # 0 -> 00, 127 -> 7f, -128 -> 80, -1 -> ff
    assert lines == ["00", "7f", "80", "ff"]


def test_design_p_toy_simulation(tmp_path: Path):
    """Verify simulation of Design P on a 2x2 toy module."""
    w = np.array([[3, -2], [5, 1]], dtype=np.int8)
    op = LinearOp(name="toy_p", in_features=2, out_features=2, weight=w)
    manifest, _tiles = tile_weight_matrix(w, tile_n=2, tile_k=2)

    # Input activations: 5 test vectors
    x = np.array(
        [
            [10, -4],
            [-1, 2],
            [-128, 127],
            [0, 0],
            [15, -30],
        ],
        dtype=np.int8,
    )

    y_exp = linear_reference_int64(x, w)

    # Write vectors
    x_hex, y_hex = write_hex_vectors(x, y_exp, tmp_path)
    w_hex = write_weight_hex_vectors(w, tmp_path)

    # Generate DUT and TB
    dut_code = generate_full_module_p(op, manifest)
    dut_file = tmp_path / "dut_p.v"
    dut_file.write_text(dut_code, encoding="utf-8")

    tb_code = generate_wrapper_testbench_p(
        op=op,
        manifest=manifest,
        num_vectors=len(x),
        w_hex_path=str(w_hex),
        x_hex_path=str(x_hex),
        y_hex_path=str(y_hex),
    )
    tb_file = tmp_path / "tb_p.v"
    tb_file.write_text(tb_code, encoding="utf-8")

    # Run simulation
    report = run_simulation(dut_file, tb_file, expected_checked=5, work_dir=tmp_path)
    assert report.status == "PASS", f"Simulation failed:\n{report.stdout}\n{report.stderr}"
    assert report.checked == 5
    assert report.mismatch == 0
    assert report.timing_err == 0


def test_design_p_real_smollm_tile_simulation(tmp_path: Path):
    """Verify simulation of Design P on real SmolLM2 Layer 0 q_proj 8x16 tile."""
    artifact_path = Path("artifacts/SmolLM2-135M@93efa2f097d5/layer_0/q_proj/weight.npy")
    if not artifact_path.exists():
        pytest.skip(f"Artifact {artifact_path} not found")

    w_full = np.load(artifact_path)
    from compiler.quantization import quantize_weights

    w_q, _s_w = quantize_weights(w_full)

    # Extract 8x16 tile
    w_tile = w_q[0:8, 0:16]
    op = LinearOp(name="smollm_tile_p", in_features=16, out_features=8, weight=w_tile)
    manifest, _tiles = tile_weight_matrix(w_tile, tile_n=8, tile_k=16)

    # Generate 10 test vectors
    rng = np.random.default_rng(42)
    x = rng.integers(-128, 128, size=(10, 16), dtype=np.int8)
    y_exp = linear_reference_int64(x, w_tile)

    x_hex, y_hex = write_hex_vectors(x, y_exp, tmp_path)
    w_hex = write_weight_hex_vectors(w_tile, tmp_path)

    dut_code = generate_full_module_p(op, manifest)
    dut_file = tmp_path / "dut_smollm_p.v"
    dut_file.write_text(dut_code, encoding="utf-8")

    tb_code = generate_wrapper_testbench_p(
        op=op,
        manifest=manifest,
        num_vectors=len(x),
        w_hex_path=str(w_hex),
        x_hex_path=str(x_hex),
        y_hex_path=str(y_hex),
    )
    tb_file = tmp_path / "tb_smollm_p.v"
    tb_file.write_text(tb_code, encoding="utf-8")

    report = run_simulation(dut_file, tb_file, expected_checked=10, work_dir=tmp_path)
    assert report.status == "PASS", f"Simulation failed:\n{report.stdout}\n{report.stderr}"
    assert report.checked == 10
    assert report.mismatch == 0
    assert report.timing_err == 0


def test_design_p_negative_control_corrupted_weights(tmp_path: Path):
    """Negative control: corrupted w.hex causes simulation mismatch."""
    w = np.array([[3, -2], [5, 1]], dtype=np.int8)
    op = LinearOp(name="toy_neg_p", in_features=2, out_features=2, weight=w)
    manifest, _tiles = tile_weight_matrix(w, tile_n=2, tile_k=2)

    x = np.array([[10, -4]], dtype=np.int8)
    y_exp = linear_reference_int64(x, w)

    x_hex, y_hex = write_hex_vectors(x, y_exp, tmp_path)
    w_hex = write_weight_hex_vectors(w, tmp_path)

    # Corrupt first weight in w.hex
    w_hex.write_text("7f\nfe\n05\n01\n", encoding="utf-8")

    dut_code = generate_full_module_p(op, manifest)
    dut_file = tmp_path / "dut_neg_p.v"
    dut_file.write_text(dut_code, encoding="utf-8")

    tb_code = generate_wrapper_testbench_p(
        op=op,
        manifest=manifest,
        num_vectors=len(x),
        w_hex_path=str(w_hex),
        x_hex_path=str(x_hex),
        y_hex_path=str(y_hex),
    )
    tb_file = tmp_path / "tb_neg_p.v"
    tb_file.write_text(tb_code, encoding="utf-8")

    report = run_simulation(dut_file, tb_file, expected_checked=1, work_dir=tmp_path)
    assert report.status == "FAIL" or report.mismatch > 0
