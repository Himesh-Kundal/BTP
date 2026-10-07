"""Differential verification proving Y_F == Y_P == Y_ref (T5.3).

Tests 5 distinct geometric and model configurations:
1. Toy shape (1x3, tile 1x3)
2. Symmetric square multiple shape (8x8, tile 4x4)
3. Rectangular multiple shape (16x32, tile 8x16)
4. Non-multiple shape (5x11, tile 4x4 with zero padding)
5. Real SmolLM2-135M Layer 0 q_proj slice (8x16 tile)

For each shape, verifies:
- Design F matches reference oracle bit-for-bit
- Design P matches reference oracle bit-for-bit
- Therefore Y_F == Y_P == Y_ref with zero bit errors
- Negative control: perturbing weights in P produces mismatch
"""

from pathlib import Path

import numpy as np
import pytest

from backend.baseline_p import generate_full_module_p
from backend.module_generator import generate_full_module
from backend.testbench import generate_wrapper_testbench, generate_wrapper_testbench_p
from backend.tiling import pad_activations, pad_matrix, tile_weight_matrix
from backend.vectors import write_hex_vectors, write_weight_hex_vectors
from ir.linear import LinearOp
from verification.reference import linear_reference_int64
from verification.simulator import run_simulation


def _run_differential_shape(
    shape_id: str,
    w: np.ndarray,
    tile_n: int,
    tile_k: int,
    tmp_path: Path,
    num_vectors: int = 5,
    seed: int = 42,
) -> None:
    """Execute differential verification on both Design F and Design P."""
    work_dir = tmp_path / shape_id
    work_dir.mkdir(parents=True, exist_ok=True)

    n, k = w.shape
    op = LinearOp(name=f"diff_{shape_id}", in_features=k, out_features=n, weight=w)
    manifest, _tiles = tile_weight_matrix(w, tile_n=tile_n, tile_k=tile_k)

    rng = np.random.default_rng(seed)
    x = rng.integers(-128, 128, size=(num_vectors, k), dtype=np.int8)
    y_exp = linear_reference_int64(x, w)

    # Pad inputs if non-multiple
    if k != manifest.k_padded:
        x_stim, _ = pad_activations(x, manifest.tile_k)
    else:
        x_stim = x

    # Pad weights for Design P loading if non-multiple
    if w.shape != (manifest.n_padded, manifest.k_padded):
        w_pad, _, _ = pad_matrix(w, manifest.tile_n, manifest.tile_k)
    else:
        w_pad = w

    x_hex, y_hex = write_hex_vectors(x_stim, y_exp, work_dir)
    w_hex = write_weight_hex_vectors(w_pad, work_dir)

    # --- 1. Design F ---
    f_dir = work_dir / "design_f"
    f_dir.mkdir(parents=True, exist_ok=True)
    f_rtl = generate_full_module(op, manifest)
    f_dut = f_dir / "dut_f.v"
    f_dut.write_text(f_rtl, encoding="utf-8")

    f_tb = generate_wrapper_testbench(
        op=op,
        num_vectors=num_vectors,
        x_hex_path=str(x_hex),
        y_hex_path=str(y_hex),
        module_name=f"llmhw_{op.name}",
        manifest=manifest,
    )
    f_tb_file = f_dir / "tb_f.v"
    f_tb_file.write_text(f_tb, encoding="utf-8")

    report_f = run_simulation(f_dut, f_tb_file, expected_checked=num_vectors, work_dir=f_dir)
    assert report_f.status == "PASS", f"Design F failed:\n{report_f.stdout}\n{report_f.stderr}"
    assert report_f.checked == num_vectors
    assert report_f.mismatch == 0
    assert report_f.timing_err == 0

    # --- 2. Design P ---
    p_dir = work_dir / "design_p"
    p_dir.mkdir(parents=True, exist_ok=True)
    p_rtl = generate_full_module_p(op, manifest)
    p_dut = p_dir / "dut_p.v"
    p_dut.write_text(p_rtl, encoding="utf-8")

    p_tb = generate_wrapper_testbench_p(
        op=op,
        manifest=manifest,
        num_vectors=num_vectors,
        w_hex_path=str(w_hex),
        x_hex_path=str(x_hex),
        y_hex_path=str(y_hex),
        module_name=f"llmhw_{op.name}_p",
    )
    p_tb_file = p_dir / "tb_p.v"
    p_tb_file.write_text(p_tb, encoding="utf-8")

    report_p = run_simulation(p_dut, p_tb_file, expected_checked=num_vectors, work_dir=p_dir)
    assert report_p.status == "PASS", f"Design P failed:\n{report_p.stdout}\n{report_p.stderr}"
    assert report_p.checked == num_vectors
    assert report_p.mismatch == 0
    assert report_p.timing_err == 0


def test_differential_shape1_toy(tmp_path: Path):
    """Shape 1: Toy shape (1x3, tile 1x3)."""
    w = np.array([[3, -2, 7]], dtype=np.int8)
    _run_differential_shape("s1_toy_1x3", w, tile_n=1, tile_k=3, tmp_path=tmp_path)


def test_differential_shape2_square_multiple(tmp_path: Path):
    """Shape 2: Symmetric square multiple shape (8x8, tile 4x4)."""
    rng = np.random.default_rng(101)
    w = rng.integers(-127, 128, size=(8, 8), dtype=np.int8)
    _run_differential_shape("s2_square_8x8", w, tile_n=4, tile_k=4, tmp_path=tmp_path)


def test_differential_shape3_rectangular_multiple(tmp_path: Path):
    """Shape 3: Rectangular multiple shape (16x32, tile 8x16)."""
    rng = np.random.default_rng(202)
    w = rng.integers(-127, 128, size=(16, 32), dtype=np.int8)
    _run_differential_shape("s3_rect_16x32", w, tile_n=8, tile_k=16, tmp_path=tmp_path)


def test_differential_shape4_non_multiple_padded(tmp_path: Path):
    """Shape 4: Non-multiple shape with padding (5x11, tile 4x4 -> 8x12)."""
    rng = np.random.default_rng(303)
    w = rng.integers(-127, 128, size=(5, 11), dtype=np.int8)
    _run_differential_shape("s4_non_mult_5x11", w, tile_n=4, tile_k=4, tmp_path=tmp_path)


def test_differential_shape5_real_smollm_tile(tmp_path: Path):
    """Shape 5: Real SmolLM2 Layer 0 q_proj slice (8x16 tile)."""
    artifact_path = Path("artifacts/SmolLM2-135M@93efa2f097d5/layer_0/q_proj/weight.npy")
    if not artifact_path.exists():
        pytest.skip(f"Artifact {artifact_path} not found")

    w_full = np.load(artifact_path)
    from compiler.quantization import quantize_weights

    w_q, _ = quantize_weights(w_full)
    w_slice = w_q[0:8, 0:16]
    _run_differential_shape("s5_smollm_8x16", w_slice, tile_n=8, tile_k=16, tmp_path=tmp_path)


def test_differential_negative_control(tmp_path: Path):
    """Negative control: corrupted weights in P must cause differential mismatch."""
    w = np.array([[3, -2], [5, 1]], dtype=np.int8)
    op = LinearOp(name="diff_neg", in_features=2, out_features=2, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=2)

    x = np.array([[10, -4]], dtype=np.int8)
    y_exp = linear_reference_int64(x, w)

    x_hex, y_hex = write_hex_vectors(x, y_exp, tmp_path)
    w_hex = write_weight_hex_vectors(w, tmp_path)

    # Invert the weight file
    w_hex.write_text("00\n00\n00\n00\n", encoding="utf-8")

    p_rtl = generate_full_module_p(op, manifest)
    p_dut = tmp_path / "dut_neg_p.v"
    p_dut.write_text(p_rtl, encoding="utf-8")

    p_tb = generate_wrapper_testbench_p(
        op=op,
        manifest=manifest,
        num_vectors=1,
        w_hex_path=str(w_hex),
        x_hex_path=str(x_hex),
        y_hex_path=str(y_hex),
        module_name=f"llmhw_{op.name}_p",
    )
    p_tb_file = tmp_path / "tb_neg_p.v"
    p_tb_file.write_text(p_tb, encoding="utf-8")

    report_p = run_simulation(p_dut, p_tb_file, expected_checked=1, work_dir=tmp_path)
    assert report_p.status == "FAIL" or report_p.mismatch > 0
