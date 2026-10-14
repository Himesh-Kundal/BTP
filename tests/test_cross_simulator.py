"""Cross-simulator verification and Verilator runner tests (T6.3).

Verifies:
1. Verilator lint-only clean status on generated Design F, Design P, and Design P_bus.
2. Verilator compilation and execution via run_verilator_simulation.
3. Cross-simulator equality: Y_iverilog == Y_verilator == Y_ref_int64 across
   representative shapes including non-multiple padded shapes.
4. Real SmolLM2 layer slice execution on both simulators.
5. Negative control: corrupted vectors trigger FAIL in Verilator.
"""

from pathlib import Path

import numpy as np
import pytest

from backend.baseline_p import generate_full_module_p, generate_full_module_p_bus
from backend.module_generator import generate_full_module
from backend.testbench import generate_wrapper_testbench
from backend.tiling import pad_activations, tile_weight_matrix
from backend.vectors import write_hex_vectors
from compiler.calibration import (
    compute_activation_scale,
    compute_activation_stat,
    quantize_activations,
)
from compiler.quantization import quantize_weights
from ir.linear import LinearOp
from verification.reference import linear_reference_int64
from verification.simulator import lint_verilator, run_simulation, run_verilator_simulation


def test_verilator_lint_clean_designs(tmp_path: Path) -> None:
    """Verify that generated Design F, P, and P_bus pass verilator --lint-only -Wall cleanly."""
    w = np.array([[12, -7, 0, 15], [-3, 8, 22, -1], [5, -12, 3, 0]], dtype=np.int8)
    b = np.array([100, -200, 50], dtype=np.int32)
    n, k = w.shape
    op = LinearOp(name="lint_test", in_features=k, out_features=n, weight=w, bias=b)
    manifest, _ = tile_weight_matrix(w, tile_n=4, tile_k=4)

    # 1. Design F
    rtl_f = generate_full_module(op, manifest)
    f_file = tmp_path / "dut_f.v"
    f_file.write_text(rtl_f, encoding="utf-8")
    res_f = lint_verilator(f_file)
    assert res_f.returncode == 0, f"Verilator lint failed on Design F:\n{res_f.stderr}"

    # 2. Design P
    rtl_p = generate_full_module_p(op, manifest)
    p_file = tmp_path / "dut_p.v"
    p_file.write_text(rtl_p, encoding="utf-8")
    res_p = lint_verilator(p_file)
    assert res_p.returncode == 0, f"Verilator lint failed on Design P:\n{res_p.stderr}"

    # 3. Design P_bus
    rtl_p_bus = generate_full_module_p_bus(op, manifest)
    p_bus_file = tmp_path / "dut_p_bus.v"
    p_bus_file.write_text(rtl_p_bus, encoding="utf-8")
    res_p_bus = lint_verilator(p_bus_file)
    assert res_p_bus.returncode == 0, f"Verilator lint failed on Design P_bus:\n{res_p_bus.stderr}"


def test_verilator_simulation_toy(tmp_path: Path) -> None:
    """Verify run_verilator_simulation on 1x3 toy shape."""
    w = np.array([[2, -3, 1]], dtype=np.int8)
    x = np.array([[10, 20, 30], [-5, 0, 5], [127, -128, 0]], dtype=np.int8)
    y_exp = linear_reference_int64(x, w)

    op = LinearOp(name="toy_verilator", in_features=3, out_features=1, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=1, tile_k=3)

    work_dir = tmp_path / "toy"
    work_dir.mkdir(parents=True, exist_ok=True)
    x_hex, y_hex = write_hex_vectors(x, y_exp, work_dir)

    rtl = generate_full_module(op, manifest)
    dut_file = work_dir / f"{op.name}.v"
    dut_file.write_text(rtl, encoding="utf-8")

    tb = generate_wrapper_testbench(
        op=op,
        num_vectors=3,
        x_hex_path=str(x_hex),
        y_hex_path=str(y_hex),
        module_name=f"llmhw_{op.name}",
        manifest=manifest,
    )
    tb_file = work_dir / f"tb_{op.name}.v"
    tb_file.write_text(tb, encoding="utf-8")

    report = run_verilator_simulation(
        dut_path=dut_file,
        tb_path=tb_file,
        expected_checked=3,
        work_dir=work_dir,
    )

    assert report.status == "PASS", (
        f"Verilator toy simulation failed:\n{report.stdout}\n{report.stderr}"
    )
    assert report.checked == 3
    assert report.mismatch == 0
    assert report.timing_err == 0


@pytest.mark.parametrize(
    ("shape_name", "n", "k", "tile_n", "tile_k"),
    [
        ("square_4x4", 4, 4, 4, 4),
        ("rect_8x16", 8, 16, 4, 8),
        ("non_mult_5x11", 5, 11, 4, 4),
    ],
)
def test_cross_simulator_equality(
    shape_name: str,
    n: int,
    k: int,
    tile_n: int,
    tile_k: int,
    tmp_path: Path,
) -> None:
    """Verify Y_iverilog == Y_verilator == Y_ref across multiple geometries."""
    rng = np.random.default_rng(42)
    w = rng.integers(-127, 128, size=(n, k), dtype=np.int8)
    num_vectors = 5
    x = rng.integers(-128, 128, size=(num_vectors, k), dtype=np.int8)
    y_exp = linear_reference_int64(x, w)

    op_name = f"cross_{shape_name}"
    op = LinearOp(name=op_name, in_features=k, out_features=n, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=tile_n, tile_k=tile_k)

    if k != manifest.k_padded:
        x_stim, _ = pad_activations(x, manifest.tile_k)
    else:
        x_stim = x

    work_dir = tmp_path / shape_name
    work_dir.mkdir(parents=True, exist_ok=True)
    x_hex, y_hex = write_hex_vectors(x_stim, y_exp, work_dir)

    rtl = generate_full_module(op, manifest)
    dut_file = work_dir / f"{op_name}.v"
    dut_file.write_text(rtl, encoding="utf-8")

    tb = generate_wrapper_testbench(
        op=op,
        num_vectors=num_vectors,
        x_hex_path=str(x_hex),
        y_hex_path=str(y_hex),
        module_name=f"llmhw_{op_name}",
        manifest=manifest,
    )
    tb_file = work_dir / f"tb_{op_name}.v"
    tb_file.write_text(tb, encoding="utf-8")

    # Run Icarus Verilog
    iv_dir = work_dir / "iverilog"
    rep_iv = run_simulation(
        dut_path=dut_file,
        tb_path=tb_file,
        expected_checked=num_vectors,
        work_dir=iv_dir,
    )
    assert rep_iv.status == "PASS", (
        f"Icarus failed on {shape_name}:\n{rep_iv.stdout}\n{rep_iv.stderr}"
    )

    # Run Verilator
    vl_dir = work_dir / "verilator"
    rep_vl = run_verilator_simulation(
        dut_path=dut_file,
        tb_path=tb_file,
        expected_checked=num_vectors,
        work_dir=vl_dir,
    )
    assert rep_vl.status == "PASS", (
        f"Verilator failed on {shape_name}:\n{rep_vl.stdout}\n{rep_vl.stderr}"
    )

    # Verify cross-simulator agreement
    assert rep_iv.checked == rep_vl.checked == num_vectors
    assert rep_iv.mismatch == rep_vl.mismatch == 0
    assert rep_iv.timing_err == rep_vl.timing_err == 0


def test_cross_simulator_real_smollm_slice(tmp_path: Path) -> None:
    """Verify cross-simulator equality on real SmolLM2 Layer 0 q_proj slice."""
    repo_root = Path(__file__).resolve().parent.parent
    art_dir = repo_root / "artifacts" / "SmolLM2-135M@93efa2f097d5" / "layer_0" / "q_proj"
    w_float = np.load(art_dir / "weight.npy")
    x_float = np.load(art_dir / "calib" / "x_sample.npy")

    w_q, _ = quantize_weights(w_float)
    stat = compute_activation_stat(x_float, method="absmax")
    scale = compute_activation_scale(stat)
    x_q = quantize_activations(x_float, scale)

    # 8x16 real slice
    w_slice = w_q[:8, :16].copy()
    x_stim = x_q[:5, :16].copy()
    y_exp = linear_reference_int64(x_stim, w_slice)

    op_name = "cross_smollm_q"
    op = LinearOp(name=op_name, in_features=16, out_features=8, weight=w_slice)
    manifest, _ = tile_weight_matrix(w_slice, tile_n=8, tile_k=16)

    work_dir = tmp_path / "smollm_q"
    work_dir.mkdir(parents=True, exist_ok=True)
    x_hex, y_hex = write_hex_vectors(x_stim, y_exp, work_dir)

    rtl = generate_full_module(op, manifest)
    dut_file = work_dir / f"{op_name}.v"
    dut_file.write_text(rtl, encoding="utf-8")

    tb = generate_wrapper_testbench(
        op=op,
        num_vectors=5,
        x_hex_path=str(x_hex),
        y_hex_path=str(y_hex),
        module_name=f"llmhw_{op_name}",
        manifest=manifest,
    )
    tb_file = work_dir / f"tb_{op_name}.v"
    tb_file.write_text(tb, encoding="utf-8")

    rep_iv = run_simulation(dut_file, tb_file, expected_checked=5, work_dir=work_dir / "iv")
    rep_vl = run_verilator_simulation(
        dut_file, tb_file, expected_checked=5, work_dir=work_dir / "vl"
    )

    assert rep_iv.status == "PASS"
    assert rep_vl.status == "PASS"
    assert rep_iv.checked == rep_vl.checked == 5
    assert rep_iv.mismatch == rep_vl.mismatch == 0


def test_verilator_negative_control_corrupted_vectors(tmp_path: Path) -> None:
    """Verify that corrupting expected vectors causes run_verilator_simulation to FAIL."""
    w = np.array([[10, -5], [3, 12]], dtype=np.int8)
    x = np.array([[1, 2], [-3, 4]], dtype=np.int8)
    y_exp = linear_reference_int64(x, w)
    # Corrupt expected output
    y_exp_bad = y_exp.copy()
    y_exp_bad[0, 0] += 99

    op = LinearOp(name="bad_v", in_features=2, out_features=2, weight=w)
    manifest, _ = tile_weight_matrix(w, 2, 2)

    work_dir = tmp_path / "bad"
    work_dir.mkdir(parents=True, exist_ok=True)
    x_hex, y_hex = write_hex_vectors(x, y_exp_bad, work_dir)

    rtl = generate_full_module(op, manifest)
    dut_file = work_dir / f"{op.name}.v"
    dut_file.write_text(rtl, encoding="utf-8")

    tb = generate_wrapper_testbench(
        op=op,
        num_vectors=2,
        x_hex_path=str(x_hex),
        y_hex_path=str(y_hex),
        module_name=f"llmhw_{op.name}",
        manifest=manifest,
    )
    tb_file = work_dir / f"tb_{op.name}.v"
    tb_file.write_text(tb, encoding="utf-8")

    report = run_verilator_simulation(
        dut_path=dut_file,
        tb_path=tb_file,
        expected_checked=2,
        work_dir=work_dir,
    )

    assert report.status == "FAIL", "Corrupted vector was not caught by Verilator!"
    assert report.mismatch > 0
