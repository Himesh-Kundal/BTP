"""Real-layer verification tests across all 7 projection types and 3 depths (T6.2).

Verifies Claim C1 (bit-exact match against independent reference oracle) on real
SmolLM2-135M weights and real captured activations:
1. Level L2: 8x16 tile slice for all 7 projection types at layers 0, 14, 29 (21 configs).
2. Level L3: 16x32 multi-tile slice across layers 0, 14, 29 verifying inter-tile adder trees.
"""

from pathlib import Path

import numpy as np
import pytest

from backend.module_generator import generate_full_module
from backend.testbench import generate_wrapper_testbench
from backend.tiling import tile_weight_matrix
from backend.vectors import write_hex_vectors
from compiler.calibration import (
    compute_activation_scale,
    compute_activation_stat,
    quantize_activations,
)
from compiler.quantization import quantize_weights
from ir.linear import LinearOp
from verification.reference import linear_reference_int64
from verification.simulator import run_simulation

LAYERS = [0, 14, 29]
PROJECTIONS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

CONFIGS_21 = [(layer, proj) for layer in LAYERS for proj in PROJECTIONS]


def _get_artifact_dir(layer: int, proj: str) -> Path:
    """Resolve path to real model weight and activation artifacts."""
    repo_root = Path(__file__).resolve().parent.parent
    primary = repo_root / "artifacts" / "SmolLM2-135M@93efa2f097d5" / f"layer_{layer}" / proj
    if primary.exists():
        return primary
    fallback = repo_root / "artifacts" / "smollm2-135m@93efa2f" / f"layer_{layer}" / proj
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not locate artifacts for layer {layer} {proj}")


def _load_and_quantize_slice(
    layer: int,
    proj: str,
    slice_n: int,
    slice_k: int,
    num_vectors: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Load real float weights and captured activations, quantize to INT8, and extract slice."""
    art_dir = _get_artifact_dir(layer, proj)
    w_float = np.load(art_dir / "weight.npy")
    x_float = np.load(art_dir / "calib" / "x_sample.npy")

    # Quantize weights per Rule N2
    w_q, _ = quantize_weights(w_float)

    # Calibrate and quantize activations per Rule N3
    stat = compute_activation_stat(x_float, method="absmax")
    scale = compute_activation_scale(stat)
    x_q = quantize_activations(x_float, scale)

    # Slice weights and activations
    w_slice = w_q[:slice_n, :slice_k].copy()
    x_stim = x_q[:num_vectors, :slice_k].copy()

    return w_slice, x_stim


@pytest.mark.parametrize(
    ("layer", "proj"),
    CONFIGS_21,
    ids=[f"L{l}_{p}" for l, p in CONFIGS_21],
)
def test_real_layer_l2_tile_slices(layer: int, proj: str, tmp_path: Path) -> None:
    """Verify bit-exact Claim C1 on Level L2 (8x16 tile) real slices across all 21 configurations."""
    slice_n, slice_k = 8, 16
    num_vectors = 5
    w_slice, x_stim = _load_and_quantize_slice(
        layer=layer,
        proj=proj,
        slice_n=slice_n,
        slice_k=slice_k,
        num_vectors=num_vectors,
    )

    y_exp = linear_reference_int64(x_stim, w_slice)

    op_name = f"l2_L{layer}_{proj}"
    op = LinearOp(
        name=op_name,
        in_features=slice_k,
        out_features=slice_n,
        weight=w_slice,
        is_quantized=True,
    )
    manifest, _ = tile_weight_matrix(w_slice, tile_n=slice_n, tile_k=slice_k, layer_name=op.name)

    work_dir = tmp_path / op_name
    work_dir.mkdir(parents=True, exist_ok=True)

    x_hex, y_hex = write_hex_vectors(x_stim, y_exp, work_dir)

    rtl = generate_full_module(op, manifest, latency=1)
    dut_file = work_dir / f"{op_name}.v"
    dut_file.write_text(rtl, encoding="utf-8")

    tb = generate_wrapper_testbench(
        op=op,
        num_vectors=num_vectors,
        x_hex_path=str(x_hex),
        y_hex_path=str(y_hex),
        module_name=f"llmhw_{op_name}",
        manifest=manifest,
        latency=1,
    )
    tb_file = work_dir / f"tb_{op_name}.v"
    tb_file.write_text(tb, encoding="utf-8")

    report = run_simulation(
        dut_path=dut_file,
        tb_path=tb_file,
        expected_checked=num_vectors,
        work_dir=work_dir,
    )

    assert report.status == "PASS", (
        f"L2 tile failed on L{layer}_{proj}:\n{report.stdout}\n{report.stderr}"
    )
    assert report.checked == num_vectors
    assert report.mismatch == 0
    assert report.timing_err == 0


@pytest.mark.parametrize(
    ("layer", "proj"),
    [
        (0, "q_proj"),
        (14, "gate_proj"),
        (29, "down_proj"),
    ],
    ids=["L0_q_proj_multi", "L14_gate_proj_multi", "L29_down_proj_multi"],
)
def test_real_layer_l3_multi_tile_slices(layer: int, proj: str, tmp_path: Path) -> None:
    """Verify bit-exact Claim C1 on Level L3 (16x32 multi-tile: 2x2 grid) real slices."""
    slice_n, slice_k = 16, 32
    tile_n, tile_k = 8, 16
    num_vectors = 5
    w_slice, x_stim = _load_and_quantize_slice(
        layer=layer,
        proj=proj,
        slice_n=slice_n,
        slice_k=slice_k,
        num_vectors=num_vectors,
    )

    y_exp = linear_reference_int64(x_stim, w_slice)

    op_name = f"l3_L{layer}_{proj}"
    op = LinearOp(
        name=op_name,
        in_features=slice_k,
        out_features=slice_n,
        weight=w_slice,
        is_quantized=True,
    )
    manifest, _ = tile_weight_matrix(w_slice, tile_n=tile_n, tile_k=tile_k, layer_name=op.name)

    work_dir = tmp_path / op_name
    work_dir.mkdir(parents=True, exist_ok=True)

    x_hex, y_hex = write_hex_vectors(x_stim, y_exp, work_dir)

    rtl = generate_full_module(op, manifest, latency=1)
    dut_file = work_dir / f"{op_name}.v"
    dut_file.write_text(rtl, encoding="utf-8")

    tb = generate_wrapper_testbench(
        op=op,
        num_vectors=num_vectors,
        x_hex_path=str(x_hex),
        y_hex_path=str(y_hex),
        module_name=f"llmhw_{op_name}",
        manifest=manifest,
        latency=1,
    )
    tb_file = work_dir / f"tb_{op_name}.v"
    tb_file.write_text(tb, encoding="utf-8")

    report = run_simulation(
        dut_path=dut_file,
        tb_path=tb_file,
        expected_checked=num_vectors,
        work_dir=work_dir,
    )

    assert report.status == "PASS", (
        f"L3 multi-tile failed on L{layer}_{proj}:\n{report.stdout}\n{report.stderr}"
    )
    assert report.checked == num_vectors
    assert report.mismatch == 0
    assert report.timing_err == 0
