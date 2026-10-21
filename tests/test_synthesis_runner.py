"""Tests for Yosys synthesis flow runner and metric parser (T7.1).

Verifies:
1. Generic gate synthesis of Design F with cell count and flop extraction.
2. iCE40 FPGA LUT-4 cell mapping and LUT4 counting.
3. Comparative synthesis of Design F, Design P, and Design P_bus.
4. Determinism of synthesis metrics across identical reruns.
"""

from pathlib import Path

import numpy as np

from backend.tiling import tile_weight_matrix
from ir.linear import LinearOp
from synth.yosys_runner import synthesize_design


def test_yosys_generic_synthesis_toy(tmp_path: Path) -> None:
    """Verify generic gate synthesis and metric extraction on Design F."""
    w = np.array([[12, -7, 0, 15], [-3, 8, 22, -1]], dtype=np.int8)
    op = LinearOp(name="synth_toy_f", in_features=4, out_features=2, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=2)

    res = synthesize_design(
        op=op,
        manifest=manifest,
        design_type="F",
        target="generic",
        work_dir=tmp_path / "generic_f",
    )

    assert res.returncode == 0, f"Synthesis failed:\n{res.stderr}"
    assert res.num_cells_total > 0
    assert res.num_cells_comb > 0
    assert res.num_cells_seq > 0  # Registered wrapper flip-flops
    assert res.ltp_depth > 0
    assert res.target == "generic"


def test_yosys_ice40_synthesis_toy(tmp_path: Path) -> None:
    """Verify iCE40 LUT-4 synthesis and proxy metric extraction on Design F."""
    w = np.array([[12, -7, 0, 15], [-3, 8, 22, -1]], dtype=np.int8)
    op = LinearOp(name="synth_toy_ice40", in_features=4, out_features=2, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=2)

    res = synthesize_design(
        op=op,
        manifest=manifest,
        design_type="F",
        target="ice40",
        work_dir=tmp_path / "ice40_f",
    )

    assert res.returncode == 0, f"iCE40 synthesis failed:\n{res.stderr}"
    assert res.num_lut4 > 0
    assert res.num_cells_seq > 0
    assert res.ltp_depth > 0
    assert res.target == "ice40"


def test_yosys_synthesis_design_p_and_p_bus(tmp_path: Path) -> None:
    """Verify comparative synthesis across Design F, Design P, and Design P_bus."""
    w = np.array([[15, -25, 35, -45], [10, -20, 30, -40]], dtype=np.int8)
    op = LinearOp(name="synth_comp", in_features=4, out_features=2, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=2)

    # 1. Design F (Fixed constant weights)
    res_f = synthesize_design(
        op=op, manifest=manifest, design_type="F", target="generic", work_dir=tmp_path / "f"
    )

    # 2. Design P (Programmable baseline with W_reg storage)
    res_p = synthesize_design(
        op=op, manifest=manifest, design_type="P", target="generic", work_dir=tmp_path / "p"
    )

    # 3. Design P_bus (Logic-only baseline with primary-input weight bus)
    res_p_bus = synthesize_design(
        op=op, manifest=manifest, design_type="P_bus", target="generic", work_dir=tmp_path / "p_bus"
    )

    assert res_f.returncode == 0
    assert res_p.returncode == 0
    assert res_p_bus.returncode == 0

    # Sequential cells: P has W_reg array (8*4*2 = 64 bits) + output registers, so strictly more flops than P_bus and F
    assert res_p.num_cells_seq > res_p_bus.num_cells_seq
    # Both F and P_bus have only output registers (F prunes provably unreachable upper bits)
    assert res_f.num_cells_seq <= 65  # 64 data + 1 valid
    assert res_p_bus.num_cells_seq <= 65

    # Fixed weights yield significantly smaller combinational and total area
    assert res_f.num_cells_comb < res_p_bus.num_cells_comb
    assert res_f.num_cells_total < res_p_bus.num_cells_total
    assert res_f.num_cells_total < res_p.num_cells_total


def test_synthesis_determinism(tmp_path: Path) -> None:
    """Verify bit-for-bit identical synthesis metrics across repeated invocations."""
    w = np.array([[10, -20], [30, -40]], dtype=np.int8)
    op = LinearOp(name="synth_det", in_features=2, out_features=2, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=2)

    res1 = synthesize_design(
        op=op, manifest=manifest, design_type="F", target="generic", work_dir=tmp_path / "run1"
    )
    res2 = synthesize_design(
        op=op, manifest=manifest, design_type="F", target="generic", work_dir=tmp_path / "run2"
    )

    assert res1.returncode == 0
    assert res2.returncode == 0
    assert res1.num_cells_total == res2.num_cells_total
    assert res1.num_cells_comb == res2.num_cells_comb
    assert res1.num_cells_seq == res2.num_cells_seq
    assert res1.ltp_depth == res2.ltp_depth
    assert res1.cell_counts == res2.cell_counts
