"""Tests for Sequencer Schedule & Microprogram Generator (Task UB.5)."""

import json
from pathlib import Path

from compiler.format_assignment import assign_formats_and_scales
from compiler.rom_packing import pack_rom_artifacts
from compiler.schedule import Opcode, estimate_op_cycles, generate_schedule
from frontend.importer_v2 import build_graph_v2_from_artifacts, load_model_artifacts_v2

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_model_v2"
CONFIG_FROZEN_PATH = REPO_ROOT / "numerics" / "config_frozen.json"


def test_microprogram_generation_and_cycle_estimation(tmp_path: Path):
    """Verify microprogram generation, compact layer-loop structure, and cycle estimates."""
    artifacts = load_model_artifacts_v2(TINY_ARTIFACTS_DIR)
    graph = build_graph_v2_from_artifacts(TINY_ARTIFACTS_DIR)

    with open(CONFIG_FROZEN_PATH, "r", encoding="utf-8") as f:
        frozen_config = json.load(f)

    fmt_res = assign_formats_and_scales(
        graph=graph,
        weights_dict=artifacts["tensors"],
        frozen_config=frozen_config,
    )

    rom_dir = tmp_path / "roms"
    addr_map = pack_rom_artifacts(
        graph=graph,
        format_result=fmt_res,
        weights_dict=artifacts["tensors"],
        out_dir=rom_dir,
        tile_r=8,
        tile_c=16,
        max_ctx=32,
        head_dim=16,
    )

    prog = generate_schedule(
        address_map=addr_map,
        config=artifacts["config"],
        tile_r=8,
        tile_c=16,
    )

    # 1. Compact program: < 25 instructions covers the entire model decoder
    assert len(prog.instructions) < 25
    assert prog.layer_loop_start < prog.layer_loop_end
    assert prog.num_layers == 2
    assert len(prog.layer_rom_bases) == 2

    # 2. Cycles estimate is strictly positive and non-zero
    assert prog.total_cycles_per_token > 0
    for ins in prog.instructions:
        assert ins.cycle_estimate > 0

    # 3. Hex word serialization
    for ins in prog.instructions:
        hex_w = ins.to_hex_word()
        assert len(hex_w) == 32  # 128 bits = 32 hex chars

    # 4. Save and reload
    prog_dir = tmp_path / "prog"
    prog.save(prog_dir)
    assert (prog_dir / "schedule.json").exists()
    assert (prog_dir / "program.hex").exists()

    with open(prog_dir / "schedule.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["num_layers"] == 2
    assert len(data["instructions"]) == len(prog.instructions)


def test_estimate_op_cycles_formulas():
    """Verify cycle estimate calculations across opcodes."""
    c_lin = estimate_op_cycles(Opcode.LINEAR, [64, 64], tile_r=8, tile_c=16)
    # 64/8 = 8, 64/16 = 4 -> 8*4*8 = 256 + 8 = 264
    assert c_lin == 264

    c_norm = estimate_op_cycles(Opcode.RMSNORM, [64])
    # 2*64 + 12 = 140
    assert c_norm == 140

    c_halt = estimate_op_cycles(Opcode.HALT, [0])
    assert c_halt == 1
