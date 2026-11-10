"""Tests for Compiled-Artifact Software Interpreter (Task UB.7)."""

import json
from pathlib import Path

from compiler.format_assignment import assign_formats_and_scales
from compiler.interpreter import CompiledArtifactInterpreter
from compiler.rom_packing import pack_rom_artifacts
from compiler.schedule import generate_schedule
from frontend.importer_v2 import build_graph_v2_from_artifacts, load_model_artifacts_v2

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_model_v2"
CONFIG_FROZEN_PATH = REPO_ROOT / "numerics" / "config_frozen.json"


def test_compiled_artifact_interpreter_execution(tmp_path: Path):
    """Test compiled-artifact interpreter execution on tiny model ROM images."""
    artifacts = load_model_artifacts_v2(TINY_ARTIFACTS_DIR)
    graph = build_graph_v2_from_artifacts(TINY_ARTIFACTS_DIR)

    with open(CONFIG_FROZEN_PATH, "r", encoding="utf-8") as f:
        frozen_config = json.load(f)

    fmt_res = assign_formats_and_scales(
        graph=graph,
        weights_dict=artifacts["tensors"],
        frozen_config=frozen_config,
    )

    rom_dir = tmp_path / "compiled_roms"
    tile_r, tile_c = 8, 16

    addr_map = pack_rom_artifacts(
        graph=graph,
        format_result=fmt_res,
        weights_dict=artifacts["tensors"],
        out_dir=rom_dir,
        tile_r=tile_r,
        tile_c=tile_c,
        max_ctx=32,
        head_dim=16,
    )

    prog = generate_schedule(
        address_map=addr_map,
        config=artifacts["config"],
        tile_r=tile_r,
        tile_c=tile_c,
    )
    prog.save(rom_dir)
    fmt_res.save_json(rom_dir / "format_table.json")

    # Instantiate Interpreter
    interp = CompiledArtifactInterpreter(rom_dir)

    # Decode first step
    prompt_tokens = [12, 45, 78, 101]
    interp.reset()

    generated_tokens: list[int] = []
    for pos, tok in enumerate(prompt_tokens):
        logits, next_tok = interp.decode_step(tok, pos=pos)
        assert logits.shape == (artifacts["config"]["vocab_size"],)
        assert 0 <= next_tok < artifacts["config"]["vocab_size"]
        generated_tokens.append(next_tok)

    # Rerun determinism check
    interp.reset()
    rerun_tokens: list[int] = []
    for pos, tok in enumerate(prompt_tokens):
        _, next_tok = interp.decode_step(tok, pos=pos)
        rerun_tokens.append(next_tok)

    assert generated_tokens == rerun_tokens, "Interpreter execution must be strictly deterministic"


def test_compiled_artifact_interpreter_trace_capture(tmp_path: Path):
    """Test that interpreter captures intermediate traces for every operation."""
    artifacts = load_model_artifacts_v2(TINY_ARTIFACTS_DIR)
    graph = build_graph_v2_from_artifacts(TINY_ARTIFACTS_DIR)

    with open(CONFIG_FROZEN_PATH, "r", encoding="utf-8") as f:
        frozen_config = json.load(f)

    fmt_res = assign_formats_and_scales(
        graph=graph,
        weights_dict=artifacts["tensors"],
        frozen_config=frozen_config,
    )

    rom_dir = tmp_path / "compiled_roms_trace"
    tile_r, tile_c = 8, 16

    addr_map = pack_rom_artifacts(
        graph=graph,
        format_result=fmt_res,
        weights_dict=artifacts["tensors"],
        out_dir=rom_dir,
        tile_r=tile_r,
        tile_c=tile_c,
        max_ctx=32,
        head_dim=16,
    )

    prog = generate_schedule(
        address_map=addr_map,
        config=artifacts["config"],
        tile_r=tile_r,
        tile_c=tile_c,
    )
    prog.save(rom_dir)
    fmt_res.save_json(rom_dir / "format_table.json")

    interp = CompiledArtifactInterpreter(rom_dir)
    interp.reset()
    _, _ = interp.decode_step(token_id=42, pos=0)

    # Check captured traces
    assert "embed_tokens_tok42" in interp.traces
    assert "layer_attn_norm_L0" in interp.traces
    assert "layer_q_proj_L0" in interp.traces
    assert "layer_swiglu_L0" in interp.traces
    assert "final_norm_L0" in interp.traces
    assert "lm_head_L0" in interp.traces
    for trace_arr in interp.traces.values():
        assert trace_arr.size > 0

