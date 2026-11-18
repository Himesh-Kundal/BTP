"""Top-Level Accelerator Generator (Task UD.1).

Composes and generates the complete model-specific hardware accelerator:
- Microprogram sequencer (sequencer.v)
- Non-linear units (embed_gather.v, rmsnorm.v, rope.v, attention.v, residual_add.v, swiglu.v, argmax.v)
- RxC streaming linear engine (linear_engine.v)
- Synchronous dynamic memories (kv_cache.v, activation buffers)
- Immutable ROM instances (rom_weights, rom_requant_params, rom_tables, rom_program, rom_layer_bases)
- Top-level step execution interface (clk, rst_n, seq_reset, step_valid, step_ready, tok_in, out_valid, tok_out)

Strict Layering (Golden Rule 6):
Only pure Python and NumPy are used. Never import torch or transformers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from compiler.format_assignment import assign_formats_and_scales
from compiler.rom_packing import pack_rom_artifacts
from compiler.schedule import generate_schedule
from ir.graph_v2 import GraphV2


def generate_top_accel_verilog(
    config: dict[str, Any],
    rom_files: dict[str, str] | None = None,
    module_name: str = "top_accel",
    rom_depths: dict[str, int] | None = None,
) -> str:
    """Generate parameterized top-level Verilog composition for the model accelerator.

    Args:
        config: Model configuration dictionary (vocab_size, hidden_size, etc.).
        rom_files: Optional mapping of ROM names to hex file paths.
        module_name: Top-level module name.
        rom_depths: Optional mapping of ROM depths (weights, params, prog, bases).

    Returns:
        Verilog source string for top_accel.
    """
    vocab_size = config.get("vocab_size", 256)
    hidden_size = config.get("hidden_size", 64)
    intermediate_size = config.get("intermediate_size", 128)
    num_layers = config.get("num_hidden_layers", 2)
    num_q_heads = config.get("num_attention_heads", 4)
    num_kv_heads = config.get("num_key_value_heads", 2)
    head_dim = config.get("head_dim", 16)
    max_ctx = config.get("max_position_embeddings", 32)
    tile_r = config.get("tile_r", 8)
    tile_c = config.get("tile_c", 16)

    depths = rom_depths or {}
    w_depth = max(32768, depths.get("weights", 32768))
    p_depth = max(2048, depths.get("params", 2048))
    prog_depth = max(256, depths.get("prog", 256))
    bases_depth = max(512, depths.get("bases", 512))

    w_addr_w = max(16, (w_depth - 1).bit_length())
    p_addr_w = max(16, (p_depth - 1).bit_length())
    bases_addr_w = max(9, (bases_depth - 1).bit_length())

    files = rom_files or {}
    w_file = files.get("weights", "")
    p_file = files.get("requant_params", "")
    prog_file = files.get("program", "")
    bases_file = files.get("layer_bases", "")
    exp2_file = files.get("exp2", "")
    rsqrt_file = files.get("rsqrt", "")
    silu_file = files.get("silu", "")
    cos_file = files.get("rope_cos", "")
    sin_file = files.get("rope_sin", "")

    # We read the canonical template from rtl/lib/top_accel.v
    template_path = Path(__file__).resolve().parent.parent / "rtl" / "lib" / "top_accel.v"
    content = template_path.read_text(encoding="utf-8")

    param_block = (
        f"module {module_name} #(\n"
        f"    parameter VOCAB_SIZE        = {vocab_size},\n"
        f"    parameter HIDDEN_SIZE       = {hidden_size},\n"
        f"    parameter INTERMEDIATE_SIZE = {intermediate_size},\n"
        f"    parameter NUM_LAYERS        = {num_layers},\n"
        f"    parameter NUM_Q_HEADS       = {num_q_heads},\n"
        f"    parameter NUM_KV_HEADS      = {num_kv_heads},\n"
        f"    parameter HEAD_DIM          = {head_dim},\n"
        f"    parameter MAX_CTX           = {max_ctx},\n"
        f"    parameter TILE_R            = {tile_r},\n"
        f"    parameter TILE_C            = {tile_c},\n"
        f"    parameter PROG_ADDR_W       = 8,\n"
        f"    parameter LAYER_W           = 5,\n"
        f"    parameter ROM_WEIGHTS_ADDR_W = {w_addr_w},\n"
        f"    parameter ROM_PARAMS_ADDR_W  = {p_addr_w},\n"
        f"    parameter ROM_BASES_ADDR_W   = {bases_addr_w},\n"
        f"    parameter ROM_WEIGHTS_DEPTH = {w_depth},\n"
        f"    parameter ROM_PARAMS_DEPTH  = {p_depth},\n"
        f"    parameter ROM_PROG_DEPTH    = {prog_depth},\n"
        f"    parameter ROM_BASES_DEPTH   = {bases_depth},\n"
        f'    parameter ROM_WEIGHTS_FILE  = "{w_file}",\n'
        f'    parameter ROM_PARAMS_FILE   = "{p_file}",\n'
        f'    parameter ROM_PROG_FILE     = "{prog_file}",\n'
        f'    parameter ROM_BASES_FILE    = "{bases_file}",\n'
        f'    parameter ROM_EXP2_FILE     = "{exp2_file}",\n'
        f'    parameter ROM_RSQRT_FILE    = "{rsqrt_file}",\n'
        f'    parameter ROM_SILU_FILE     = "{silu_file}",\n'
        f'    parameter ROM_COS_FILE      = "{cos_file}",\n'
        f'    parameter ROM_SIN_FILE      = "{sin_file}"\n'
        f")("
    )
    custom_content = re.sub(r"module\s+\w+\s*#\(.*?\)\s*\(", param_block, content, count=1, flags=re.DOTALL)

    return custom_content


def emit_hardware_bundle(
    graph: GraphV2,
    weights_dict: dict[str, Any],
    config: dict[str, Any],
    frozen_config: dict[str, Any],
    out_dir: Path | str,
    tile_r: int = 8,
    tile_c: int = 16,
    max_ctx: int = 32,
    calib_scales: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Emit complete hardware bundle: ROMs, address map, program, and top_accel.v."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Format assignment pass
    fmt_res = assign_formats_and_scales(
        graph=graph,
        weights_dict=weights_dict,
        frozen_config=frozen_config,
        calib_scales=calib_scales,
    )
    fmt_res.save_json(out_path / "format_table.json")

    # 2. ROM packing pass
    addr_map = pack_rom_artifacts(
        graph=graph,
        format_result=fmt_res,
        weights_dict=weights_dict,
        out_dir=out_path,
        tile_r=tile_r,
        tile_c=tile_c,
        max_ctx=max_ctx,
        head_dim=config.get("head_dim", 16),
    )

    # 3. Schedule generation
    microprog = generate_schedule(
        address_map=addr_map,
        config=config,
        tile_r=tile_r,
        tile_c=tile_c,
    )
    microprog.save(out_path)

    # 4. Emit 128-bit words for weights (weights_128b.hex)
    w_hex_path = out_path / "weights.hex"
    w128_path = out_path / "weights_128b.hex"
    raw_w_bytes = w_hex_path.read_bytes()
    chunk_size = tile_c  # 16 bytes = 128 bits

    if len(raw_w_bytes) > 0 and len(raw_w_bytes) % 3 == 0:
        # Fast path: every line in weights.hex is exactly 'xx\n' (3 bytes)
        byte_arr = np.frombuffer(raw_w_bytes, dtype=np.uint8).reshape(-1, 3)[:, :2]
        n_bytes = byte_arr.shape[0]
        rem = n_bytes % chunk_size
        if rem != 0:
            pad_len = chunk_size - rem
            pad_arr = np.full((pad_len, 2), ord("0"), dtype=np.uint8)
            byte_arr = np.vstack([byte_arr, pad_arr])
        words_arr = byte_arr.reshape(-1, chunk_size, 2)[:, ::-1, :].reshape(-1, chunk_size * 2)
        w128_count = words_arr.shape[0]
        nl_col = np.full((w128_count, 1), ord("\n"), dtype=np.uint8)
        out_arr = np.hstack([words_arr, nl_col])
        with open(w128_path, "wb") as f128:
            # Write in 1M-word chunks to bound memory
            f128.writelines(out_arr[idx : idx + 1_048_576].tobytes() for idx in range(0, w128_count, 1_048_576))
    else:
        raw_weights_lines = raw_w_bytes.decode("utf-8").strip().splitlines()
        weights_128b_lines: list[str] = []
        for i in range(0, len(raw_weights_lines), chunk_size):
            chunk = raw_weights_lines[i : i + chunk_size]
            while len(chunk) < chunk_size:
                chunk.append("00")
            weights_128b_lines.append("".join(reversed(chunk)))
        w128_count = len(weights_128b_lines)
        w128_path.write_text("\n".join(weights_128b_lines) + "\n", encoding="utf-8")

    # 5. Emit Layer Bases ROM hex (layer_bases.hex)
    layer_bases_lines: list[str] = []
    num_layers = config.get("num_hidden_layers", 2)
    # Inside layer loop there are 15 instructions (pc 1 to 15, op_idx 0 to 14)
    # The linear projections in layer loop:
    # op_idx 1: q_proj
    # op_idx 2: k_proj
    # op_idx 3: v_proj
    # op_idx 7: o_proj
    # op_idx 10: gate_proj
    # op_idx 11: up_proj
    # op_idx 13: down_proj
    weights_map = addr_map.get("weights", {})
    requant_map = addr_map.get("requant_params", {})

    proj_by_op_idx = {
        0: "attn_norm",
        1: "q_proj",
        2: "k_proj",
        3: "v_proj",
        7: "o_proj",
        9: "mlp_norm",
        10: "gate_proj",
        11: "up_proj",
        13: "down_proj",
    }

    for l_idx in range(num_layers):
        for op_idx in range(16):
            proj = proj_by_op_idx.get(op_idx)
            if proj is not None:
                node_id = f"{proj}_{l_idx}"
                w_base = weights_map.get(node_id, {}).get("base_addr", 0)
                p_base = requant_map.get(node_id, {}).get("base_addr", 0)
            else:
                w_base = 0
                p_base = 0
            layer_bases_lines.append(f"{w_base & 0xFFFFFFFF:08x}{p_base & 0xFFFFFFFF:08x}")

    bases_path = out_path / "layer_bases.hex"
    bases_path.write_text("\n".join(layer_bases_lines) + "\n", encoding="utf-8")

    # 6. Emit individual table hex files for dedicated ROMs
    tbl_meta = addr_map.get("tables", {})
    tables_lines = (out_path / "tables.hex").read_text(encoding="utf-8").strip().splitlines()

    for tbl_name in ("rope_cos", "rope_sin", "rsqrt_mantissa_lut", "exp2_lut", "silu_lut"):
        info = tbl_meta.get(tbl_name)
        if info:
            b = info["base_addr"]
            length = info["length"]
            if tbl_name == "exp2_lut":
                from numerics.ops import build_exp2_fraction_lut

                exp2_arr = build_exp2_fraction_lut(256, 16)
                sub_lines = [f"{int(v) & 0x3FFFF:05x}" for v in exp2_arr]
            else:
                sub_lines = tables_lines[b : b + length]
            short_name = tbl_name.replace("_lut", "").replace("_mantissa", "").replace("mantissa_", "")
            (out_path / f"tables_{short_name}.hex").write_text("\n".join(sub_lines) + "\n", encoding="utf-8")

    # 7. Emit top_accel.v configured with paths
    rom_files = {
        "weights": (out_path / "weights_128b.hex").as_posix(),
        "requant_params": (out_path / "requant_params.hex").as_posix(),
        "program": (out_path / "program.hex").as_posix(),
        "layer_bases": (out_path / "layer_bases.hex").as_posix(),
        "exp2": (out_path / "tables_exp2.hex").as_posix(),
        "rsqrt": (out_path / "tables_rsqrt.hex").as_posix(),
        "silu": (out_path / "tables_silu.hex").as_posix(),
        "rope_cos": (out_path / "tables_rope_cos.hex").as_posix(),
        "rope_sin": (out_path / "tables_rope_sin.hex").as_posix(),
    }

    rom_depths = {
        "weights": w128_count,
        "params": addr_map.get("requant_image", {}).get("total_words", 2048),
        "prog": 256,
        "bases": max(512, len(layer_bases_lines)),
    }

    top_verilog = generate_top_accel_verilog(config, rom_files=rom_files, rom_depths=rom_depths)
    top_file = out_path / "top_accel.v"
    top_file.write_text(top_verilog, encoding="utf-8")

    return {
        "out_dir": str(out_path),
        "address_map": addr_map,
        "microprogram": microprog,
        "top_file": str(top_file),
    }
