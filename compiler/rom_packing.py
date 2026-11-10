"""ROM Packing Pass & Address Map Generation (Task UB.4).

Serializes all immutable model parameters (weights, lookup tables, and dyadic
requantization parameters) into flat `$readmemh`-compatible `.hex` files and
generates a comprehensive `address_map.json` with cryptographic SHA256 hashes.

Strict Layering (Golden Rule 6):
Only pure Python, NumPy, and IR dataclasses are used. Never import torch or transformers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.tiling import pad_matrix
from compiler.format_assignment import FormatAssignmentResult
from ir.graph_v2 import GraphV2, OpKind
from numerics.ops import (
    build_exp2_fraction_lut,
    build_reciprocal_lut,
    build_rope_cos_sin_tables,
    build_rsqrt_mantissa_lut,
    build_silu_lut,
)


def compute_bytes_sha256(data: bytes) -> str:
    """Compute standard SHA256 hex digest of bytes."""
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def int_to_twos_comp_hex(val: int, bits: int) -> str:
    """Convert a signed or unsigned integer into two's complement hex string of fixed length."""
    val_int = int(val)
    if val_int < 0:
        val_int = (1 << bits) + val_int
    mask = (1 << bits) - 1
    val_int = val_int & mask
    hex_len = (bits + 3) // 4
    return f"{val_int:0{hex_len}x}"


def pack_rom_artifacts(
    graph: GraphV2,
    format_result: FormatAssignmentResult,
    weights_dict: dict[str, np.ndarray],
    out_dir: str | Path,
    tile_r: int = 8,
    tile_c: int = 16,
    max_ctx: int = 256,
    rope_theta: float = 100000.0,
    head_dim: int = 64,
) -> dict[str, Any]:
    """Tile and pack all immutable weights, requant parameters, and tables into ROM images."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    address_map: dict[str, Any] = {
        "engine_config": {"tile_r": tile_r, "tile_c": tile_c},
        "weights": {},
        "requant_params": {},
        "tables": {},
    }

    # =========================================================================
    # 1. Pack Weight ROM (weights.hex)
    # =========================================================================
    weight_hex_lines: list[str] = []
    current_weight_offset = 0

    # Collect all linear/embedding nodes
    weight_nodes = [
        n for n in graph.nodes if n.op in (OpKind.Linear, OpKind.Embed) and len(n.inputs) > 1
    ]

    for node in weight_nodes:
        w_name = node.inputs[1]
        w_arr = weights_dict.get(w_name)
        if w_arr is None:
            continue

        # Quantize float weights to symmetric INT8 [-127, 127]
        if w_arr.dtype != np.int8:
            ch_max = np.max(np.abs(w_arr), axis=-1, keepdims=True)
            ch_max = np.maximum(ch_max, 1e-8)
            w_int8 = np.clip(np.round(w_arr / ch_max * 127.0), -127, 127).astype(np.int8)
        else:
            w_int8 = w_arr

        # Pad matrix to exact (tile_r, tile_c) multiples
        w_padded, n_pad, k_pad = pad_matrix(w_int8, tile_n=tile_r, tile_k=tile_c)

        # Tile ordering: loop over tile rows, then tile cols, then inner tile elements
        num_tiles_r = n_pad // tile_r
        num_tiles_c = k_pad // tile_c

        start_offset = current_weight_offset
        tile_count = 0

        for tr in range(num_tiles_r):
            for tc in range(num_tiles_c):
                tile = w_padded[tr * tile_r : (tr + 1) * tile_r, tc * tile_c : (tc + 1) * tile_c]
                # Emit row-major tile elements
                for r in range(tile_r):
                    for c in range(tile_c):
                        val = int(tile[r, c])
                        weight_hex_lines.append(int_to_twos_comp_hex(val, 8))
                        current_weight_offset += 1
                tile_count += 1

        length = current_weight_offset - start_offset
        address_map["weights"][node.id] = {
            "tensor_name": w_name,
            "base_addr": start_offset,
            "length": length,
            "orig_shape": list(w_arr.shape),
            "padded_shape": [n_pad, k_pad],
            "num_tiles": [num_tiles_r, num_tiles_c],
        }

    weight_hex_file = out_path / "weights.hex"
    with open(weight_hex_file, "w", encoding="utf-8") as f:
        f.write("\n".join(weight_hex_lines) + "\n")

    weight_bytes = weight_hex_file.read_bytes()
    address_map["weights_image"] = {
        "file": "weights.hex",
        "total_words": current_weight_offset,
        "word_bits": 8,
        "sha256": compute_bytes_sha256(weight_bytes),
    }

    # =========================================================================
    # 2. Pack Requant Parameters ROM (requant_params.hex)
    # =========================================================================
    # Each entry is a 64-bit word: {M[31:0], 26'b0, shift[5:0]} -> 16 hex chars
    param_hex_lines: list[str] = []
    current_param_offset = 0

    for node_id, params in format_result.requant_params.items():
        m_list = params["M"]
        s_list = params["shift"]
        start_offset = current_param_offset

        for M_val, s_val in zip(m_list, s_list, strict=False):
            # 32 bits for M, 32 bits for shift
            m_hex = int_to_twos_comp_hex(M_val, 32)
            s_hex = int_to_twos_comp_hex(s_val, 32)
            param_hex_lines.append(f"{m_hex}{s_hex}")
            current_param_offset += 1

        length = current_param_offset - start_offset
        address_map["requant_params"][node_id] = {
            "base_addr": start_offset,
            "length": length,
            "channels": len(m_list),
        }

    param_hex_file = out_path / "requant_params.hex"
    with open(param_hex_file, "w", encoding="utf-8") as f:
        f.write("\n".join(param_hex_lines) + "\n")

    param_bytes = param_hex_file.read_bytes()
    address_map["requant_image"] = {
        "file": "requant_params.hex",
        "total_words": current_param_offset,
        "word_bits": 64,
        "sha256": compute_bytes_sha256(param_bytes),
    }

    # =========================================================================
    # 3. Pack Mathematical Lookup Tables ROM (tables.hex)
    # =========================================================================
    table_hex_lines: list[str] = []
    current_table_offset = 0

    def _add_table(name: str, values: np.ndarray, bits: int) -> None:
        nonlocal current_table_offset
        start_offset = current_table_offset
        for v in values.flat:
            table_hex_lines.append(int_to_twos_comp_hex(int(v), bits))
            current_table_offset += 1
        length = current_table_offset - start_offset
        address_map["tables"][name] = {
            "base_addr": start_offset,
            "length": length,
            "shape": list(values.shape),
            "bits": bits,
        }

    # 3a. RoPE tables
    cos_tab, sin_tab = build_rope_cos_sin_tables(
        max_ctx=max_ctx,
        head_dim=head_dim,
        theta=rope_theta,
    )
    _add_table("rope_cos", cos_tab, 16)
    _add_table("rope_sin", sin_tab, 16)

    # 3b. RMSNorm rsqrt mantissa LUT (1024 entries, Q0.16)
    rsqrt_lut = build_rsqrt_mantissa_lut(table_size=1024, out_bits=16)
    _add_table("rsqrt_mantissa_lut", rsqrt_lut, 16)

    # 3c. Softmax exp2 fraction LUT (256 entries, Q0.16)
    exp2_lut = build_exp2_fraction_lut(table_size=256, out_bits=16)
    _add_table("exp2_lut", exp2_lut, 16)

    # 3d. Softmax reciprocal LUT (256 entries, Q0.16)
    recip_lut = build_reciprocal_lut(table_size=256, out_bits=16)
    _add_table("reciprocal_lut", recip_lut, 16)

    # 3e. SwiGLU SiLU LUT (512 entries, 8 bits)
    silu_lut = build_silu_lut(in_min=-256, in_max=255)
    _add_table("silu_lut", silu_lut, 8)

    table_hex_file = out_path / "tables.hex"
    with open(table_hex_file, "w", encoding="utf-8") as f:
        f.write("\n".join(table_hex_lines) + "\n")

    table_bytes = table_hex_file.read_bytes()
    address_map["tables_image"] = {
        "file": "tables.hex",
        "total_words": current_table_offset,
        "word_bits": 16,
        "sha256": compute_bytes_sha256(table_bytes),
    }

    # 4. Save address_map.json
    map_file = out_path / "address_map.json"
    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(address_map, f, indent=2, sort_keys=True)

    return address_map
