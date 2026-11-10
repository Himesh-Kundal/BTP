"""Sequencer Schedule and Microprogram Generator (Task UB.5).

Emits static microprogram instructions for the model accelerator sequencer:
- Compact layer-loop body (tens of instructions rather than thousands)
- Per-layer ROM base table mapping layer iterations to ROM offsets
- Program ROM image (`program.hex`)
- Analytical cycle estimates per op and per decoded token

Strict Layering (Golden Rule 6):
Only pure Python and NumPy are used. Never import torch or transformers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any


class Opcode(IntEnum):
    HALT = 0
    EMBED = 1
    RMSNORM = 2
    LINEAR = 3
    ROPE = 4
    ATTENTION = 5
    ADD = 6
    SWIGLU = 7
    ARGMAX = 8


@dataclass
class Instruction:
    opcode: Opcode
    name: str
    src_buf: int
    dst_buf: int
    dims: list[int]
    rom_weight_base: int = 0
    rom_param_base: int = 0
    flags: int = 0
    cycle_estimate: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "opcode": self.opcode.name,
            "opcode_val": int(self.opcode),
            "name": self.name,
            "src_buf": self.src_buf,
            "dst_buf": self.dst_buf,
            "dims": list(self.dims),
            "rom_weight_base": self.rom_weight_base,
            "rom_param_base": self.rom_param_base,
            "flags": self.flags,
            "cycle_estimate": self.cycle_estimate,
        }

    def to_hex_word(self) -> str:
        """Pack instruction into a 128-bit hexadecimal word.
        
        Layout (128 bits):
        [127:120] opcode (8 bits)
        [119:112] src_buf (8 bits)
        [111:104] dst_buf (8 bits)
        [103:96]  flags (8 bits)
        [95:64]   rom_weight_base (32 bits)
        [63:32]   rom_param_base (32 bits)
        [31:16]   dim_0 (16 bits)
        [15:0]    dim_1 (16 bits)
        """
        dim0 = self.dims[0] if len(self.dims) > 0 else 0
        dim1 = self.dims[1] if len(self.dims) > 1 else 0

        w_high = (
            (int(self.opcode) << 24)
            | ((self.src_buf & 0xFF) << 16)
            | ((self.dst_buf & 0xFF) << 8)
            | (self.flags & 0xFF)
        )
        return (
            f"{w_high:08x}"
            f"{self.rom_weight_base & 0xFFFFFFFF:08x}"
            f"{self.rom_param_base & 0xFFFFFFFF:08x}"
            f"{dim0 & 0xFFFF:04x}{dim1 & 0xFFFF:04x}"
        )


@dataclass
class Microprogram:
    instructions: list[Instruction]
    layer_loop_start: int
    layer_loop_end: int
    num_layers: int
    layer_rom_bases: list[dict[str, int]]
    total_cycles_per_token: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "instructions": [ins.to_dict() for ins in self.instructions],
            "layer_loop_start": self.layer_loop_start,
            "layer_loop_end": self.layer_loop_end,
            "num_layers": self.num_layers,
            "layer_rom_bases": self.layer_rom_bases,
            "total_cycles_per_token": self.total_cycles_per_token,
        }

    def save(self, out_dir: str | Path) -> None:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        # 1. schedule.json
        with open(p / "schedule.json", "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        # 2. program.hex
        hex_lines = [ins.to_hex_word() for ins in self.instructions]
        with open(p / "program.hex", "w", encoding="utf-8") as f:
            f.write("\n".join(hex_lines) + "\n")


def estimate_op_cycles(
    opcode: Opcode,
    dims: list[int],
    tile_r: int = 8,
    tile_c: int = 16,
    max_ctx: int = 256,
) -> int:
    """Compute analytical hardware cycles for a given operation and engine grid."""
    if opcode == Opcode.LINEAR:
        # dims: [N, K]
        n = dims[0]
        k = dims[1]
        tiles_n = (n + tile_r - 1) // tile_r
        tiles_k = (k + tile_c - 1) // tile_c
        # T = tiles_n * tiles_k * tile_r (streaming) + pipeline latency
        return (tiles_n * tiles_k * tile_r) + 8

    elif opcode == Opcode.RMSNORM:
        # dims: [D] -> sum of squares + rsqrt pipe + normalize
        d = dims[0]
        return (2 * d) + 12

    elif opcode == Opcode.ROPE:
        # dims: [num_heads * head_dim]
        d = dims[0]
        return (d // 2) + 4

    elif opcode == Opcode.ATTENTION:
        # dims: [num_q_heads, num_kv_heads, head_dim]
        # Multi-head dot products + softmax + context accumulate over ctx positions
        num_q = dims[0]
        return (num_q * max_ctx) + 24

    elif opcode == Opcode.ADD:
        d = dims[0]
        return d + 4

    elif opcode == Opcode.SWIGLU:
        d = dims[0]
        return d + 8

    elif opcode == Opcode.EMBED:
        d = dims[0]
        return d + 2

    elif opcode == Opcode.ARGMAX:
        v = dims[0]
        return v + 4

    return 1


def generate_schedule(
    address_map: dict[str, Any],
    config: dict[str, Any],
    tile_r: int = 8,
    tile_c: int = 16,
) -> Microprogram:
    """Generate static microprogram and layer loop schedule."""
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    vocab_size = config["vocab_size"]
    num_layers = config["num_hidden_layers"]
    num_q_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    max_ctx = config.get("max_position_embeddings", 256)

    instructions: list[Instruction] = []
    layer_rom_bases: list[dict[str, int]] = []

    # Helper to lookup ROM addresses
    weights_map = address_map.get("weights", {})
    requant_map = address_map.get("requant_params", {})

    # Buffer IDs:
    # 0: BUF_RES (Residual stream container)
    # 1: BUF_ACT (Normed activations)
    # 2: BUF_Q (Query)
    # 3: BUF_K (Key)
    # 4: BUF_V (Value)
    # 5: BUF_ATTN_OUT (Attention context out)
    # 6: BUF_MLP_GATE (Gate proj out)
    # 7: BUF_MLP_UP (Up proj out)
    # 8: BUF_MLP_SWIGLU (SwiGLU out)
    # 9: BUF_LOGITS (Vocab logits)

    # 1. Embed Token: tok_in -> BUF_RES
    embed_w_base = weights_map.get("embed_tokens", {}).get("base_addr", 0)
    embed_p_base = requant_map.get("embed_tokens", {}).get("base_addr", 0)
    ins_embed = Instruction(
        opcode=Opcode.EMBED,
        name="embed_gather",
        src_buf=0,
        dst_buf=0,
        dims=[hidden_size],
        rom_weight_base=embed_w_base,
        rom_param_base=embed_p_base,
        cycle_estimate=estimate_op_cycles(Opcode.EMBED, [hidden_size]),
    )
    instructions.append(ins_embed)

    # 2. Layer Loop Template
    loop_start_pc = len(instructions)

    # For each layer, extract per-layer ROM bases
    for l_idx in range(num_layers):
        l_bases: dict[str, int] = {}
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            node_id = f"{proj}_{l_idx}"
            l_bases[f"{proj}_w"] = weights_map.get(node_id, {}).get("base_addr", 0)
            l_bases[f"{proj}_p"] = requant_map.get(node_id, {}).get("base_addr", 0)
        layer_rom_bases.append(l_bases)

    # Single parameterized layer body:
    # 2a. Attention RMSNorm: BUF_RES -> BUF_ACT
    instructions.append(
        Instruction(
            opcode=Opcode.RMSNORM,
            name="layer_attn_norm",
            src_buf=0,
            dst_buf=1,
            dims=[hidden_size],
            cycle_estimate=estimate_op_cycles(Opcode.RMSNORM, [hidden_size]),
        )
    )

    # 2b. Q, K, V Projections: BUF_ACT -> BUF_Q, BUF_K, BUF_V
    instructions.append(
        Instruction(
            opcode=Opcode.LINEAR,
            name="layer_q_proj",
            src_buf=1,
            dst_buf=2,
            dims=[num_q_heads * head_dim, hidden_size],
            cycle_estimate=estimate_op_cycles(
                Opcode.LINEAR, [num_q_heads * head_dim, hidden_size], tile_r, tile_c
            ),
        )
    )
    instructions.append(
        Instruction(
            opcode=Opcode.LINEAR,
            name="layer_k_proj",
            src_buf=1,
            dst_buf=3,
            dims=[num_kv_heads * head_dim, hidden_size],
            cycle_estimate=estimate_op_cycles(
                Opcode.LINEAR, [num_kv_heads * head_dim, hidden_size], tile_r, tile_c
            ),
        )
    )
    instructions.append(
        Instruction(
            opcode=Opcode.LINEAR,
            name="layer_v_proj",
            src_buf=1,
            dst_buf=4,
            dims=[num_kv_heads * head_dim, hidden_size],
            cycle_estimate=estimate_op_cycles(
                Opcode.LINEAR, [num_kv_heads * head_dim, hidden_size], tile_r, tile_c
            ),
        )
    )

    # 2c. RoPE on Q and K
    instructions.append(
        Instruction(
            opcode=Opcode.ROPE,
            name="layer_q_rope",
            src_buf=2,
            dst_buf=2,
            dims=[num_q_heads, head_dim],
            cycle_estimate=estimate_op_cycles(Opcode.ROPE, [num_q_heads * head_dim]),
        )
    )
    instructions.append(
        Instruction(
            opcode=Opcode.ROPE,
            name="layer_k_rope",
            src_buf=3,
            dst_buf=3,
            dims=[num_kv_heads, head_dim],
            cycle_estimate=estimate_op_cycles(Opcode.ROPE, [num_kv_heads * head_dim]),
        )
    )

    # 2d. Attention: Q, K, V -> BUF_ATTN_OUT
    instructions.append(
        Instruction(
            opcode=Opcode.ATTENTION,
            name="layer_attention",
            src_buf=2,
            dst_buf=5,
            dims=[num_q_heads, num_kv_heads, head_dim],
            cycle_estimate=estimate_op_cycles(
                Opcode.ATTENTION, [num_q_heads, num_kv_heads, head_dim], max_ctx=max_ctx
            ),
        )
    )

    # 2e. O proj: BUF_ATTN_OUT -> BUF_ACT
    instructions.append(
        Instruction(
            opcode=Opcode.LINEAR,
            name="layer_o_proj",
            src_buf=5,
            dst_buf=1,
            dims=[hidden_size, hidden_size],
            cycle_estimate=estimate_op_cycles(
                Opcode.LINEAR, [hidden_size, hidden_size], tile_r, tile_c
            ),
        )
    )

    # 2f. Residual Add 1: BUF_RES + BUF_ACT -> BUF_RES
    instructions.append(
        Instruction(
            opcode=Opcode.ADD,
            name="layer_res_add_attn",
            src_buf=1,
            dst_buf=0,
            dims=[hidden_size],
            cycle_estimate=estimate_op_cycles(Opcode.ADD, [hidden_size]),
        )
    )

    # 2g. MLP RMSNorm: BUF_RES -> BUF_ACT
    instructions.append(
        Instruction(
            opcode=Opcode.RMSNORM,
            name="layer_mlp_norm",
            src_buf=0,
            dst_buf=1,
            dims=[hidden_size],
            cycle_estimate=estimate_op_cycles(Opcode.RMSNORM, [hidden_size]),
        )
    )

    # 2h. Gate & Up projections: BUF_ACT -> BUF_MLP_GATE, BUF_MLP_UP
    instructions.append(
        Instruction(
            opcode=Opcode.LINEAR,
            name="layer_gate_proj",
            src_buf=1,
            dst_buf=6,
            dims=[intermediate_size, hidden_size],
            cycle_estimate=estimate_op_cycles(
                Opcode.LINEAR, [intermediate_size, hidden_size], tile_r, tile_c
            ),
        )
    )
    instructions.append(
        Instruction(
            opcode=Opcode.LINEAR,
            name="layer_up_proj",
            src_buf=1,
            dst_buf=7,
            dims=[intermediate_size, hidden_size],
            cycle_estimate=estimate_op_cycles(
                Opcode.LINEAR, [intermediate_size, hidden_size], tile_r, tile_c
            ),
        )
    )

    # 2i. SwiGLU: SiLU(GATE) * UP -> BUF_MLP_SWIGLU
    instructions.append(
        Instruction(
            opcode=Opcode.SWIGLU,
            name="layer_swiglu",
            src_buf=6,
            dst_buf=8,
            dims=[intermediate_size],
            cycle_estimate=estimate_op_cycles(Opcode.SWIGLU, [intermediate_size]),
        )
    )

    # 2j. Down proj: BUF_MLP_SWIGLU -> BUF_ACT
    instructions.append(
        Instruction(
            opcode=Opcode.LINEAR,
            name="layer_down_proj",
            src_buf=8,
            dst_buf=1,
            dims=[hidden_size, intermediate_size],
            cycle_estimate=estimate_op_cycles(
                Opcode.LINEAR, [hidden_size, intermediate_size], tile_r, tile_c
            ),
        )
    )

    # 2k. Residual Add 2: BUF_RES + BUF_ACT -> BUF_RES
    instructions.append(
        Instruction(
            opcode=Opcode.ADD,
            name="layer_res_add_mlp",
            src_buf=1,
            dst_buf=0,
            dims=[hidden_size],
            cycle_estimate=estimate_op_cycles(Opcode.ADD, [hidden_size]),
        )
    )

    loop_end_pc = len(instructions) - 1

    # 3. Final Norm: BUF_RES -> BUF_ACT
    instructions.append(
        Instruction(
            opcode=Opcode.RMSNORM,
            name="final_norm",
            src_buf=0,
            dst_buf=1,
            dims=[hidden_size],
            cycle_estimate=estimate_op_cycles(Opcode.RMSNORM, [hidden_size]),
        )
    )

    # 4. LM Head: BUF_ACT -> BUF_LOGITS
    lm_w_base = weights_map.get("lm_head", {}).get("base_addr", 0)
    lm_p_base = requant_map.get("lm_head", {}).get("base_addr", 0)
    instructions.append(
        Instruction(
            opcode=Opcode.LINEAR,
            name="lm_head",
            src_buf=1,
            dst_buf=9,
            dims=[vocab_size, hidden_size],
            rom_weight_base=lm_w_base,
            rom_param_base=lm_p_base,
            cycle_estimate=estimate_op_cycles(
                Opcode.LINEAR, [vocab_size, hidden_size], tile_r, tile_c
            ),
        )
    )

    # 5. ArgMax: BUF_LOGITS -> tok_out
    instructions.append(
        Instruction(
            opcode=Opcode.ARGMAX,
            name="argmax",
            src_buf=9,
            dst_buf=0,
            dims=[vocab_size],
            cycle_estimate=estimate_op_cycles(Opcode.ARGMAX, [vocab_size]),
        )
    )

    # 6. Halt
    instructions.append(
        Instruction(
            opcode=Opcode.HALT,
            name="halt",
            src_buf=0,
            dst_buf=0,
            dims=[0],
            cycle_estimate=1,
        )
    )

    # Calculate total per-token cycle estimate
    layer_cycles = sum(ins.cycle_estimate for ins in instructions[loop_start_pc : loop_end_pc + 1])
    prologue_cycles = sum(ins.cycle_estimate for ins in instructions[:loop_start_pc])
    epilogue_cycles = sum(ins.cycle_estimate for ins in instructions[loop_end_pc + 1 :])

    total_cycles = prologue_cycles + (layer_cycles * num_layers) + epilogue_cycles

    return Microprogram(
        instructions=instructions,
        layer_loop_start=loop_start_pc,
        layer_loop_end=loop_end_pc,
        num_layers=num_layers,
        layer_rom_bases=layer_rom_bases,
        total_cycles_per_token=total_cycles,
    )
