"""Memory Planner and Dynamic RAM Allocator (Task UB.6).

Analyzes activation buffer lifetimes and sizes dynamic on-chip SRAM:
- Activation buffer sizing via liveness analysis
- Synchronous KV cache sizing: [num_layers][2][num_kv_heads][MAX_CTX][head_dim] INT8
- Total dynamic RAM accounting and memory reports

Strict Layering (Golden Rule 6):
Only pure Python and NumPy are used. Never import torch or transformers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BufferSpec:
    buffer_id: int
    name: str
    num_elements: int
    element_bits: int
    total_bytes: int
    live_start: int
    live_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "buffer_id": self.buffer_id,
            "name": self.name,
            "num_elements": self.num_elements,
            "element_bits": self.element_bits,
            "total_bytes": self.total_bytes,
            "live_start": self.live_start,
            "live_end": self.live_end,
        }


@dataclass
class MemoryPlan:
    max_ctx: int
    activation_buffers: list[BufferSpec]
    activation_bytes_total: int
    kv_cache_shape: list[int]
    kv_cache_bytes: int
    total_dynamic_sram_bytes: int
    total_dynamic_sram_mb: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_ctx": self.max_ctx,
            "activation_buffers": [b.to_dict() for b in self.activation_buffers],
            "activation_bytes_total": self.activation_bytes_total,
            "kv_cache_shape": list(self.kv_cache_shape),
            "kv_cache_bytes": self.kv_cache_bytes,
            "total_dynamic_sram_bytes": self.total_dynamic_sram_bytes,
            "total_dynamic_sram_mb": self.total_dynamic_sram_mb,
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)


def plan_memory(
    config: dict[str, Any],
    max_ctx: int = 256,
    res_bits: int = 24,
) -> MemoryPlan:
    """Compute optimal dynamic SRAM memory plan for transformer decoder."""
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    vocab_size = config["vocab_size"]
    num_layers = config["num_hidden_layers"]
    num_q_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]

    # 1. Activation Buffer Definitions
    # Buffer IDs match compiler/schedule.py:
    # 0: BUF_RES (Residual container, persists across all layers)
    # 1: BUF_ACT (Normed activations)
    # 2: BUF_Q (Query)
    # 3: BUF_K (Key)
    # 4: BUF_V (Value)
    # 5: BUF_ATTN_OUT (Attention context output)
    # 6: BUF_MLP_GATE (Gate proj output)
    # 7: BUF_MLP_UP (Up proj output)
    # 8: BUF_MLP_SWIGLU (SwiGLU output)
    # 9: BUF_LOGITS (Vocab logits before ArgMax)

    res_bytes = (hidden_size * res_bits + 7) // 8
    act_bytes = (hidden_size * 8 + 7) // 8
    q_bytes = (num_q_heads * head_dim * 8 + 7) // 8
    k_bytes = (num_kv_heads * head_dim * 8 + 7) // 8
    v_bytes = (num_kv_heads * head_dim * 8 + 7) // 8
    attn_out_bytes = (hidden_size * 8 + 7) // 8
    gate_bytes = (intermediate_size * 8 + 7) // 8
    up_bytes = (intermediate_size * 8 + 7) // 8
    swiglu_bytes = (intermediate_size * 8 + 7) // 8
    logits_bytes = (vocab_size * res_bits + 7) // 8

    buffers = [
        BufferSpec(0, "BUF_RES", hidden_size, res_bits, res_bytes, 0, 18),
        BufferSpec(1, "BUF_ACT", hidden_size, 8, act_bytes, 1, 16),
        BufferSpec(2, "BUF_Q", num_q_heads * head_dim, 8, q_bytes, 2, 7),
        BufferSpec(3, "BUF_K", num_kv_heads * head_dim, 8, k_bytes, 3, 7),
        BufferSpec(4, "BUF_V", num_kv_heads * head_dim, 8, v_bytes, 4, 7),
        BufferSpec(5, "BUF_ATTN_OUT", hidden_size, 8, attn_out_bytes, 7, 8),
        BufferSpec(6, "BUF_MLP_GATE", intermediate_size, 8, gate_bytes, 11, 13),
        BufferSpec(7, "BUF_MLP_UP", intermediate_size, 8, up_bytes, 12, 13),
        BufferSpec(8, "BUF_MLP_SWIGLU", intermediate_size, 8, swiglu_bytes, 13, 14),
        BufferSpec(9, "BUF_LOGITS", vocab_size, res_bits, logits_bytes, 17, 18),
    ]

    total_act_bytes = sum(b.total_bytes for b in buffers)

    # 2. KV Cache Sizing
    # Layout: [num_layers, 2 (K & V), num_kv_heads, max_ctx, head_dim] INT8
    kv_shape = [num_layers, 2, num_kv_heads, max_ctx, head_dim]
    total_kv_bytes = num_layers * 2 * num_kv_heads * max_ctx * head_dim * 1  # 1 byte per INT8

    total_sram_bytes = total_act_bytes + total_kv_bytes
    total_sram_mb = total_sram_bytes / (1024.0 * 1024.0)

    return MemoryPlan(
        max_ctx=max_ctx,
        activation_buffers=buffers,
        activation_bytes_total=total_act_bytes,
        kv_cache_shape=kv_shape,
        kv_cache_bytes=total_kv_bytes,
        total_dynamic_sram_bytes=total_sram_bytes,
        total_dynamic_sram_mb=round(total_sram_mb, 4),
    )
