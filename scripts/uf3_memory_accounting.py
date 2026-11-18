"""UF.3 / EA3: Memory Accounting across ROMs, KV Cache, and Buffers.

Audits and accounts for all on-chip and ROM memory storage for SmolLM2-135M at MAX_CTX=256:
1. Immutable ROM: Weight ROM, Parameter ROM, Table ROMs (SiLU, rsqrt, exp2, RoPE), Program ROM.
2. Mutable SRAM: KV Cache at MAX_CTX=256, Activation buffers.
3. Ratio and breakdown calculations.
Emits results/part2/EA3.json.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "part2"
HW_DIR = REPO_ROOT / "build" / "smollm2_hw"


def run_ea3() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "EA3.json"

    # 1. Weights
    # 30 layers * 7 projections + tied embeddings
    raw_weights_bytes = 30 * (576 * 576 + 192 * 576 * 2 + 576 * 576 + 1536 * 576 * 3) + 49152 * 576
    padded_weights_words = 162849792  # from address_map.json / weights.hex
    padded_weights_bits = padded_weights_words * 8

    # 2. Parameters
    param_words = 288990  # from address_map.json / requant_params.hex
    param_word_bits = 64
    param_bits = param_words * param_word_bits

    # 3. Tables
    silu_lut_bits = 512 * 16
    rsqrt_lut_bits = 256 * 16
    exp2_lut_bits = 256 * 16
    rope_table_bits = 256 * 32 * 2 * 16
    tables_total_bits = silu_lut_bits + rsqrt_lut_bits + exp2_lut_bits + rope_table_bits

    # 4. Program
    program_ins_bits = 20 * 32
    layer_bases_bits = 30 * 9 * 32
    program_total_bits = program_ins_bits + layer_bases_bits

    # Total Immutable ROM
    total_rom_bits = padded_weights_bits + param_bits + tables_total_bits + program_total_bits

    # 5. Dynamic State - KV Cache (30 layers * 3 kv_heads * 256 pos * 64 dim * 8 bits * 2 [k, v])
    kv_cache_elements = 30 * 3 * 256 * 64 * 2
    kv_cache_bits = kv_cache_elements * 8

    # 6. Dynamic State - Activation Buffers
    buffers = {
        "buf_act": 576 * 8,
        "buf_res": 576 * 16,
        "buf_gate": 1536 * 8,
        "buf_up": 1536 * 8,
        "buf_swiglu": 1536 * 8,
        "buf_q": 576 * 8,
        "buf_attn_ctx": 576 * 8,
        "scores_buf": 9 * 256 * 16,
        "prob_mem": 256 * 16,
        "buf_logits": 49152 * 24,
    }
    total_buffers_bits = sum(buffers.values())

    # Total Mutable SRAM
    total_sram_bits = kv_cache_bits + total_buffers_bits

    grand_total_bits = total_rom_bits + total_sram_bits

    report = {
        "task": "UF.3",
        "experiment": "EA3",
        "model": "SmolLM2-135M",
        "max_ctx": 256,
        "hidden_size": 576,
        "intermediate_size": 1536,
        "vocab_size": 49152,
        "num_layers": 30,
        "num_q_heads": 9,
        "num_kv_heads": 3,
        "head_dim": 64,
        "rom_accounting": {
            "weight_rom": {
                "raw_int8_elements": raw_weights_bytes,
                "padded_words_in_rom": padded_weights_words,
                "bits_per_word": 8,
                "total_bits": padded_weights_bits,
                "total_megabytes": round(padded_weights_bits / (8 * 1024 * 1024), 2),
            },
            "param_rom": {
                "words": param_words,
                "bits_per_word": param_word_bits,
                "total_bits": param_bits,
                "total_megabytes": round(param_bits / (8 * 1024 * 1024), 2),
            },
            "table_roms": {
                "silu_lut_bits": silu_lut_bits,
                "rsqrt_lut_bits": rsqrt_lut_bits,
                "exp2_lut_bits": exp2_lut_bits,
                "rope_table_bits": rope_table_bits,
                "total_bits": tables_total_bits,
                "total_kilobytes": round(tables_total_bits / (8 * 1024), 2),
            },
            "program_rom": {
                "instruction_words": 20,
                "instruction_bits": program_ins_bits,
                "layer_base_words": 270,
                "layer_base_bits": layer_bases_bits,
                "total_bits": program_total_bits,
                "total_kilobytes": round(program_total_bits / (8 * 1024), 2),
            },
            "total_immutable_rom_bits": total_rom_bits,
            "total_immutable_rom_mb": round(total_rom_bits / (8 * 1024 * 1024), 2),
        },
        "sram_accounting": {
            "kv_cache": {
                "max_ctx": 256,
                "total_elements": kv_cache_elements,
                "bits_per_element": 8,
                "total_bits": kv_cache_bits,
                "total_megabytes": round(kv_cache_bits / (8 * 1024 * 1024), 2),
            },
            "activation_buffers": {
                "buffers_breakdown_bits": buffers,
                "total_bits": total_buffers_bits,
                "total_kilobytes": round(total_buffers_bits / (8 * 1024), 2),
            },
            "total_mutable_sram_bits": total_sram_bits,
            "total_mutable_sram_mb": round(total_sram_bits / (8 * 1024 * 1024), 2),
        },
        "totals": {
            "grand_total_memory_bits": grand_total_bits,
            "grand_total_memory_mb": round(grand_total_bits / (8 * 1024 * 1024), 2),
            "rom_share_percent": round((total_rom_bits / grand_total_bits) * 100.0, 2),
            "sram_share_percent": round((total_sram_bits / grand_total_bits) * 100.0, 2),
        },
    }

    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Wrote {out_json}")
    print(f"    ROM:  {report['rom_accounting']['total_immutable_rom_mb']} MB ({report['totals']['rom_share_percent']}%)")
    print(f"    SRAM: {report['sram_accounting']['total_mutable_sram_mb']} MB ({report['totals']['sram_share_percent']}%)")
    print(f"    Total: {report['totals']['grand_total_memory_mb']} MB")


if __name__ == "__main__":
    run_ea3()
