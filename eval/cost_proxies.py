"""Hardware Cost Proxies Calculator for Frozen Numerics Configuration (Task UA.9)."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "results" / "part2" / "cost_proxies.json"


def compute_cost_proxies() -> dict:
    # Model dimensions
    layers = 30
    d = 576
    d_ff = 1536
    n_q = 9
    n_kv = 3
    hd = 64
    vocab = 49152
    max_ctx = 256

    # 1. Weight ROM Bits
    layer_weights = layers * (
        (n_q * hd) * d +  # q_proj
        (n_kv * hd) * d + # k_proj
        (n_kv * hd) * d + # v_proj
        d * (n_q * hd) +  # o_proj
        d_ff * d +        # gate_proj
        d_ff * d +        # up_proj
        d * d_ff          # down_proj
    )
    tied_embed_weights = vocab * d
    total_weights = layer_weights + tied_embed_weights

    rom_weight_bits = total_weights * 8
    rom_weight_bytes = total_weights

    # 2. Requantization Parameters ROM Bits
    layer_out_channels = (n_q * hd) + (n_kv * hd) + (n_kv * hd) + d + d_ff + d_ff + d
    total_requant_words = (layers * layer_out_channels) + vocab
    requant_word_bits = 32 + 6  # 32-bit M + 6-bit s
    rom_requant_bits = total_requant_words * requant_word_bits
    rom_requant_bytes = (rom_requant_bits + 7) // 8

    # 3. Lookup Table ROM Bits
    rsqrt_lut_bits = 256 * 16
    exp2_lut_bits = 256 * 16
    recip_lut_bits = 256 * 16
    silu_lut_bits = 512 * 8
    rope_tab_bits = max_ctx * (hd // 2) * 16 * 2  # cos + sin
    gamma_bits = (layers * 2 + 1) * d * 16

    rom_lut_bits = rsqrt_lut_bits + exp2_lut_bits + recip_lut_bits + silu_lut_bits + rope_tab_bits + gamma_bits
    rom_lut_bytes = (rom_lut_bits + 7) // 8

    total_rom_bits = rom_weight_bits + rom_requant_bits + rom_lut_bits
    total_rom_mb = round(total_rom_bits / (8 * 1024 * 1024), 2)

    # 4. SRAM Dynamic Memory (KV cache + Activation buffers)
    kv_cache_bytes = 2 * layers * n_kv * max_ctx * hd * 1  # 2 for K and V, INT8
    kv_cache_kb = round(kv_cache_bytes / 1024, 2)

    act_buffer_bytes = (d * 3) + (d_ff * 1) + (max_ctx * 2)  # residual + swiglu + softmax
    total_sram_kb = round((kv_cache_bytes + act_buffer_bytes) / 1024, 2)

    # 5. Arithmetic Datapath Widths
    mac_multiplier_bits = [8, 8]
    mac_accumulator_bits = 32
    requant_multiplier_bits = [32, 31]
    residual_adder_bits = 24

    return {
        "model": "SmolLM2-135M",
        "frozen_config": "INT8_DYADIC_RES24_FROZEN",
        "max_ctx": max_ctx,
        "rom": {
            "weight_bits": rom_weight_bits,
            "weight_bytes": rom_weight_bytes,
            "weight_mb": round(rom_weight_bytes / (1024 * 1024), 2),
            "requant_param_bits": rom_requant_bits,
            "requant_param_kb": round(rom_requant_bytes / 1024, 2),
            "lookup_table_bits": rom_lut_bits,
            "lookup_table_kb": round(rom_lut_bytes / 1024, 2),
            "total_rom_bits": total_rom_bits,
            "total_rom_mb": total_rom_mb,
        },
        "sram": {
            "kv_cache_bytes": kv_cache_bytes,
            "kv_cache_kb": kv_cache_kb,
            "act_buffer_bytes": act_buffer_bytes,
            "total_sram_kb": total_sram_kb,
        },
        "datapath": {
            "mac_multiplier_bits": mac_multiplier_bits,
            "mac_accumulator_bits": mac_accumulator_bits,
            "widest_multiplier_bits": requant_multiplier_bits,
            "residual_stream_bits": residual_adder_bits,
        },
    }


def main():
    proxies = compute_cost_proxies()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(proxies, indent=2))
    print(f"[+] Hardware cost proxies written to: {OUTPUT_FILE}")
    print(f"    Total ROM: {proxies['rom']['total_rom_mb']} MB (Weights: {proxies['rom']['weight_mb']} MB, Requant: {proxies['rom']['requant_param_kb']} KB, Tables: {proxies['rom']['lookup_table_kb']} KB)")
    print(f"    Total SRAM: {proxies['sram']['total_sram_kb']} KB (KV Cache @ MAX_CTX={proxies['max_ctx']}: {proxies['sram']['kv_cache_kb']} KB)")
    print(f"    Widest Multiplier: {proxies['datapath']['widest_multiplier_bits'][0]} x {proxies['datapath']['widest_multiplier_bits'][1]} bits")


if __name__ == "__main__":
    main()
