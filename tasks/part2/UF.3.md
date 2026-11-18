# Task UF.3 — Memory Accounting (EA3)

## Objective
Account for all memory storage in the SmolLM2-135M accelerator at `MAX_CTX = 256`: immutable ROM (weights, parameters, tables, program) vs mutable SRAM (KV cache, activation buffers). Emit `results/part2/EA3.json`.

## Implementation (`scripts/uf3_memory_accounting.py`)
- Analyzed memory structures across the compiled bundle and architecture contracts:
  - Weight ROM: `162,849,792` INT8 words (`155.30 MB`).
  - Parameter ROM: `288,990` 64-bit words (`2.20 MB`).
  - Table ROMs: SiLU, rsqrt, exp2, and RoPE tables (`34.0 KB`).
  - Program ROM: Microcode schedule and base addresses (`1.13 KB`).
  - KV Cache: 30 layers x 3 KV heads x 256 pos x 64 dim x 8 bits x 2 (`2.81 MB`).
  - Activation Buffers: 10 internal buffers (`156.31 KB`).

## Results (`results/part2/EA3.json`)
- **Total Immutable ROM:** `157.54 MB` (**98.15%** of on-chip storage).
- **Total Mutable SRAM:** `2.97 MB` (**1.85%** of on-chip storage).
- **Grand Total Storage:** `160.51 MB`.

## Status
Status: complete
