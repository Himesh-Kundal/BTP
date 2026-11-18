# Task UE.1 — Generate Full SmolLM2-135M ROM Images, Program, and Top

## Objective
Generate the complete hardware bundle (`weights_128b.hex`, `requant_params.hex`, `tables_*.hex`, `program.hex`, `layer_bases.hex`, `top_accel.v`) for `HuggingFaceTB/SmolLM2-135M` (`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`), measure generation wall time, peak RAM, and disk sizes, and fix any scalability bottlenecks discovered.

## Scalability Fixes Implemented
1. **Sequencer Layer Index Width (`rtl/lib/sequencer.v`):**
   - `layer_table_addr` was previously 8-bit `{cur_layer[3:0], layer_op_idx}`, which wrapped at layer 16. Parameterized to `[LAYER_W+3:0]` (9 bits for 30 layers) using `cur_layer[LAYER_W-1:0]`.
2. **Top-Level ROM Address Widths (`rtl/lib/top_accel.v`, `backend/top_generator.py`):**
   - Parameterized `ROM_WEIGHTS_ADDR_W` (24 bits for 10,174,464 128-bit weight words), `ROM_PARAMS_ADDR_W` (19–20 bits for 288,990 64-bit requant words), and `ROM_BASES_ADDR_W` (9 bits for 512 depth covering 30×16 layer bases).
   - Fixed `embed_word_addr` calculation in `top_accel.v` to use full 32-bit arithmetic so `tok_in * HIDDEN_SIZE` never overflows 16 bits for `vocab_size = 49152`.
3. **Vectorized Streaming ROM Packing (`compiler/rom_packing.py`, `backend/top_generator.py`):**
   - Replaced nested Python loops over 162.8M weight bytes with NumPy tile transposition + 256-entry ASCII lookup table and chunked binary file writes/hashing.
   - Vectorized `weights_128b.hex` packing via NumPy byte-buffer reshaping.

## Verification & Measurements (`results/part2/ue1_generation.json`)
- **Total generation time:** `10.996s` (`extract`: 3.257s, `graph_build`: 1.963s, `emit_bundle`: 5.776s)
- **Peak RSS memory:** `2.505 GB` (within `8.97 GB` available RAM budget)
- **Total HW bundle size:** `799.03 MB` (`weights_128b.hex`: 335.76 MB, `weights.hex`: 488.37 MB, `top_accel.v`: 27.29 KB)
- **ROM Word Counts:**
  - `weights_8b_words`: 162,791,424
  - `weights_128b_words`: 10,174,464
  - `requant_params_64b_words`: 288,990
  - `program_128b_words`: 20

## Status
Status: complete
Commit: UE.1: full SmolLM2-135M ROM and top generation with scalability fixes
