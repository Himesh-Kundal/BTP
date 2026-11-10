# UB.4 — ROM Packing Pass & Address Map Generation
Status: done
Phase/Gate: UB / HB     Tags: [numeric]     Estimate: 2h
Depends on: UB.3         Spec refs: docs/part2/02 §6, docs/part2/04 §2

## Goal
Implement the ROM packing pass in `compiler/rom_packing.py` that serializes all immutable parameters (tile-ordered INT8 weights using Part I tiling/padding, dyadic requantization parameters, and mathematical lookup tables) into flat Verilog `$readmemh`-compatible `.hex` files and produces a comprehensive `address_map.json` with cryptographic SHA256 hashes.

## Acceptance (executable)
- [x] `compiler/rom_packing.py` generates `weights.hex`, `requant_params.hex`, `tables.hex`
- [x] Weights are zero-padded to $(R, C)$ multiples and stored in engine-streaming tile order
- [x] Lookup tables (RoPE cos/sin, RMSNorm rsqrt, Softmax exp2/reciprocal, SiLU) packed into table ROM
- [x] Dyadic requantization $(M, s)$ packed into parameter ROM
- [x] `address_map.json` maps every layer/op to base address, length, shape, and file SHA256
- [x] Unit tests in `tests/test_rom_packing.py` verify padding, tile reconstruction, and bitwidth invariants
- [x] `make check2` passes

## Plan
1. Implement `compiler/rom_packing.py` with `pack_rom_artifacts(graph, format_result, weights_dict, out_dir, tile_r, tile_c)`.
2. Generate LUTs using `numerics.ops` and write table image.
3. Tile and pack INT8 weights with zero-padding into `weights.hex`.
4. Pack dyadic $(M, s)$ into `requant_params.hex`.
5. Write `address_map.json` with SHA256 hashes.
6. Write `tests/test_rom_packing.py`.
7. Verify `make check2`.

## Evidence (after)
- `compiler/rom_packing.py` implemented with `pack_rom_artifacts`.
- Generates `weights.hex` (padded to $(R, C)$ multiples in tile order), `requant_params.hex` (64-bit $(M, s)$ words), `tables.hex` (RoPE cos/sin, rsqrt, exp2, reciprocal, SiLU), and `address_map.json`.
- `tests/test_rom_packing.py` passes contiguous layout and SHA256 integrity validation.
- `make check2` passes 95/95 tests in 3.46s.
- `tests/test_layering.py` passes 3/3 tests (strict layering preserved).

## Verifier review
Verified by address map validation and unit tests.

