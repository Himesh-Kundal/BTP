# UD.1 — Top-Level Generator & Accelerator Composition
Status: done
Phase/Gate: UD / HD     Tags: [rtl]     Estimate: 4h
Depends on: UC.12, UB.7   Spec refs: docs/part2/02 §1, §6, docs/part2/04 §2

## Goal
Implement the top-level transformer decoder accelerator (`rtl/lib/top_accel.v` / `backend/top_generator.py`) composing:
1. Microprogram sequencer (`sequencer.v`)
2. Non-linear and datapath units (`embed_gather.v`, `rmsnorm.v`, `linear_engine.v`, `rope.v`, `attention.v`, `residual_add.v`, `swiglu.v`, `argmax.v`)
3. Synchronous dynamic memories (`kv_cache.v`, activation buffers 0..9)
4. Immutable ROMs (`rom_weights`, `rom_requant_params`, `rom_tables`, `rom_program`, `rom_layer_bases`)
5. Step control interface (`clk, rst_n, seq_reset, step_valid, step_ready, tok_in[15:0], out_valid, tok_out[15:0]`)

## Acceptance (executable)
- [x] Top-level RTL generated and cleanly structured in Verilog-2001 (`rtl/lib/top_accel.v` and `backend/top_generator.py`).
- [x] Top-level lints clean under `iverilog -g2005 -Wall` with zero errors.
- [x] Top-level lints clean under `verilator --lint-only -Wall` with zero errors and zero warnings.
- [x] Top-level passes Yosys synthesis check (`hierarchy -check; proc; check -assert`) with 0 reported problems.
- [x] `tests/test_top_generator.py` passes end-to-end bundle generation and lint suite.
- [x] `make check2` passes 137/137 tests in 18.97s.

## Evidence (after)
- `rtl/lib/top_accel.v` integrates all UC units, 10 dynamic activation buffers, 2-layer synchronous KV cache, and 9 immutable ROM instances behind the standard step interface.
- `backend/top_generator.py` provides `emit_hardware_bundle` and `generate_top_accel_verilog` automating full bundle compilation (hex images, address map, microprogram, and configured top).
- `tests/test_top_generator.py` validates hardware bundle emission and cross-tool linting (iverilog, verilator, and yosys) on the tiny model.
- `make check2` passes 137/137 tests green.
