# UC.12 — `sequencer` & Microprogram Dispatcher
Status: done
Phase/Gate: UC / HC     Tags: [rtl, sequencer]     Estimate: 3h
Depends on: UB.5, UC.1–UC.11   Spec refs: docs/part2/02 §6, docs/part2/04 §UC.12

## Goal
Implement `rtl/lib/sequencer.v`, the microcode execution engine of the SmolLM2 decoder accelerator:
1. Executes static 128-bit instructions from `program.hex` holding the decoded microprogram.
2. Unpacks instruction fields: `opcode[7:0]`, `src_buf[7:0]`, `dst_buf[7:0]`, `flags[7:0]`, `rom_weight_base[31:0]`, `rom_param_base[31:0]`, `dim0[15:0]`, `dim1[15:0]`.
3. Manages compact layer loop: repeats `[layer_loop_start .. layer_loop_end]` for `num_layers` iterations, providing `cur_layer` and selecting layer ROM bases.
4. Unit protocol dispatch: drives `start` pulse to appropriate unit (`embed_gather`, `rmsnorm`, `linear_engine`, `rope`, `attention`, `residual_add`, `swiglu`, `argmax`) and waits for `done`.
5. Verifies op dispatch order against `compiler/interpreter.py` on the tiny model schedule.

## Acceptance (executable)
- [x] `rtl/lib/sequencer.v` implemented and lints clean in `iverilog -g2005 -Wall` and `verilator --lint-only -Wall`.
- [x] Program ROM reading and 128-bit instruction field decoding verified.
- [x] Layer loop execution verified: prologue (embed) -> N layer iterations -> epilogue (final_norm, lm_head, argmax, halt).
- [x] Exact op execution sequence matches `compiler/interpreter.py` step for step.
- [x] Unit test `tests/test_rtl_sequencer.py` passes and is added to `make check2`.

## Evidence
- `tests/test_rtl_sequencer.py` passes 2/2 tests:
  1. `test_sequencer_execution_order_vs_interpreter`: 34/34 ops matched step-for-step across opcodes, layers, source/destination buffers, and dimensions.
  2. `test_sequencer_single_layer_loop`: multi-layer loop index iteration and next token capture verified.
- `verilator --lint-only -Wall` and `iverilog -g2005 -Wall` 100% clean with 0 warnings.
- Added to `Makefile` `check2`. Pass recorded in `make check2` (135/135 tests green).

