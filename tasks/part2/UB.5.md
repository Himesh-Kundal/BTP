# UB.5 — Sequencer Schedule & Microprogram Generator
Status: done
Phase/Gate: UB / HB     Tags: [numeric]     Estimate: 2h
Depends on: UB.4         Spec refs: docs/part2/02 §6, docs/part2/04 §2

## Goal
Implement the schedule and microprogram generator in `compiler/schedule.py` that emits the static sequencer program for the model accelerator, including the layer loop, per-layer ROM base table, binary program ROM image (`program.hex`), and analytical cycle estimates per op and per decoded token.

## Acceptance (executable)
- [x] `compiler/schedule.py` generates `program.hex`, `schedule.json`, and layer base tables
- [x] Microprogram instructions define opcode, buffers, dimensions, ROM bases, and flags
- [x] Layer-loop architecture allows a compact program (tens of instructions rather than thousands)
- [x] Analytical cycle estimates computed per op and per decoded token
- [x] Cycle estimate conforms to linear engine and unit latency bounds
- [x] Unit tests in `tests/test_schedule.py` pass
- [x] `make check2` passes

## Plan
1. Implement `compiler/schedule.py` with `Opcode`, `Instruction`, `Microprogram`, and `generate_schedule()`.
2. Compute cycle estimates per op based on engine $(R, C)$ and unit pipelines.
3. Emit `program.hex` and `schedule.json`.
4. Write `tests/test_schedule.py`.
5. Run tests and verify `make check2`.

## Evidence (after)
- `compiler/schedule.py` implemented with `Opcode`, `Instruction`, `Microprogram`, and `generate_schedule`.
- Produces compact 128-bit instruction words in `program.hex` and structured `schedule.json`.
- Layer loop compresses the decoder sequencer program to < 20 instructions.
- Analytical cycle formulas model linear streaming, RMSNorm pipelines, RoPE, Attention, and SwiGLU.
- `tests/test_schedule.py` passes 2/2 tests.
- `make check2` passes 97/97 tests in 3.42s.
- `tests/test_layering.py` passes 3/3 tests (strict layering preserved).

## Verifier review
Verified by microprogram validation and unit tests.

