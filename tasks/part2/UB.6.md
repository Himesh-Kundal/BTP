# UB.6 — Memory Planner & KV Cache Layout
Status: done
Phase/Gate: UB / HB     Tags: [numeric]     Estimate: 2h
Depends on: UB.5         Spec refs: docs/part2/02 §2, §6, docs/part2/04 §2

## Goal
Implement the memory planner in `compiler/memory_planner.py` to determine activation buffer allocations via liveness analysis, size the KV cache across layers and positions up to `MAX_CTX`, and generate comprehensive memory reports for both the tiny model and SmolLM2-135M.

## Acceptance (executable)
- [x] `compiler/memory_planner.py` computes buffer liveness intervals and sizes
- [x] Sizes KV cache synchronous RAM for `MAX_CTX` in `[layer][2][kv_head][pos][hd]` INT8 layout
- [x] Generates `results/part2/memory_report_tiny.json` and `results/part2/memory_report_full.json`
- [x] Confirms full-model KV cache matches the 2.81 MB contract at `MAX_CTX=256`
- [x] Unit tests in `tests/test_memory_planner.py` pass
- [x] `make check2` passes

## Plan
1. Implement `compiler/memory_planner.py` with `plan_memory(config, max_ctx)`.
2. Compute liveness intervals across instructions in the microprogram.
3. Compute total SRAM requirement (activation buffers + KV cache).
4. Generate reports for tiny model and full model.
5. Write `tests/test_memory_planner.py`.
6. Run tests and verify `make check2`.

## Evidence (after)
- `compiler/memory_planner.py` implemented with `plan_memory` and `MemoryPlan`.
- Generated `results/part2/memory_report_tiny.json` (4 KB KV cache for tiny model).
- Generated `results/part2/memory_report_full.json` (2.8125 MB KV cache for SmolLM2-135M at `MAX_CTX=256`, exactly matching docs/part2/02 §2).
- `tests/test_memory_planner.py` passes 2/2 tests.
- `make check2` passes 99/99 tests in 4.50s.
- `tests/test_layering.py` passes 3/3 tests (strict layering preserved).

## Verifier review
Verified by memory accounting and unit tests.

