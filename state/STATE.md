# STATE (Part II) — one page, always current

## Handoff note (rewrite at the end of every session)
- Done this session: Phase UB (Compiler Extension) 100% complete. Tasks UB.1–UB.7 passed and Gate HB approved:
  - UB.1: Graph IR v2 in `ir/graph_v2.py` with node ops, tensor kinds, edge formats, lossless JSON serialization, and Invariants I1–I7 verified with negative controls.
  - UB.2: Frontend v2 importer in `frontend/importer_v2.py` extracting full structure with cryptographic SHA256 integrity (`artifacts/tiny_model_v2`).
  - UB.3: Format & dyadic scale assignment pass in `compiler/format_assignment.py`, folding $\gamma$, $1/\sqrt{d}$, and per-channel weights into dyadic $(M, s)$ scales.
  - UB.4: ROM packing pass in `compiler/rom_packing.py` generating `weights.hex`, `requant_params.hex`, `tables.hex`, and `address_map.json`.
  - UB.5: Sequencer schedule & microprogram generator in `compiler/schedule.py` with parameterized layer loop and analytical cycle estimates.
  - UB.6: Memory planner in `compiler/memory_planner.py` sizing activation buffers and confirming 2.81 MB KV cache RAM footprint.
  - UB.7: Compiled-artifact software interpreter in `compiler/interpreter.py` executing microprograms directly on packed ROM hex images.
  - UB.G: Gate HB review document written (`results/gates/HB.md`), all 5 Explain-It questions answered. `make check2` passes (101 tests).
- Next action (exact first step): Phase UC (RTL Unit Library) -> Task UC.1 (`rtl/lib/numerics.vh` and `requant` unit).
- Gotchas / things I learned: Parameterizing the sequencer layer loop reduces microprogram ROM storage from thousands of instructions to < 20 instructions by maintaining a layer ROM base address table.
- Background jobs running (command, log path, expected finish, success criteria): None.

## Project
- Start date: 2026-11-09          Current week: 2 of 14
- Final deadline: End of Semester (W14)            Mid-term review: W7
- Supervisor: Prof. Mrityunjoy Chakraborty (per Part I report).
- Institute AI-assistance policy: Full transparency / disclosure in report appendix and README.
- Part I final tag: `part1-final` (created in U0.1 at `bf06427`)
- Part II Gate tags: `gate-H0`, `gate-HA`, `gate-HB`

## Where we are
- Current phase: UC (RTL Unit Library)      Current task: UC.1
- Last green commit: gate-HB                Last gate tag: gate-HB
- Gates passed (Part II): H0, HA, HB

## Next 3 tasks
1. UC.1 (`rtl/lib/numerics.vh` shared macros/functions and `requant` unit)
2. UC.2 (ROM generator, module template, weight audit v2, no-write-path interface test)
3. UC.3 (`rmsnorm` unit bit-exact against `numerics/ops.py` and mutant tests)

## Open blockers (see state/BLOCKERS.md)
- none

## Yellow items awaiting the human
- none

## Open decisions (docs/part2/03 §2)
- O2-1 ROM-backed engine for the full model: default applies (confirmed by D2-02 feasibility study)
- O2-2 quality thresholds: set at HA from measurements
- O2-3 FPGA/ASIC-flow expectations from supervisor: generic gates + ice40 LUT proxy
- O2-4 area technology/liberty: default generic (+ liberty if provided)
- O2-5 evaluation corpus: default WikiText-2 + prompt battery
- O2-6 MAX_CTX: default 256, set at U0.4/UB.6
- O2-7 combined report vs Part II report: default combined
- O2-8 harness capabilities: native Linux EDA (Icarus, Verilator, Yosys) + uv
- O2-9 early start of UA before semester: in progress

## Fallback ladder
- Target: F0 (full 30-layer token-exact). Decision date for ADR-2-F: end of W9.

## Environment and budgets (source of truth is results/env.json and results/part2/budgets.json)
- Tool versions: python 3.13.5, torch 2.15.0+cpu, transformers 5.22.1, numpy 2.5.4, iverilog 12.0, verilator 5.032, yosys 0.52
- Machine: 20 CPU threads, 15.35 GB RAM (9.0 GB available), 598 GB free disk
- Verilator throughput: 156k cycles/s on 1024-MAC array (compile: 4.5s)
- Extrapolated flat spatial RTL: 11.88 GB vs ROM weight storage: 128.25 MB (31.6x reduction)
- `make check2` runtime: 3.04s (36 tests)
- Full test suite: 234 tests passed in 146.39s

## Parking lot (ideas NOT to implement now)
- none
