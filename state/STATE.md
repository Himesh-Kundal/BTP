# STATE (Part II) — one page, always current

## Handoff note (rewrite at the end of every session)
- Done this session: Phase UD (Tiny Model End-to-End) 100% complete. Tasks UD.1–UD.6 passed and Gate HD approved:
  - UD.1: Top-level accelerator generator (`backend/top_generator.py`, `rtl/lib/top_accel.v`) composing sequencer, units, ROMs, RAMs, and step interface.
  - UD.2: Testbench generator (`sim/tb_generator.py`) driving step interface, dumping traces, checking `TB2_RESULT` with negative controls.
  - UD.3: Multi-prompt token & trace exactness across 100 random prompts (326 total steps, 82,944 trace elements matched bit-for-bit, sequences up to MAX_CTX=32) against Python integer decoder, and first-divergence diagnosis tool verified on an injected fault.
  - UD.4: Cross-simulator equality: Icarus Verilog (`iverilog`) vs Verilator (`--binary --timing`) with 100% trace and token equality (empty `diff -r`).
  - UD.5: Top-level mutation suite (`results/top_mutation_matrix.json`) covering 7 fault models (causal off, kv-head map, pos off-by-one, rope table shift, residual scale, tie-break, ROM word swap) with 100% kill rate (7/7 killed).
  - UD.6: Determinism and structural audits (`verification/structural_audit.py`, `tests/test_audits_v2.py`): zero expected values in RTL, immutable weights (ROM-only, zero write ports), byte-identical generator and simulation determinism, clean lint across Icarus, Verilator, and Yosys.
  - UD.G: Gate HD review report written (`results/gates/HD.md`), all 5 Explain-It questions answered. `make check2` passes (155 tests).
- Next action (exact first step): Phase UE (Full-Model Scale-Up) -> Task UE.1 (Generate full SmolLM2 ROM images, program, and top; measure generation time/size; fix scalability issues).
- Gotchas / things I learned: Verilator compilation overhead can be bypassed during multi-prompt testing by executing all prompts continuously within a single simulation binary run, speeding up test suites from minutes to seconds.
- Background jobs running (command, log path, expected finish, success criteria): None.

## Project
- Start date: 2026-11-09          Current week: 2 of 14
- Final deadline: End of Semester (W14)            Mid-term review: W7
- Supervisor: Prof. Mrityunjoy Chakraborty (per Part I report).
- Institute AI-assistance policy: Full transparency / disclosure in report appendix and README.
- Part I final tag: `part1-final` (created in U0.1 at `bf06427`)
- Part II Gate tags: `gate-H0`, `gate-HA`, `gate-HB`, `gate-HC`, `gate-HD`

## Where we are
- Current phase: UE (Full-model scale-up)      Current task: UE.1
- Last green commit: gate-HD                Last gate tag: gate-HD
- Gates passed (Part II): H0, HA, HB, HC, HD

## Next 3 tasks
1. UE.1 (Generate full SmolLM2 ROM images, program, and top; measure generation time/size)
2. UE.2 (Verilator build of the full top; record build time, RAM, sim speed)
3. UE.3 (Layer-0 block: trace equality for the first positions and several later positions)

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
