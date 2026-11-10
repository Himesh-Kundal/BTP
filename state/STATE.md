# STATE (Part II) — one page, always current

## Handoff note (rewrite at the end of every session)
- Done this session: Phase UC (RTL Unit Library) 100% complete. Tasks UC.1–UC.13 passed and Gate HC approved:
  - UC.1: `rtl/lib/numerics.vh` & `rtl/lib/requant.v` with rounding and saturation.
  - UC.2: `rtl/lib/rom_sync.v`, `backend/rom_generator.py`, and `verification/weight_audit_v2.py` with immutable weights audit.
  - UC.3: `rtl/lib/rmsnorm.v` bit-exact against `numerics/ops.py` integer oracle.
  - UC.4: `rtl/lib/rope.v` rotary position embedding unit with Q0.14 table lookups.
  - UC.5: `rtl/lib/softmax.v` base-2 exponential decomposition and normalization unit.
  - UC.6: `rtl/lib/swiglu.v` MLP activation unit with 512-word SiLU LUT.
  - UC.7: `rtl/lib/residual_add.v` saturating adder & `rtl/lib/embed_gather.v` token gather unit.
  - UC.8: `rtl/lib/argmax.v` scale-normalized token extractor with strict lowest-ID tie breaking.
  - UC.9: `rtl/lib/kv_cache.v` dual-port INT8 synchronous RAM & `rtl/lib/kv_addr_gen.v`.
  - UC.10: `rtl/lib/linear_engine.v` R×C MAC array with ROM streaming and requantization.
  - UC.11: `rtl/lib/attention.v` datapath with GQA and causal historical context loop.
  - UC.12: `rtl/lib/sequencer.v` microcode engine matching `compiler/interpreter.py` op order.
  - UC.13: Mutation suite v2 in `verification/mutate2.py` achieving 100.0% kill rate (21/21 mutants killed M10–M30).
  - UC.G: Gate HC review report written (`results/gates/HC.md`), all 5 Explain-It questions answered. `make check2` passes (136 tests).
- Next action (exact first step): Phase UD (Tiny Model End-to-End) -> Task UD.1 (Top-level generator: composition, ROM instances, step interface).
- Gotchas / things I learned: Softmax expects synchronous registered read latency from memory; matching pipeline register delays between memory reads and accumulators guarantees zero-bubble throughput and bit-exact equivalence.
- Background jobs running (command, log path, expected finish, success criteria): None.

## Project
- Start date: 2026-11-09          Current week: 2 of 14
- Final deadline: End of Semester (W14)            Mid-term review: W7
- Supervisor: Prof. Mrityunjoy Chakraborty (per Part I report).
- Institute AI-assistance policy: Full transparency / disclosure in report appendix and README.
- Part I final tag: `part1-final` (created in U0.1 at `bf06427`)
- Part II Gate tags: `gate-H0`, `gate-HA`, `gate-HB`, `gate-HC`

## Where we are
- Current phase: UD (Tiny model end to end)      Current task: UD.1
- Last green commit: gate-HC                Last gate tag: gate-HC
- Gates passed (Part II): H0, HA, HB, HC

## Next 3 tasks
1. UD.1 (Top-level generator: composition, ROM instances, step interface)
2. UD.2 (Testbench generator: feed prompt tokens, generate N tokens, dump traces, summary line)
3. UD.3 (Token-exact and trace-exact on >= 100 random prompts incl. sequences up to MAX_CTX)

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
