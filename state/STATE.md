# STATE (Part II) — one page, always current

## Handoff note (rewrite at the end of every session)
- Done this session: Phase UA (Numerics First) 100% complete. Tasks UA.1–UA.9 passed and Gate HA approved:
  - UA.1: NumPy FP32 reference decoder in `eval/fp32/decoder.py` validated against Hugging Face (diff <= 3.29e-5).
  - UA.2: Evaluation harness in `eval/harness.py`, pinned WikiText-2 dataset hash, 20-prompt battery.
  - UA.3: Integer math op library in `numerics/ops.py` (dyadic scale, sat, rshift_rnd, LUT builders).
  - UA.4: Integer decoder oracle in `numerics/decoder.py` (exact bit-level step vs scorer across 50 sequences).
  - UA.5: Per-op quantization ablation (`results/part2/ua5_ablation.csv`), isolating RMSNorm as primary quality contributor.
  - UA.6: Format sweep across widths and LUT sizes (`results/part2/numerics_sweep.csv`).
  - UA.7: Frozen integer config in `numerics/config_frozen.json` and `ADR-2-1_frozen_numerics_configuration.md`.
  - UA.8: Trace diff diagnostic tool in `verification/trace_diff.py` with fault-injection negative controls.
  - UA.9: Hardware cost proxies in `eval/cost_proxies.py` (129.28 MB ROM, 2.81 MB SRAM).
  - UA.G: Gate HA review document written (`results/gates/HA.md`), 5 Explain-It questions answered. `make check2` passes (81 tests).
- Next action (exact first step): Phase UB (Compiler Extension) -> Task UB.1 (Graph IR v2 in `ir/graph_v2.py`).
- Gotchas / things I learned: RMSNorm accounts for almost the entire perplexity delta in quantized transformer inference; using a 16-bit mantissa LUT with dyadic scale completely preserves quality.
- Background jobs running (command, log path, expected finish, success criteria): None.

## Project
- Start date: 2026-11-09          Current week: 2 of 14
- Final deadline: End of Semester (W14)            Mid-term review: W7
- Supervisor: Prof. Mrityunjoy Chakraborty (per Part I report).
- Institute AI-assistance policy: Full transparency / disclosure in report appendix and README.
- Part I final tag: `part1-final` (created in U0.1 at `bf06427`)
- Part II Gate tags: `gate-H0`, `gate-HA`

## Where we are
- Current phase: UB (Compiler Extension)    Current task: UB.1
- Last green commit: gate-HA                Last gate tag: gate-HA
- Gates passed (Part II): H0, HA

## Next 3 tasks
1. UB.1 (Graph IR v2 in `ir/graph_v2.py` representing complete transformer graph with Invariants I1–I7)
2. UB.2 (Frontend importer v2 in `frontend/importer_v2.py` extracting full structure from Hugging Face)
3. UB.3 (Format and dyadic scale assignment pass applying frozen `config_frozen.json` scales)

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
