# STATE (Part II) — one page, always current

## Handoff note (rewrite at the end of every session)
- Done this session: Task U0.1 (Audit and reproduce Part I) completed. Verified clean `make check` (230/230 passed in 146.76s), `make sim` (4 vectors, 0 mismatches), `make regression` (124/124 tests across 7 suites in 23.17s), and `make demo` (all 7 Layer 0 projections bit-exact in 3.94s). Performed independent fresh-clone reproduction in `/tmp/fresh_clone_test` (230/230 passed). Tagged `part1-final` at `bf06427`. Archived Part I state to `state/archive/part1_STATE.md`. Generated comprehensive audit report `results/part2/part1_audit.md` re-deriving arithmetic shares and core claims.
- Next action (exact first step): Task U0.2 (Retrospective and process trim) & Task U0.3 (Model facts v2).
- Gotchas / things I learned: Model weight artifacts under `artifacts/` are properly gitignored; in a fresh clone, `./llmhw extract --layer 14,29 --proj all` deterministically recreates the multi-depth test fixtures from the pinned Hugging Face revision.
- Background jobs running (command, log path, expected finish, success criteria): None.

## Project
- Start date: 2026-11-09          Current week: 1 of 14
- Final deadline: End of Semester (W14)            Mid-term review: W7
- Supervisor: Prof. Mrityunjoy Chakraborty (per Part I report).
- Institute AI-assistance policy: Full transparency / disclosure in report appendix and README.
- Part I final tag: `part1-final` (created in U0.1 at `bf06427`)

## Where we are
- Current phase: U0            Current task: U0.1 (Complete) -> U0.2
- Last green commit: bf06427    Last gate tag: part1-final (Part I gates G0–G8 are history)
- Gates passed (Part II): none

## Next 3 tasks
1. U0.2 (retrospective: 3 things that worked, 3 that slowed us down, trim unused scripts)
2. U0.3 (model facts v2: audit SmolLM2-135M config, weights, tie embedding, shapes)
3. U0.4 (Part II skeleton, directory structure, makefile targets, and budgets)

## Open blockers (see state/BLOCKERS.md)
- none

## Yellow items awaiting the human
- none

## Open decisions (docs/part2/03 §2)
- O2-1 ROM-backed engine for the full model: default applies (confirmed)
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
- Machine: Linux 6.12.113+deb13-amd64 x86_64
- Part I test suite: 230 tests passed in 146.76s
- Part I regression: 124 tests passed in 23.17s
- Part I demo runtime: 3.94s

## Parking lot (ideas NOT to implement now)
- none
