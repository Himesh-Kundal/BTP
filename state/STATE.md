# STATE (Part II) — one page, always current

## Handoff note (rewrite at the end of every session)
- Done this session: Phase U0 (Audit & Bootstrap) 100% complete. All tasks U0.1–U0.5 passed and Gate H0 approved:
  - U0.1: Part I reproduced, tagged `part1-final`, audited in `results/part2/part1_audit.md`.
  - U0.2: Retrospective authored in `ADR-2-0`, motivating Decision D2-02 (ROM engine).
  - U0.3: `docs/model_facts.md` v2 complete (RoPE theta=100000, tied embeddings, 134.5M params).
  - U0.4: Part II skeleton, Makefile targets (`check2`, `demo2-fast`, `budgets`), and machine budgets in `results/part2/budgets.json` (Verilator 156k cyc/s on 1024 MACs, 11.88 GB flat RTL extrapolated vs 128.25 MB ROM).
  - U0.5: Deterministic tiny random model ($L=2, d=64, H=4, KV=2, V=256$) implemented in `frontend/tiny_model.py`, tested with 100% coverage in `tests/test_tiny_model.py`.
  - U0.G: Gate H0 review document written (`results/gates/H0.md`). `make check2` passes in 3.04s (36 tests), `make check` passes in 146s (234 tests).
- Next action (exact first step): Phase UA (Numerics First) -> Task UA.1 (NumPy fp32 decoder reference).
- Gotchas / things I learned: SmolLM2-135M explicitly sets `rope_theta = 100000.0` (1e5) in `rope_parameters`, not 10,000. Assuming 10k would have caused silent trigonometric divergence.
- Background jobs running (command, log path, expected finish, success criteria): None.

## Project
- Start date: 2026-11-09          Current week: 2 of 14
- Final deadline: End of Semester (W14)            Mid-term review: W7
- Supervisor: Prof. Mrityunjoy Chakraborty (per Part I report).
- Institute AI-assistance policy: Full transparency / disclosure in report appendix and README.
- Part I final tag: `part1-final` (created in U0.1 at `bf06427`)
- Part II Gate tags: `gate-H0`

## Where we are
- Current phase: UA (Numerics First)        Current task: UA.1
- Last green commit: gate-H0                Last gate tag: gate-H0
- Gates passed (Part II): H0

## Next 3 tasks
1. UA.1 (NumPy fp32 decoder in `eval/fp32/` validated against Hugging Face)
2. UA.2 (Evaluation harness: fixed WikiText-2 windows + 20-prompt generation battery)
3. UA.3 (`numerics/ops.py`: integer math library, dyadic scale `(M, s)`, LUT builders)

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
