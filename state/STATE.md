# STATE (Part II) — one page, always current

## Handoff note (rewrite at the end of every session)
- Done this session: Phase UE (Full-Model Scale-Up) tasks UE.1–UE.4, UE.6, UE.7 executed; UE.5 running in background:
  - UE.1 (`54d507d`): Full SmolLM2-135M ROM packing & top_accel generation (`10.996s`, peak RSS `2.505 GB`, `799.03 MB` bundle). Fixed sequencer 9-bit layer addr, 24-bit weight ROM / 20-bit param ROM addr, 32-bit embed addr, and vectorized ROM packing.
  - UE.2 (`51e31f0`): Verilator full-top build (`3.187s`, `0.261 GB` RSS) and 2-step sim (`825.2 kcycles/s`, `27,220,724` cycles). Fixed SwiGLU `LUT_SIZE=512`, `rsqrt_lut[0]` 16-bit clamp, and interpreter RMSNorm/residual alignment.
  - UE.3 (`73e8167`): Layer-0 trace & token equality across 12 positions (`pos=0..11`, 216 files / 728,064 elements bit-exact, `first_divergence: null`, `-O3` speed `2,265.07 kcycles/s`). Vectorized interpreter (`78x` speedup).
  - UE.4 (`071fa8b`): Full 30-layer decode across 10 prompts $\times$ 16 new tokens (`285` steps) 100% token-exact; `L0`/`L14`/`L29` sampled traces 100% identical (`1,344` files / `2,279,424` elements); mean `14,100,262` cycles/token.
  - UE.5 (`scripts/ue5_long_context.py`): 256-position (`MAX_CTX=256`) Verilator simulation running in background (`task-8282`).
  - UE.6 (`results/part2/ue6_host_demo.json`): Host demo & fp32 HF greedy comparison on `p01..p10` (`mean_match_len = 0`, `0.0%` agreement).
  - UE.7 (`results/part2/quality_final.json`): WikiText-2 evaluation (`fp32_ppl = 89.67`, `int_ppl = 1,130,744.23`, `top1_agreement = 0.16%`) -> **BLOCKER-UE7-01** logged and escalated.
- Next action (exact first step): Await human decision on `BLOCKER-UE7-01` in `state/BLOCKERS.md` (fix `compiler/format_assignment.py` RMSNorm `1/s_act` scale + per-layer calibration and re-run UE.1–UE.7, or proceed with hardware-only Gate HE).
- Gotchas / things I learned: `compiler/format_assignment.py` folds RMSNorm `gamma` without dividing by `s_act = 0.05` (20x under-scaling), causing 97.5% of `L0/attn_norm` activations to quantize to zero and collapsing integer perplexity.
- Background jobs running (command, log path, expected finish, success criteria): `PYTHONPATH=. uv run python scripts/ue5_long_context.py` (`task-8282`, 256-position Verilator run).

## Project
- Start date: 2026-11-09          Current week: 2 of 14
- Final deadline: End of Semester (W14)            Mid-term review: W7
- Supervisor: Prof. Mrityunjoy Chakraborty (per Part I report).
- Institute AI-assistance policy: Full transparency / disclosure in report appendix and README.
- Part I final tag: `part1-final` (created in U0.1 at `bf06427`)
- Part II Gate tags: `gate-H0`, `gate-HA`, `gate-HB`, `gate-HC`, `gate-HD`

## Where we are
- Current phase: UE (Full-model scale-up)      Current task: UE.7 / Gate HE (BLOCKED on BLOCKER-UE7-01)
- Last green commit: 071fa8b (UE.4)           Last gate tag: gate-HD
- Gates passed (Part II): H0, HA, HB, HC, HD

## Next 3 tasks
1. Resolve BLOCKER-UE7-01 (human decision on `compiler/format_assignment.py` fix & re-run vs waive)
2. Complete UE.5 long-context verification (`task-8282`)
3. Gate HE review (`results/gates/HE.md`)

## Open blockers (see state/BLOCKERS.md)
- BLOCKER-UE7-01: Integer model perplexity blow-up on SmolLM2-135M (`int_ppl=1,130,744.23`, `top1=0.16%`). Escalated in `state/BLOCKERS.md`.

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
