# STATE (Part II) — one page, always current

## Handoff note (rewrite at the end of every session)
- Done this session:
- Next action (exact first step):
- Gotchas / things I learned:
- Background jobs running (command, log path, expected finish, success criteria):

## Project
- Start date: <YYYY-MM-DD>          Current week: <n> of 14
- Final deadline: <date>            Mid-term review: <date>
- Supervisor: Prof. M. Chakraborty (per Part I report). Part II format, page limit and deadlines: <fill in>
- Institute AI-assistance policy: <fill in>
- Part I final tag: `part1-final` (created in U0.1)

## Where we are
- Current phase: U0            Current task: U0.1
- Last green commit: <hash>    Last gate tag: none (Part I gates G0–G8 are history)
- Gates passed (Part II): none

## Next 3 tasks
1. U0.1 (audit and reproduce Part I)
2. U0.3 (model facts v2)
3. U0.4 (Part II skeleton and budgets)

## Open blockers (see state/BLOCKERS.md)
- none

## Yellow items awaiting the human
- none

## Open decisions (docs/part2/03 §2)
- O2-1 ROM-backed engine for the full model: default applies, **please confirm**
- O2-2 quality thresholds: set at HA from measurements
- O2-3 FPGA/ASIC-flow expectations from supervisor: unanswered
- O2-4 area technology/liberty: default generic (+ liberty if provided)
- O2-5 evaluation corpus: default WikiText-2 + prompt battery
- O2-6 MAX_CTX: default 256, set at U0.4/UB.6
- O2-7 combined report vs Part II report: default combined
- O2-8 harness capabilities: unanswered
- O2-9 early start of UA before semester: unanswered

## Fallback ladder
- Target: F0 (full 30-layer token-exact). Decision date for ADR-2-F: end of W9.

## Environment and budgets (filled by U0.4; source of truth is results/env.json and results/part2/budgets.json)
- Tool versions (python, torch, transformers, iverilog, verilator, yosys): <fill in>
- Machine: CPU cores <n>, RAM <GB>, free disk <GB>
- Verilator speed (cycles/s on ~1000-MAC design): <n>
- Full-model sim estimate (cycles/token × tokens ÷ speed): <n>
- `make check2` runtime: <s>     Verilator full build: <time, RAM>
- RTL bytes per weight (Part I spatial) and extrapolated full-model size: <n>

## Parking lot (ideas NOT to implement now)
-
