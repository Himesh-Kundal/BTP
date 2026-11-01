# STATE — keep to one page, always current

## Handoff note (rewrite at the end of every session)
- Done this session:
- Next action (exact first step):
- Gotchas / things I learned:

## Project
- Start date: <YYYY-MM-DD>          Current week: <n> of 14
- Final deadline: <date>            Mid-term review: <date>
- Supervisor / reviewer preferences (report format, page limit, demo rules): <fill in, see O-2>
- Institute AI-assistance policy: <fill in, see O-6>

## Where we are
- Current phase: P0            Current task: T0.1
- Last green commit: <hash>    Last gate tag: none
- Gates passed: none

## Next 3 tasks
1. T0.1
2. T0.2
3. T0.3

## Open blockers (see state/BLOCKERS.md)
- none

## Yellow items awaiting the human
- none

## Open decisions (see docs/03 §2)
- O-1 F-vs-P as core, ROM mode as stretch: default applies
- O-2 supervisor format/deadlines: unanswered
- O-3 area technology for Yosys: default = generic gates + ice40 LUT proxy
- O-4 stretch S1: not started until G6
- O-5 harness capabilities: unanswered
- O-6 AI-assistance policy: unanswered

## Environment (filled by T0.1; source of truth is results/env.json)
- python, torch, transformers, iverilog, vvp, verilator, yosys versions: <fill in>
- Model: SmolLM2-135M, revision <pinned hash>

## Budgets (filled when measured; frozen after)
- `make check` runtime: <s>      `make regression` runtime: <s>
- Full 576×576 layer generation time / RTL size: <T4.5>
- Synthesis timeout per run: <s>

## Parking lot (ideas NOT to implement now)
- 
