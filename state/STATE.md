# STATE — keep to one page, always current
 
## Handoff note (rewrite at the end of every session)
- Done this session: T6.7 complete. Implemented waveform walkthrough script `scripts/generate_waveforms.py` and documentation `docs/walkthroughs/waveforms.md`. Generated clean VCD dumps for passing transaction (L=1 delay), M1 sign-flip arithmetic fault, and M9 latency protocol fault. Documented cycle-by-cycle ASCII timing diagrams, picosecond event logs, and viva defense Q&A. Full test suite green (200/200 in 90.36s).
- Next action (exact first step): Start T6.G (Gate G6 review, compile `results/gates/G6.md`, draft `docs/walkthroughs/P6.md`, tag `gate-G6`).
- Gotchas / things I learned: Signal event logging directly from VCD confirms synchronous registered behavior; non-blocking signal comparisons cleanly distinguish arithmetic corruptions from timing/latency violations.

## Project
- Start date: 2026-11-01          Current week: 1 of 14
- Final deadline: TBD            Mid-term review: TBD
- Supervisor / reviewer preferences (report format, page limit, demo rules): Standard thesis format (see O-2)
- Institute AI-assistance policy: Full transparency / disclosure in report appendix (see O-6)

## Where we are
- Current phase: P6 (Verification Hardening)                   Current task: T6.7 (done)
- Last green commit: 986e9ca                                  Last gate tag: gate-G5
- Gates passed: G0, G1, G2, G3, G4, G5

## Next 3 tasks
1. T6.G (Gate G6 review, G6.md report, Phase 6 walkthrough P6.md, tag gate-G6)
2. T7.1 (Yosys synthesis flow scripts for Design F and Design P)
3. T7.2 (Fairness protocol checklist implementation)

## Open blockers (see state/BLOCKERS.md)
- none

## Yellow items awaiting the human
- none

## Open decisions (see docs/03 §2)
- O-1 F-vs-P as core, ROM mode as stretch: default applies
- O-2 supervisor format/deadlines: standard defaults apply until specified
- O-3 area technology for Yosys: default = generic gates + ice40 LUT proxy
- O-4 stretch S1: not started until G6
- O-5 harness capabilities: uv package manager + native bash/EDA tools
- O-6 AI-assistance policy: disclosed in appendix

## Environment (filled by T0.1; source of truth is results/env.json)
- iverilog: Icarus Verilog version 12.0 (stable) ()
- vvp: Icarus Verilog runtime version 12.0 (stable) ()
- verilator: Verilator 5.032 2025-01-01 rev (Debian 5.032-1+b2)
- yosys: Yosys 0.52 (git sha1 fee39a3284c90249e1d9684cf6944ffbbcbb8f90)
- python: 3.13.5
- torch: 2.15.0+cpu, transformers: 5.22.1, numpy: 2.5.4, pytest: 9.2.0, ruff: 0.17.3
- Model: SmolLM2-135M, revision pinned

## Budgets (filled when measured; frozen after)
- `make check` runtime: ~97s (full pytest suite)      `make regression` runtime: 22.2s [FROZEN T6.6]
- Full 576×576 layer generation time / RTL size: 0.168s / 7.33 MB (224,750 lines) [FROZEN T4.5]
- Synthesis timeout per run: <s>

## Parking lot (ideas NOT to implement now)
- none
