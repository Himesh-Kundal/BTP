# STATE — keep to one page, always current
 
## Handoff note (rewrite at the end of every session)
- Done this session: Gate G6 complete and passed! Compiled Gate G6 report `results/gates/G6.md`, drafted Phase 6 walkthrough `docs/walkthroughs/P6.md`, tagged repository `gate-G6`. All 6 Gate G6 checklist criteria audited PASS (L1 matrix, real L2/L3 slices across 7 types at 3 depths, 100% M1-M9 mutation kill rate, Verilator clean & cross-sim parity, regression runner 22.2s budget frozen, waveform walkthrough). Full test suite green (200/200 in 90.36s).
- Next action (exact first step): Start Phase 7 / Task T7.1 (Yosys synthesis flow scripts for Design F, Design P, and Design P_bus; parse `stat` and `ltp` to JSON).
- Gotchas / things I learned: Verification hardening is complete and ironclad. Now transitioning into synthesis evaluation where fairness protocol (Rule N10, identical flags/widths/targets) is critical.

## Project
- Start date: 2026-11-01          Current week: 1 of 14
- Final deadline: TBD            Mid-term review: TBD
- Supervisor / reviewer preferences (report format, page limit, demo rules): Standard thesis format (see O-2)
- Institute AI-assistance policy: Full transparency / disclosure in report appendix (see O-6)

## Where we are
- Current phase: P7 (Synthesis Evaluation)                    Current task: T7.1
- Last green commit: 8d31c76                                  Last gate tag: gate-G6
- Gates passed: G0, G1, G2, G3, G4, G5, G6

## Next 3 tasks
1. T7.1 (Yosys synthesis flow scripts for Design F and Design P, generic gates + ice40 LUT proxy, parse stat/ltp to JSON)
2. T7.2 (Fairness protocol checklist implementation in script)
3. T7.3 (Experiments E1 and E3: area vs tile size, depth proxy, with >= 5 weight seeds)

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
