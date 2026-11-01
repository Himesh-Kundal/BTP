# STATE — keep to one page, always current
 
## Handoff note (rewrite at the end of every session)
- Done this session: T4.5 complete. Implemented backend/module_generator.py with complete Design F hierarchy (all tile cores, reduction adder trees over j, registered wrapper with latency L=1 and synchronous reset). Simulated real SmolLM2 8x16 tile in Icarus with zero mismatch/timing error. Measured and froze full 576x576 layer budgets.
- Next action (exact first step): Start T4.6 (interface conformance test and weight port audit).
- Gotchas / things I learned: 2592 tile cores for full 576x576 generate in 0.17s and 7.33MB; iverilog compiles all 224,750 lines cleanly without warnings.

## Project
- Start date: 2026-11-01          Current week: 1 of 14
- Final deadline: TBD            Mid-term review: TBD
- Supervisor / reviewer preferences (report format, page limit, demo rules): Standard thesis format (see O-2)
- Institute AI-assistance policy: Full transparency / disclosure in report appendix (see O-6)

## Where we are
- Current phase: P4 (Specialization, Tiling, Backend F)   Current task: T4.5 (done)
- Last green commit: c252d12    Last gate tag: gate-G3
- Gates passed: G0, G1, G2, G3

## Next 3 tasks
1. T4.6 (`weight_audit.py`: interface conformance & weight audit)
2. T4.7 (`test_determinism.py`: RTL generation determinism)
3. T4.G (Gate G4 review and RTL checklist audit)

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
- `make check` runtime: ~25.0s      `make regression` runtime: TBD
- Full 576×576 layer generation time / RTL size: 0.168s / 7.33 MB (224,750 lines) [FROZEN T4.5]
- Synthesis timeout per run: <s>

## Parking lot (ideas NOT to implement now)
- none
