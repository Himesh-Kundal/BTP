# STATE — keep to one page, always current
 
## Handoff note (rewrite at the end of every session)
- Done this session: T5.1 complete. Implemented `backend/baseline_p.py` for Design P (with `W_reg` array and serial load port) and `P_bus` variant (logic-only primary bus). Verified Rule N10 width conformance against Design F and linting under iverilog, Verilator, and Yosys check -assert. Regression suite green (134/134).
- Next action (exact first step): Start T5.2 (Testbench support for P weight-load phase with `w.hex` generation and simulation).
- Gotchas / things I learned: TileManifest uses `n_padded` and `k_padded`. Wrapper output bus `y_flat` spans full `acc_bits * n_padded` bits per docs/02 §6 with valid_in register clock gating.

## Project
- Start date: 2026-11-01          Current week: 1 of 14
- Final deadline: TBD            Mid-term review: TBD
- Supervisor / reviewer preferences (report format, page limit, demo rules): Standard thesis format (see O-2)
- Institute AI-assistance policy: Full transparency / disclosure in report appendix (see O-6)

## Where we are
- Current phase: P5 (Baseline P)                              Current task: T5.1 (done)
- Last green commit: fb0b883                                  Last gate tag: gate-G4
- Gates passed: G0, G1, G2, G3, G4

## Next 3 tasks
1. T5.2 (Testbench support for P weight-load phase)
2. T5.3 (Differential verification Y_F == Y_P == Y_ref across >= 5 shapes)
3. T5.G (Gate G5 review)

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
