# STATE — keep to one page, always current

## Handoff note (rewrite at the end of every session)
- Done this session: Phase 0 (Bootstrap) complete. Environment locked with uv; EDA tools installed (iverilog, verilator, yosys); independent integer oracle implemented with 9 hand-verified test cases; architectural layering enforced; state tracking initialized; Gate G0 report compiled.
- Next action (exact first step): Tag gate-G0 and start Phase 1 (T1.1: minimal LinearOp IR dataclass).
- Gotchas / things I learned: Pinned PyTorch CPU index in pyproject.toml ensures minimal environment footprint on CPU-only machines.

## Project
- Start date: 2026-11-01          Current week: 1 of 14
- Final deadline: TBD            Mid-term review: TBD
- Supervisor / reviewer preferences (report format, page limit, demo rules): Standard thesis format (see O-2)
- Institute AI-assistance policy: Full transparency / disclosure in report appendix (see O-6)

## Where we are
- Current phase: P0 (ready for P1)   Current task: T0.G (done)
- Last green commit: da904ef    Last gate tag: gate-G0
- Gates passed: G0

## Next 3 tasks
1. T1.1 (Minimal LinearOp IR dataclass)
2. T1.2 (Codegen v0: combinational fixed-weight core)
3. T1.3 (Vector writer and testbench generator)

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
- `make check` runtime: ~0.1s      `make regression` runtime: TBD
- Full 576×576 layer generation time / RTL size: <T4.5>
- Synthesis timeout per run: <s>

## Parking lot (ideas NOT to implement now)
- none
