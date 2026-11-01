# STATE — keep to one page, always current

## Handoff note (rewrite at the end of every session)
- Done this session: Phase 1 (Walking skeleton) complete. Minimal LinearOp IR implemented, combinational fixed-weight Verilog-2001 codegen, vector writer, self-checking testbench, simulation runner with negative control, mutation testing suite M1–M3 with 100% kill rate, `make demo-toy` operational, Gate G1 report compiled.
- Next action (exact first step): Tag gate-G1 and plan Phase 2 (T2.1: pinned SmolLM2-135M model loader).
- Gotchas / things I learned: Generated combinational core adheres to explicit signed arithmetic (8'sd, 16'sd, 32'sd) with zero-weight term pruning and zero weight ports.

## Project
- Start date: 2026-11-01          Current week: 1 of 14
- Final deadline: TBD            Mid-term review: TBD
- Supervisor / reviewer preferences (report format, page limit, demo rules): Standard thesis format (see O-2)
- Institute AI-assistance policy: Full transparency / disclosure in report appendix (see O-6)

## Where we are
- Current phase: P1 (ready for P2)   Current task: T1.G (done)
- Last green commit: e028f10    Last gate tag: gate-G1
- Gates passed: G0, G1

## Next 3 tasks
1. T2.1 (Pinned SmolLM2-135M model loader)
2. T2.2 (Architecture parser & 7 projection types extractor)
3. T2.3 (docs/model_facts.md reconciliation)

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
