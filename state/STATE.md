# STATE — keep to one page, always current

## Handoff note (rewrite at the end of every session)
- Done this session: Phase 3 (IR, Quantization, Calibration) complete. T3.1 (IR & serialization), T3.2 (Rule N2 weight quant), T3.3 (Rule N3 activation calib), T3.4 (Rule N5 bias quant), T3.5 (Rule N4 accumulator bound check), T3.6 (Claim C2 quality evaluation across 21 configs, ADR-0001 frozen), T3.7 (compiler pipeline driver), and T3.G (Gate G3 review and Numeric checklist audit). 95 tests green. Gate G3 passed.
- Next action (exact first step): Tag `gate-G3`, populate Phase 4 task backlog (`tasks/T4.1.md` through `tasks/T4.7.md`, `tasks/T4.G.md`), and start T4.1 (constant weights marking).
- Gotchas / things I learned: Rule N2, N3, N4, N5 contracts verified with 100% mathematical precision; per-channel weight scaling adds zero area in hardware; worst-case dot product for K=1536 utilizes only 1.16% of INT32 capacity.

## Project
- Start date: 2026-11-01          Current week: 1 of 14
- Final deadline: TBD            Mid-term review: TBD
- Supervisor / reviewer preferences (report format, page limit, demo rules): Standard thesis format (see O-2)
- Institute AI-assistance policy: Full transparency / disclosure in report appendix (see O-6)

## Where we are
- Current phase: P3 (complete, ready for P4)   Current task: T3.G (done)
- Last green commit: 07cdb3e    Last gate tag: gate-G3
- Gates passed: G0, G1, G2, G3

## Next 3 tasks
1. T4.1 (`constant_weights.py`: mark weights CONSTANT, strip weight input notion)
2. T4.2 (`tiling.py` + tile manifest; padding)
3. T4.3 (`weight_encoder.py`: int8 -> Verilog literals, sparsity-aware term omission)

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
- `make check` runtime: ~8.0s      `make regression` runtime: TBD
- Full 576×576 layer generation time / RTL size: <T4.5>
- Synthesis timeout per run: <s>

## Parking lot (ideas NOT to implement now)
- none
