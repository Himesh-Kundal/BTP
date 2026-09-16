# STATE — keep to one page, always current

## Handoff note (rewrite at the end of every session)
- Done this session: Gate G2 passed (`gate-G2`). Phase 3 underway: T3.1 completed (complete IR specification, serialization, artifact loading). T3.2 completed (symmetric INT8 weight quantization per rule N2). T3.3 completed (static activation calibration and symmetric INT8 quantization per rule N3). T3.4 completed (synthetic bias quantization per rule N5). T3.5 completed: accumulator bound analysis pass per rule N4 (`compiler/accumulator.py`), worst-case bound $K \cdot 128 \cdot 127 + |b_q|_{\max} < 2^{31}$ enforced, minimal bitwidth `acc_bits_min` annotated. 87 tests green.
- Next action (exact first step): Start T3.6 (Quantization quality evaluation Claim C2 and thresholds) on branch `t3.6-quant-quality`.
- Gotchas / things I learned: SmolLM2 dimensions ($K=576$ requiring 25 bits, $K=1536$ requiring 26 bits) easily fit into 32-bit signed accumulators without overflow risk; $K \ge 132105$ triggers overflow rejection.

## Project
- Start date: 2026-11-01          Current week: 1 of 14
- Final deadline: TBD            Mid-term review: TBD
- Supervisor / reviewer preferences (report format, page limit, demo rules): Standard thesis format (see O-2)
- Institute AI-assistance policy: Full transparency / disclosure in report appendix (see O-6)

## Where we are
- Current phase: P3 (IR, quantization, calibration)   Current task: T3.5 (done, ready for T3.6)
- Last green commit: 3ebee29    Last gate tag: gate-G2
- Gates passed: G0, G1, G2

## Next 3 tasks
1. T3.6 (Quantization quality evaluation Claim C2 and thresholds)
2. T3.7 (End-to-end compiler pipeline driver)
3. T3.G (Gate G3 review and verifier check)

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
