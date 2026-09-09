# STATE — keep to one page, always current

## Handoff note (rewrite at the end of every session)
- Done this session: Phase 2 (Frontend and Real Weights) completed. Pinned model loading (SmolLM2-135M at `93efa2f097d5`), linear layer extraction (all 210 projections categorized across 30 layers), model profiling (`docs/model_facts.md` 100% match with spec, 0 biases), artifact serialization (`weight.npy`, `meta.json` with SHA256), deterministic activation capture (fixed seed 42), mathematical validation (`X @ W.T == module(X)` across all 7 projection types $\le 10^{-5}$ error), Gate G2 report (`results/gates/G2.md`), and walkthrough (`docs/walkthroughs/P2.md`). 58 tests green.
- Next action (exact first step): Tag `gate-G2`, initialize Phase 3 backlog (`tasks/T3.1.md` through `tasks/T3.7.md`, `tasks/T3.G.md`), and start T3.1 (Complete IR and serialization).
- Gotchas / things I learned: SmolLM2-135M has strictly 0 linear biases across all 30 layers; PyTorch `Linear` stores weights as `[out_features, in_features]`, requiring transpose `X @ W.T` for row-major sequence tensors.

## Project
- Start date: 2026-11-01          Current week: 1 of 14
- Final deadline: TBD            Mid-term review: TBD
- Supervisor / reviewer preferences (report format, page limit, demo rules): Standard thesis format (see O-2)
- Institute AI-assistance policy: Full transparency / disclosure in report appendix (see O-6)

## Where we are
- Current phase: P2 (complete, ready for P3)   Current task: T2.G (done)
- Last green commit: 105980d    Last gate tag: gate-G2
- Gates passed: G0, G1, G2

## Next 3 tasks
1. T3.1 (Complete IR dataclasses, tensor/operators/graph, serialization)
2. T3.2 (Symmetric INT8 weight quantization per rule N2)
3. T3.3 (Activation calibration per rule N3)

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
