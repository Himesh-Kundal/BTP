# STATE — keep to one page, always current
 
## Handoff note (rewrite at the end of every session)
- Done this session: Phase 8 & Gate G8 complete. All 9 Gates (G0 through G8) have passed and are tagged. Unified CLI `llmhw` registered and tested (10/10 tests). Flagship multi-layer demo (`make demo`) compiles and simulates all 7 projections in 3.58s with 0 mismatches. Varying-shape demo verified across full $576 \times 576$ layer, non-multiple $19 \times 37$, and asymmetric $32 \times 128$ matrices. Fresh clone reproduction verified in clean directory in 9.23s. Publication-grade `README.md` and `docs/limitations.md` complete. Academic thesis draft `docs/report.md` complete with strict 44-item numerical traceability in `docs/report_traceability.md` verified by CI. Conference presentation skeleton (`docs/presentation_skeleton.md`) and comprehensive viva defense guide (`docs/viva_prep.md`) complete. Full test suite passing 230/230 tests.
- Status: PROJECT RELEASED / DEFENSE READY (Tag `gate-G8`).
- Gotchas / things I learned: Decoupling frontend PyTorch extraction from model-agnostic hardware generation and verification via standardized weight manifests enables complete fresh-clone reproduction without checking large neural network weights into git.

## Project
- Start date: 2026-11-01          Current week: 1 of 14
- Final deadline: Defense Ready   Mid-term review: Completed
- Supervisor / reviewer preferences (report format, page limit, demo rules): Standard thesis format (see O-2)
- Institute AI-assistance policy: Full transparency / disclosure in report appendix and README (see O-6)

## Where we are
- Current phase: RELEASED / DEFENSE READY                     Current task: All tasks complete (T0.1–T8.G)
- Last green commit: gate-G8                                   Last gate tag: gate-G8
- Gates passed: G0, G1, G2, G3, G4, G5, G6, G7, G8

## Key Artifacts & Deliverables
1. Unified CLI: `cli.py`, `llmhw` (`llmhw extract | compile | verify | synth`)
2. Demonstrations: `make demo` (all 7 projections in 3.58s), `scripts/run_varying_shape_demo.py`
3. Synthesis & Evaluation: `results/E1.csv`, `results/E2.csv`, `results/E3.csv`, `results/E5.json`, `results/figures/` (Figs 1–4)
4. Thesis Report: `docs/report.md`, `docs/report_traceability.md` (44 traced items)
5. Defense Preparation: `docs/viva_prep.md`, `docs/presentation_skeleton.md`
6. Release Audit: `results/gates/G8.md`, `docs/walkthroughs/P8.md`

## Open blockers (see state/BLOCKERS.md)
- none

## Yellow items awaiting the human
- none

## Open decisions (see docs/03 §2)
- O-1 F-vs-P as core, ROM mode as stretch: F-vs-P verified and evaluated across all dimensions
- O-2 supervisor format/deadlines: standard defaults applied
- O-3 area technology for Yosys: generic gates + ice40 LUT proxy fully evaluated
- O-4 stretch S1: architectural hooks designed
- O-5 harness capabilities: uv package manager + native bash/EDA tools
- O-6 AI-assistance policy: disclosed in README.md and docs/report.md Appendix E

## Environment (filled by T0.1; source of truth is results/env.json)
- iverilog: Icarus Verilog version 12.0 (stable) ()
- vvp: Icarus Verilog runtime version 12.0 (stable) ()
- verilator: Verilator 5.032 2025-01-01 rev (Debian 5.032-1+b2)
- yosys: Yosys 0.52 (git sha1 fee39a3284c90249e1d9684cf6944ffbbcbb8f90)
- python: 3.13.5
- torch: 2.15.0+cpu, transformers: 5.22.1, numpy: 2.5.4, pytest: 9.2.0, ruff: 0.17.3
- Model: SmolLM2-135M, revision pinned

## Budgets (filled when measured; frozen after)
- `make check` runtime: ~123s (230 pytest cases)        `make demo` runtime: 3.58s
- Full 576×576 layer generation time / RTL size: 0.54s / 30.02 MB (548,223 lines)
- Full 576×576 static weight audit time: 0.85s (331,776 weights)

## Parking lot (ideas NOT to implement now)
- none
