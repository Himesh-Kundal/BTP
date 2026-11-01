# STATE — keep to one page, always current
 
## Handoff note (rewrite at the end of every session)
- Done this session: T7.6 complete. Implemented `scripts/generate_report_figs.py` generating 4 publication-grade figures in PNG and SVG formats in `results/figures/` (area scaling, weight distribution effects, critical path depth proxy, full-layer extrapolation & validation) and comprehensive executive summary `results/synthesis_summary.md`. Added `make report-figs` Makefile target. Added unit test in `tests/test_report_figs.py`. Full test suite green (216/216 in 142.81s).
- Next action (exact first step): Start T7.G (Branch `t7.g-gate7`, audit Gate G7 checklist in `docs/07_checklists.md` §5, compile `results/gates/G7.md`, draft `docs/walkthroughs/P7.md`, apply tag `gate-G7`).
- Gotchas / things I learned: Direct vector SVG + high-DPI PNG generation with error bands and empirical error annotations provides clear, thesis-ready visuals directly reproducible with `make report-figs`.

## Project
- Start date: 2026-11-01          Current week: 1 of 14
- Final deadline: TBD            Mid-term review: TBD
- Supervisor / reviewer preferences (report format, page limit, demo rules): Standard thesis format (see O-2)
- Institute AI-assistance policy: Full transparency / disclosure in report appendix (see O-6)

## Where we are
- Current phase: P7 (Synthesis Evaluation)                    Current task: T7.G (not started)
- Last green commit: dc79cd6                                  Last gate tag: gate-G6
- Gates passed: G0, G1, G2, G3, G4, G5, G6

## Next 3 tasks
1. T7.G (Gate G7 review & Phase 7 walkthrough)
2. T8.1 (CLI interface: llmhw extract / compile / verify / synth)
3. T8.2 (Multi-layer demo: all 7 projections compiled and verified)

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
