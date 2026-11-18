# STATE (Part II) — one page, always current

## Handoff note (rewrite at the end of every session)
- **Done this session:** Completed Phase UE (Gate HE) and Phase UF (Gate HF) in their entirety:
  - **ADR-2-2:** Resolved `BLOCKER-UE7-01` via dynamic exponent shift correction in `rmsnorm.v` and per-layer activation calibration in `compiler/format_assignment.py` (Top-1 ArgMax Agreement rose to 66.13%, beating Gate HA 35.0% threshold).
  - **UE.1 (`scripts/ue1_generate_full.py`):** Full 30-layer SmolLM2-135M hardware bundle emitted in `build/smollm2_hw/` (799.11 MB, 6.64s, 2.05 GB peak RSS).
  - **UE.2 (`top_accel.v`):** Verilator compilation (4.45s, 2.42 MHz simulation rate).
  - **UE.3 (`results/part2/ue3_layer0_traces.json`):** 12 positions verified (728,064 elements across 216 files) bit-exact.
  - **UE.4 (`results/part2/full_decode.json`):** 10 prompts $\times$ 16 new tokens across all 30 layers in Verilator (285 decode steps, 2.28M elements, 100% token-exact, 14,100,262 cycles/token).
  - **UE.5 (`results/part2/ue5_long_context.json`):** 256 positions (`MAX_CTX=256`, 4,669,315,072 cycles in 1754s) 100% token-exact and trace-exact.
  - **UE.6 (`results/part2/ue6_host_demo.json`):** Battery evaluation vs FP32 greedy baseline.
  - **UE.7 (`results/part2/quality_final.json`):** Evaluated WikiText-2 test windows tied to RTL equality evidence.
  - **Gate HE (`results/gates/HE.md`):** Complete report authored; all checklist items PASSED.
  - **UF.1 / EA1 (`results/part2/EA1.csv`, `results/part2/EA1.json`):** Synthesized all 12 RTL units and systolic engine for generic gates and iCE40 FPGA LUT-4 cell mapping proxy (24 runs in 606s).
  - **UF.2 / EA2 (`results/part2/EA2.csv`, `results/part2/EA2.json`):** Matrix engine grid exploration (16 to 512 MAC lanes); analytical cycle model validated within **-2.90% error** of measured simulation.
  - **UF.3 / EA3 (`results/part2/EA3.json`):** Comprehensive memory accounting: 157.54 MB ROM (**98.15%**) vs 2.97 MB SRAM (**1.85%**) at `MAX_CTX=256`.
  - **UF.4 / EA4 (`results/part2/EA4.csv`, `results/part2/EA4.json`):** Flat spatial vs ROM-sequential on $32 \times 32$ block ($1,024$ weights): spatial requires 10× more logic gates ($213\text{k}$ vs $21.4\text{k}$) and extrapolates to $>27.9$ Billion gates ($>19,500\,\text{mm}^2$), proving physical necessity of ROM-sequential architecture.
  - **UF.5 / EA5 (`results/part2/EA5.json`):** TSMC 28nm HPC+ pre-layout ASIC projections: $147.95\,\text{mm}^2$ die area (Mid), $28.37\text{ tok/s}$ @ 400 MHz (baseline 128 lanes), scaling to $63.37\text{ tok/s}$ @ 512 lanes. Sensitivity analysis proved die area is 96.5% elastic to ROM density ($\epsilon = 0.965$) and inelastic to logic gates ($\epsilon = 0.001$).
  - **UF.6 (`docs/part2/context_comparison.md`):** Comprehensive architectural matrix comparing Groq, Etched, Tenstorrent, SambaNova, Apple NE with published sources and explicit caveats.
  - **Gate HF (`results/gates/HF.md`):** Complete report authored; all checklist items PASSED.
- **Next action (exact first step):** Execute Phase UG (Integration, Report, Viva): UG.1 (CLI v2 enhancements), UG.2 (`make demo2-fast` & `make demo2`), UG.3 (reproduce tests & README), UG.4 (whole-project report), UG.5 (slides & viva Q&A).
- **Gotchas / things I learned:** Mapping flat unrolled constant-multiplier matrices into 4-LUTs scales superlinearly; $64 \times 64$ spatial hits ABC boolean mapping limits (>10 min), while $32 \times 32$ ($1,024$ weights) synthesizes in 312s for iCE40 and 61s for generic gates.
- **Background jobs running:** none (all completed).

## Project
- Start date: 2026-11-09          Current week: 2 of 14
- Final deadline: End of Semester (W14)            Mid-term review: W7
- Supervisor: Prof. Mrityunjoy Chakraborty (per Part I report).
- Institute AI-assistance policy: Full transparency / disclosure in report appendix and README.
- Part I final tag: `part1-final` (created in U0.1 at `bf06427`)
- Part II Gate tags: `gate-H0`, `gate-HA`, `gate-HB`, `gate-HC`, `gate-HD`, `gate-HE`, `gate-HF`

## Where we are
- Current phase: UG (Integration, report, viva)      Current task: UG.1 (CLI v2)
- Last green commit: 711f3c8 (main)                 Last gate tag: gate-HF
- Gates passed (Part II): H0, HA, HB, HC, HD, HE, HF

## Next 3 tasks
1. UG.1: CLI v2 (`compile-model`, `run`, `verify`, `synth`)
2. UG.2: `make demo2-fast` and `make demo2` end-to-end verification
3. UG.3: README, architecture documentation, reproduce-from-fresh-clone test

## Open blockers (see state/BLOCKERS.md)
- none (BLOCKER-UE7-01 resolved via ADR-2-2)

## Yellow items awaiting the human
- none

## Open decisions (docs/part2/03 §2)
- O2-1 ROM-backed engine for the full model: confirmed and validated
- O2-2 quality thresholds: frozen at HA and verified at HE (66.13% Top-1 agreement)
- O2-3 FPGA/ASIC-flow expectations from supervisor: generic gates + ice40 LUT proxy
- O2-4 area technology/liberty: TSMC 28nm HPC+ analytical projection
- O2-5 evaluation corpus: WikiText-2 + prompt battery
- O2-6 MAX_CTX: 256, verified at UE.5
- O2-7 combined report vs Part II report: default combined
- O2-8 harness capabilities: native Linux EDA (Icarus, Verilator, Yosys) + uv
- O2-9 early start of UA before semester: completed

## Fallback ladder
- Target: F0 (full 30-layer token-exact) — ACHIEVED at Gate HE.

## Environment and budgets (source of truth is results/env.json and results/part2/budgets.json)
- Tool versions: python 3.13.5, torch 2.15.0+cpu, transformers 5.22.1, numpy 2.5.4, iverilog 12.0, verilator 5.032, yosys 0.52
- Machine: 20 CPU threads, 15.35 GB RAM (9.0 GB available), 598 GB free disk
- Full decode speed: 14.1M cycles/token (~2.4 MHz in Verilator)
- Projected ASIC throughput: 28.37 tok/s @ 400 MHz (148 mm² die)

## Parking lot (ideas NOT to implement now)
- none
