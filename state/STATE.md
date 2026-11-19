# STATE (Part II) — one page, always current

## Handoff note (rewrite at the end of every session)
- **Done this session:** Completed **Option 4 (`ADR-2-2`)**, **Phase UF (`Gate HF`)**, and **Phase UG (`Gate HG`)**, closing out the entire Part I + Part II project:
  - **ADR-2-2 (Option 4 Resolution of `BLOCKER-UE7-01`):** Aligned `build_rsqrt_mantissa_lut` and `build_exp2_fraction_lut` (`numerics/ops.py`), fixed SwiGLU scale propagation ($s_{\text{swiglu}, l} = 64 \cdot s_{\text{gate}} \cdot s_{\text{up}, l}$), and implemented offline compile-time SmoothQuant per-channel scale migration (`calibrate_and_migrate_artifacts_v2` in `frontend/importer_v2.py`). On 10 WikiText-2 test windows (630 tokens, `results/part2/quality_final.json`), achieved **FP32 PPL = 89.6749 vs INT8 PPL = 107.0031 ($\Delta \text{PPL} = +17.3282 \le 25.0$)**, **70.16% Top-1 ArgMax Agreement ($\ge 35.0\%$)**, and **0.2280 Mean KL Divergence ($\le 0.25$)** (`meets_ha_thresholds: true`, `status: PASS`).
  - **UE.1–UE.7 & Gate HE (`results/gates/HE.md`):** Regenerated `build/smollm2_hw/` (`12.44s`, `2.557 GB` RSS, `799.1 MB`), verified 100% bit-exact Layer-0 traces (`UE.3`: `216` files, `728,064` elements), 100% bit-exact 30-layer full decode across `10/10` prompts $\times 16$ tokens (`UE.4`: `285` steps, `2,279,424` sampled trace elements, `14,100,262` cycles/token), 256-token max-context parity (`UE.5`), and coherent English & Python generation in Verilator RTL (`UE.6`: e.g. `p01` matches FP32 greedy `16/16` tokens `" Berlin. The capital of Italy is Rome. The capital of Spain is Madrid."`; mean match length `4.5` tokens).
  - **UF.1–UF.6 & Gate HF (`results/gates/HF.md`):** Completed unit synthesis (`EA1`: `78,510` generic cells, `3,661` DFFs), MAC grid exploration (`EA2`: `-2.90%` analytical cycle model error), memory accounting (`EA3`: `157.54 MB` ROM `98.15%` vs `2.97 MB` SRAM `1.85%`), flat spatial vs ROM-sequential comparison (`EA4`: `213,022` vs `21,387` cells on $32 \times 32$; full spatial extrapolates to `27.98B` gates), TSMC 28nm HPC+ pre-layout projection (`EA5`: `147.95 mm²` Mid die area, `28.37 tok/s` @ 400 MHz for 128 lanes, `63.37 tok/s` for 512 lanes), and published accelerator comparison (`UF.6`).
  - **UG.1 (`cli.py`, `tests/test_cli.py`, `tasks/part2/UG.1.md`):** Extended unified `llmhw` CLI to v2 (`compile-model`, `run`, `verify --bundle`, `synth --unit`), passing `12/12` CLI tests.
  - **UG.2 (`Makefile`, `scripts/ug2_demo2_fast.py`, `scripts/ug2_demo2.py`, `tasks/part2/UG.2.md`):** Verified turnkey targets `make demo2-fast` (`11.45s`, 100% RTL parity + `21/21` unit & `7/7` system mutants killed) and `make demo2` (`106.16s`, 30-layer SmolLM2-135M Verilator RTL generation `' Berlin. The capital'`, 100% bit-exact with oracle and 4/4 match with FP32).
  - **UG.3 (`README.md`, `docs/limitations.md`, `scripts/ug3_clean_repro.py`, `tasks/part2/UG.3.md`):** Updated documentation and verified clean-directory reproduction in `/tmp/llmhw_clean_repro_*` (`results/part2/ug3_clean_repro.json`, `54.86s`, `status: PASS`).
  - **UG.4 (`docs/whole_project_report.md`, `docs/report_traceability.md`, `tasks/part2/UG.4.md`):** Authored the 12-section combined Part I + Part II thesis report and 59-row numerical traceability table with zero unsourced numbers.
  - **UG.5 & Gate HG (`docs/part2/viva_and_slides.md`, `results/gates/HG.md`, `tasks/part2/UG.5.md`):** Authored 10-slide presentation deck with speaker notes, answers to all 8 Viva Voce questions, AI disclosure, supervisor certificate page, and Gate HG report.
- **Next action (exact first step):** All core Part I (`G0`–`G8`) and Part II (`H0`–`HG`) tasks and gates are **100% COMPLETE**. Ready for thesis submission and Viva Voce defense (optional stretch tasks `V1`–`V4` available if desired).
- **Gotchas / things I learned:** Never run `make check2` (`test_mutation_suite_v2.py`) concurrently with Verilator simulations because `mutate2.py` mutates `rtl/lib/*.v` in-place during mutation testing.
- **Background jobs running:** none (all completed).

## Project
- Start date: 2026-11-09          Current week: 2 of 14
- Final deadline: End of Semester (W14)            Mid-term review: W7
- Supervisor: Prof. Mrityunjoy Chakraborty (per Part I report).
- Institute AI-assistance policy: Full transparency / disclosure in report appendix and README.
- Part I final tag: `part1-final` (created in U0.1 at `bf06427`)
- Part II Gate tags: `gate-H0`, `gate-HA`, `gate-HB`, `gate-HC`, `gate-HD`, `gate-HE`, `gate-HF`, `gate-HG`

## Where we are
- Current phase: COMPLETE (All Phases U0–UG done)    Current task: Gate HG (PASSED)
- Last green commit: a25c348 (main)                 Last gate tag: gate-HG
- Gates passed (Part II): H0, HA, HB, HC, HD, HE, HF, HG

## Next 3 tasks
1. Final thesis printout, human signature on certificate page (`docs/part2/viva_and_slides.md` Part 3), and Viva defense
2. (Optional Stretch) V1–V4 if further exploration is desired

## Open blockers (see state/BLOCKERS.md)
- none (BLOCKER-UE7-01 resolved via ADR-2-2)

## Yellow items awaiting the human
- none (Human physical signature on supervisor certificate page before submission)

## Open decisions (docs/part2/03 §2)
- O2-1 ROM-backed engine for the full model: confirmed and validated
- O2-2 quality thresholds: frozen at HA / ADR-2-2 and verified at HE (70.16% Top-1 agreement, +17.3282 delta PPL, 0.2280 Mean KL)
- O2-3 FPGA/ASIC-flow expectations from supervisor: generic gates + ice40 LUT proxy
- O2-4 area technology/liberty: TSMC 28nm HPC+ analytical projection
- O2-5 evaluation corpus: WikiText-2 + prompt battery
- O2-6 MAX_CTX: 256, verified at UE.5
- O2-7 combined report vs Part II report: combined report (`docs/whole_project_report.md`)
- O2-8 harness capabilities: native Linux EDA (Icarus, Verilator, Yosys) + uv
- O2-9 early start of UA before semester: completed

## Fallback ladder
- Target: F0 (full 30-layer token-exact) — ACHIEVED at Gate HE.

## Environment and budgets (source of truth is results/env.json and results/part2/budgets.json)
- Tool versions: python 3.13.5, torch 2.15.0+cpu, transformers 5.22.1, numpy 2.5.4, iverilog 12.0, verilator 5.032, yosys 0.52
- Machine: 20 CPU threads, 15.35 GB RAM (9.0 GB available), 598 GB free disk
- Full decode speed: 14.1M cycles/token (~2.42 MHz in Verilator)
- Projected ASIC throughput: 28.37 tok/s @ 400 MHz (147.95 mm² die, 128 lanes); 63.37 tok/s @ 512 lanes

## Parking lot (ideas NOT to implement now)
- none
