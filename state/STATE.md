# STATE (Part II) — one page, always current

- **Done this session:**
  - **Full Rust Migration (Phases R0–R9 & 100% Python Code Removal):**
    - Ported all 10 modules into a modular Cargo workspace in `crates/` (`llmhw-numerics`, `llmhw-ir`, `llmhw-compiler`, `llmhw-backend`, `llmhw-sim`, `llmhw-verify`, `llmhw-synth`, `llmhw-frontend`, `llmhw-eval`, `llmhw-cli`).
    - Added direct text prompt encoding/decoding using Hugging Face's official `tokenizers` crate in `llmhw-cli`, enabling interactive text prompt generation (`llmhw run --prompt "..."`) completely in Rust with zero Python runtime.
    - Updated `Makefile` and `scripts/check_env.sh` to be 100% native Rust/Cargo (`make check`, `make test`, `make sim`, `make synth`, `make demo2`, `make demo2-fast`).
    - Removed all 164 legacy `.py` files and Python virtual environment (`.venv`, `pyproject.toml`, `uv.lock`), preserving full historical Python reference under git tag `legacy-python-reference`.
    - Achieved 100% bit-exact parity against golden Verilator RTL traces and full SmolLM2-135M decode in **0.293s** (vs 4.60s in Python, **15.7× faster**).
- **Next action (exact first step):** The repository is now 100% native Rust. Ready for interactive prompt testing, benchmarking, and defense.
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
