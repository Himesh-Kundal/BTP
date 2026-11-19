# Task UG.3 — README, Architecture & Limitations Documentation, and Clean-Directory Reproduction Test

## Objective
Update top-level repository documentation (`README.md`, `docs/limitations.md`) to reflect the completed Part I + Part II compiler, architecture, CLI v2, and limitations, and verify reproducibility from an isolated clean checkout with no pre-existing `build/` or `artifacts/` directories.

## Implementation
- **`README.md`**: Updated with Part I + Part II claims table (`C1`, `C1'`, `C2'`, `EA1`–`EA5`), Part II full-decoder system architecture diagram, turnkey commands (`make check`, `make check2`, `make demo2-fast`, `make demo2`), and CLI v2 usage examples.
- **`docs/limitations.md`**: Formalized Part I + Part II architectural boundaries (`batch_size=1` greedy decode, host BPE tokenizer, parameterized `MAX_CTX=256`, sequential prefill, immutable ROM weights), numeric trade-offs (offline SmoothQuant migration vs multi-step autoregressive prefix divergence), and synthesis/ASIC projection caveats.
- **`scripts/ug3_clean_repro.py`**: Copies the repository into an isolated `/tmp/llmhw_clean_repro_*` directory excluding `build/` and `artifacts/`, and executes `make env-check`, `make demo2-fast`, and `make synth2` from scratch.

## Evidence
- `results/part2/ug3_clean_repro.json`: Clean-directory reproduction **PASSED** (`status: "PASS"`).

## Status
Status: complete
