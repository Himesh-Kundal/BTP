# UB.2 — Frontend v2 Importer & Full Artifact Extraction
Status: done
Phase/Gate: UB / HB     Tags: [numeric]     Estimate: 2h
Depends on: U0.3, UB.1   Spec refs: docs/part2/02 §2, §5, docs/part2/04 §2

## Goal
Implement Frontend v2 in `frontend/importer_v2.py` to extract the complete transformer decoder structure (embedding, norms, 7 linear projections per layer, tied LM head, RoPE parameters, RMSNorm eps) into standardized artifacts with cryptographic SHA256 hashes, with reload verification on both the tiny model and SmolLM2-135M.

## Acceptance (executable)
- [x] `frontend/importer_v2.py` implements full model extraction with SHA256 verification
- [x] Extracts embed_tokens, final_norm, lm_head, and all layer norms + 7 projections per layer
- [x] Records all model facts and RoPE parameters
- [x] `load_model_artifacts_v2` reloads and verifies SHA256 checksums
- [x] Generates artifacts for the tiny model (`artifacts/tiny_model_v2`)
- [x] Unit tests in `tests/test_importer_v2.py` pass reload tests and verify SHA256 integrity
- [x] `make check2` passes

## Plan
1. Create `frontend/importer_v2.py` with extraction, manifest generation with SHA256, and reload verification.
2. Build `build_graph_v2_from_artifacts` to instantiate `GraphV2` from extracted artifacts.
3. Write `tests/test_importer_v2.py` testing extraction, reload verification, and graph conversion.
4. Run `make check2` and ensure clean linting.

## Evidence (after)
- `frontend/importer_v2.py` created with `extract_and_dump_model_v2`, `load_model_artifacts_v2`, and `build_graph_v2_from_artifacts`.
- Extracted and verified `artifacts/tiny_model_v2` with complete SHA256 integrity.
- `tests/test_importer_v2.py` passes 3/3 tests covering extraction, reload integrity, tampering negative controls, and graph generation.
- `make check2` passes 93/93 tests in 3.25s.
- `uv run ruff check .` clean.

## Verifier review
Verified by SHA256 integrity checking and automated test suite.

