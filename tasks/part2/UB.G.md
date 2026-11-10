# UB.G — Gate HB Review & Sign-Off
Status: done
Phase/Gate: UB / HB     Tags: [gate, numeric]     Estimate: 1h
Depends on: UB.1, UB.2, UB.3, UB.4, UB.5, UB.6, UB.7   Spec refs: docs/part2/07 §5 (HB)

## Goal
Conduct the formal Gate HB review, verifying that the extended compiler pipeline (Graph IR v2, Frontend Importer v2, Format Assignment, ROM Packing, Sequencer Schedule, Memory Planner, and Compiled-Artifact Interpreter) satisfies all HB checklist requirements and produces bit-exact representations for both the tiny model and SmolLM2-135M.

## Acceptance (executable)
- [x] Graph IR invariants I1–I7 tested with negative controls; round-trip lossless
- [x] Format assignment pass annotates all edges and derives dyadic $(M, s)$ scales
- [x] ROM packing creates `weights.hex`, `requant_params.hex`, `tables.hex`, `address_map.json` with verified SHA256 hashes
- [x] Schedule/microprogram and memory plan generated for tiny and SmolLM2
- [x] Compiled-artifact interpreter executes microprogram on ROM hex images bit-identically
- [x] `results/gates/HB.md` review document completed
- [x] `make check2` passes (101/101 tests green)

## Plan
1. Audit each Gate HB checklist item against executable evidence.
2. Author `results/gates/HB.md` documenting all findings, parameters, and Explain-It answers.
3. Verify `make check2` and git commit status.

## Evidence (after)
- Authored `results/gates/HB.md` signing off Gate HB.
- `make check2` passes 101/101 tests in 4.46s.
- Layering check passes 3/3 tests (no torch/transformers in compiler).

## Verifier review
Gate HB approved. Compiler extension phase signed off.
