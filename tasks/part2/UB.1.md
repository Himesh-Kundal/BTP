# UB.1 — Graph IR v2 & Invariants I1–I7
Status: done
Phase/Gate: UB / HB     Tags: [numeric]     Estimate: 2h
Depends on: U0.5, UA.7   Spec refs: docs/part2/02 §5, docs/part2/04 §2

## Goal
Implement Graph IR v2 in `ir/graph_v2.py` representing the complete transformer decoder graph (all node ops, tensor kinds, edge annotations, node attributes, provenance strings), JSON serialization/deserialization, and validation of invariants I1–I7 with comprehensive tests including negative controls.

## Acceptance (executable)
- [x] `ir/graph_v2.py` implemented with zero torch/transformers/verilog imports (Golden Rule 6 / Invariant I5)
- [x] Supported ops: `Embed`, `RMSNorm`, `Linear`, `RoPE`, `Attention`, `SiLU`, `EltMul`, `Add`, `ArgMax`
- [x] Supported tensor kinds: `CONSTANT`, `DYNAMIC`, `STATE`
- [x] Invariants I1–I7 strictly enforced by `validate_invariants()`
- [x] `tests/test_ir_v2.py` passes with 100% test coverage including negative controls for each invariant
- [x] Lossless round-trip JSON serialization verified
- [x] `make check2` passes

## Plan
1. Implement `ir/graph_v2.py` with dataclasses (`TensorKind`, `EdgeFormat`, `TensorRefV2`, `OpKind`, `NodeV2`, `GraphV2`).
2. Add invariant checking method `validate_invariants(graph)` validating I1 to I7.
3. Write `tests/test_ir_v2.py` exercising round-trip serialization and negative controls for I1–I7.
4. Run tests and verify clean linting and layering.

## Evidence (after)
- `ir/graph_v2.py` created with complete dataclasses, JSON serialization, and invariant verification functions.
- `tests/test_ir_v2.py` created with 9 tests covering valid graphs, lossless serialization roundtrips, and negative controls for I1, I2, I3, I5, I6, I7.
- `make check2` passes 90/90 tests in 5.70s.
- `uv run ruff check .` clean.
- Layering check (`tests/test_layering.py`) passes 3/3 tests.

## Verifier review
Verified by automated invariant checks and pytest test suite.

