# UB.3 — Format and Dyadic Scale Assignment Pass
Status: done
Phase/Gate: UB / HB     Tags: [numeric]     Estimate: 2h
Depends on: UB.1, UA.7   Spec refs: docs/part2/02 §3 (N2-1–N2-7, N2-12), docs/part2/04 §2

## Goal
Implement the format and scale assignment pass in `compiler/format_assignment.py` that applies the frozen numerics configuration (`numerics/config_frozen.json`) to every edge of `GraphV2`, folding $\gamma$, $1/\sqrt{d}$, and per-channel/per-row scales into dyadic $(M, s)$ requantization parameters without any floats in RTL.

## Acceptance (executable)
- [x] `compiler/format_assignment.py` annotates all graph edges with exact `EdgeFormat`
- [x] Folds RMSNorm $\gamma$ into per-channel dyadic scales $(M, s)$
- [x] Folds $1/\sqrt{d}$ into attention scale parameters
- [x] Computes per-channel dyadic requantization parameters $(M, s)$ for all 7 linear projections
- [x] Computes per-row dyadic scales for tied LM head and embedding gather (N2-5, N2-12)
- [x] Range checks against declared bitwidths and saturation limits
- [x] Unit tests in `tests/test_format_assignment.py` pass
- [x] `make check2` passes

## Plan
1. Implement `compiler/format_assignment.py` with `assign_formats_and_scales(graph, config, artifacts)`.
2. Compute $(M, s)$ dyadic multipliers and shifts using `numerics.ops.float_to_dyadic_scale`.
3. Annotate graph tensors and produce `format_table.json`.
4. Write `tests/test_format_assignment.py` verifying mathematical correctness, format adherence, and negative controls.
5. Run tests and verify `make check2`.

## Evidence (after)
- `compiler/format_assignment.py` created with `assign_formats_and_scales` and `FormatAssignmentResult`.
- Folds scales and produces dyadic multipliers and shifts $(M, s)$ for embedding, linear projections, attention, and RMSNorms.
- `tests/test_format_assignment.py` validates format table completeness, non-negativity of $(M, s)$ with $0 \le s \le 62$, and invariant preservation.
- `make check2` passes 94/94 tests in 4.12s.
- `tests/test_layering.py` passes 3/3 tests (no torch/transformers in compiler).

## Verifier review
Verified by automated checks and pytest.

