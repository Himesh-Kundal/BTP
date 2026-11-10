# UC.8 — `argmax` Unit & Per-Row Scale Handling
Status: done
Phase/Gate: UC / HC     Tags: [rtl, numeric]     Estimate: 2h
Depends on: UC.1         Spec refs: docs/part2/02 §3 (N2-12, N2-13), docs/part2/06 §A4

## Goal
Implement `rtl/lib/argmax.v` performing streaming greedy argmax over vocabulary logits:
1. Normalizes raw accumulators by per-row dyadic scales $(M_v, s_v)$ per N2-12 before comparison.
2. Scans vocabulary indices $0 \dots V-1$ and resolves ties to the **lowest token id** per N2-13.
3. Exposes unit protocol (`start`, `busy`, `done`, `best_token_id`).
Verify with adversarial tests (ties across tokens, scale differences that flip raw accumulator order).

## Acceptance (executable)
- [x] `rtl/lib/argmax.v` implemented and lints clean in `iverilog` and `verilator`.
- [x] Resolves ties to the lowest token ID.
- [x] Normalizes per-row scales correctly, proving that a lower raw accumulator with a larger scale wins over a higher raw accumulator with a smaller scale.
- [x] `tests/test_rtl_argmax.py` passes and is added to `make check2`.
