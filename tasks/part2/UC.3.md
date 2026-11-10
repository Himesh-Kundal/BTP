# UC.3 — `rmsnorm` Unit
Status: done
Phase/Gate: UC / HC     Tags: [rtl, numeric]     Estimate: 3h
Depends on: UC.1, UA.8   Spec refs: docs/part2/02 §3 (N2-6), docs/part2/06 §A4

## Goal
Implement `rtl/lib/rmsnorm.v` conforming to unit protocol (`start`, `busy`, `done`), sum-of-squares accumulator, base-2 normalization, rsqrt mantissa lookup, gamma channel scaling, and dyadic requantization to INT8.
Verify bit-for-bit against `numerics/decoder.py:integer_rmsnorm` on random vectors and adversarial edge cases (all zeros, tiny values, huge values, equal elements).

## Acceptance (executable)
- [x] `rtl/lib/rmsnorm.v` implemented with unit protocol and clean linting in `iverilog` and `verilator`.
- [x] Bit-exact agreement with `numerics/decoder.py:integer_rmsnorm` across all test vectors.
- [x] Adversarial test cases pass (zeros, single nonzero, saturation, equal elements).
- [x] `tests/test_rtl_rmsnorm.py` passes and is added to `make check2`.

## Evidence (after)
- `rtl/lib/rmsnorm.v` implemented with 2-stage synchronous pipeline for stream reading, sum-of-squares accumulation, base-2 rsqrt mantissa normalization, and dyadic requantization.
- Verilator lint-only passed with 0 warnings.
- `tests/test_rtl_rmsnorm.py` verified bit-for-bit against Python oracle on all adversarial and random test vectors.
- `make check2` passes 110/110 tests in 4.82s.
