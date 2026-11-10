# UC.1 — `rtl/lib/numerics.vh` & `requant` Unit
Status: done
Phase/Gate: UC / HC     Tags: [rtl, numeric]     Estimate: 2h
Depends on: UA.7, UB.4   Spec refs: docs/part2/02 §3 (N2-2, N2-3), docs/part2/04 §2

## Goal
Implement `rtl/lib/numerics.vh` containing Verilog-2001 signed functions/macros for `rshift_rnd(x, s)`, `sat(x, min_val, max_val)`, and `requant(acc, M, s)`.
Implement parameterized `rtl/lib/requant.v` supporting dyadic requantization with synchronous handshake and pipeline registers.
Verify against `numerics/ops.py` with exhaustive testing on small bitwidths and random/boundary testing on full INT32/INT48 bitwidths.

## Acceptance (executable)
- [x] `rtl/lib/numerics.vh` implemented in Verilog-2001 with signed functions and macros.
- [x] `rtl/lib/requant.v` implemented with clean linting (`iverilog -g2005 -Wall` and `verilator --lint-only -Wall`).
- [x] `tests/test_rtl_numerics.py` passes exhaustive small-width tests against `numerics/ops.py` (9216 shifts, 1024 saturations).
- [x] Boundary tests (min, max, 0, -1, odd negatives, shift=0, maximum shift, 200 random vectors) pass bit-for-bit.
- [x] `make check2` passes.

## Evidence (after)
- `rtl/lib/numerics.vh` implemented with signed 64-bit and 96-bit functions and macros.
- `rtl/lib/requant.v` parameterized dyadic requantization module with `PIPELINED` option and symmetric saturation.
- `tests/test_rtl_numerics.py` tests 10,240 cases against `numerics/ops.py` with 100% agreement.
- Verilator lint-only passed with 0 warnings.
- `make check2` passes 103/103 tests in 5.17s.
