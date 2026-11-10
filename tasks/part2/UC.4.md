# UC.4 — `rope` Unit & Table ROMs
Status: done
Phase/Gate: UC / HC     Tags: [rtl, numeric]     Estimate: 2h
Depends on: UC.1, UB.4   Spec refs: docs/part2/02 §3 (N2-8), docs/part2/06 §A4

## Goal
Implement `rtl/lib/rope.v` implementing rotate-half pairing $(x_1 \cdot c - x_2 \cdot s, x_2 \cdot c + x_1 \cdot s)$ with Q1.14 fixed-point scaling, symmetric INT8 saturation $[-127, 127]$, table ROM interfaces, and unit protocol (`start`, `busy`, `done`).
Verify bit-for-bit against `numerics/decoder.py:integer_rope` for all positions $< \text{MAX\_CTX}$ and boundary values.

## Acceptance (executable)
- [x] `rtl/lib/rope.v` implemented and lints clean in `iverilog` and `verilator`.
- [x] Rotate-half pairing exactly conforms to `(i, i + hd/2)`.
- [x] Bit-exact agreement with `numerics/decoder.py:integer_rope` for positions $0 \dots \text{MAX\_CTX}-1$ and boundary values.
- [x] `tests/test_rtl_rope.py` passes and is added to `make check2`.

## Evidence (after)
- `rtl/lib/rope.v` implemented with rotate-half pairing and Q1.14 fixed-point coordinate transform.
- Verilator lint-only passed with 0 warnings.
- `tests/test_rtl_rope.py` verified bit-for-bit on 25 position × vector test configurations.
- `make check2` passes 111/111 tests.
