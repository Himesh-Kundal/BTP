# UC.6 — `swiglu` Unit
Status: done
Phase/Gate: UC / HC     Tags: [rtl, numeric]     Estimate: 2h
Depends on: UC.1         Spec refs: docs/part2/02 §3 (N2-11), docs/part2/06 §A4

## Goal
Implement `rtl/lib/swiglu.v` implementing SwiGLU activation ($\text{sat}(\text{rshift\_rnd}(\text{SiLU}(\text{gate}) \cdot \text{up} \cdot M, s))$) with SiLU lookup table interface, streaming buffers for gate and up inputs, and unit protocol (`start`, `busy`, `done`).
Verify exhaustively over the entire 8-bit SiLU input domain $\times$ representative up values against `numerics/decoder.py` and `numerics/ops.py`.

## Acceptance (executable)
- [x] `rtl/lib/swiglu.v` implemented and lints clean in `iverilog` and `verilator`.
- [x] Pipelined stream execution over gate and up buffers.
- [x] Exhaustive test over all 256 gate values $\times$ representative `up` values passes bit-for-bit against the oracle.
- [x] `tests/test_rtl_swiglu.py` passes and is added to `make check2`.

## Evidence (after)
- `rtl/lib/swiglu.v` implemented with streaming pipeline reading gate, up, and SiLU ROM with dyadic scaling and symmetric saturation.
- Verilator lint-only passed with 0 warnings.
- `tests/test_rtl_swiglu.py` ran 1792 exhaustive cases (256 gate inputs × 7 representative up inputs) bit-exact against Python oracle.
- `make check2` passes 113/113 tests.
