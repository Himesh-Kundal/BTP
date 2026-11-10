# UC.7 — `residual_add` & `embed_gather` Units
Status: done
Phase/Gate: UC / HC     Tags: [rtl, numeric]     Estimate: 2h
Depends on: UC.1         Spec refs: docs/part2/02 §3 (N2-4, N2-5), docs/part2/06 §A4

## Goal
Implement:
1. `rtl/lib/residual_add.v`: Streaming 24-bit saturating addition ($h = \text{sat}(h + \delta, \text{RES\_MIN}, \text{RES\_MAX})$) with unit protocol.
2. `rtl/lib/embed_gather.v`: Embedding table gather ($h_i = \text{sat}(\text{rshift\_rnd}(W_{t,i} \cdot M, s), \text{RES\_MIN}, \text{RES\_MAX})$) from weight ROM with unit protocol.
Verify bit-for-bit against `numerics/decoder.py` including positive/negative saturation edge cases.

## Acceptance (executable)
- [x] `rtl/lib/residual_add.v` implemented, lints clean in `iverilog` and `verilator`.
- [x] `rtl/lib/embed_gather.v` implemented, lints clean in `iverilog` and `verilator`.
- [x] Saturating edge cases (overflow beyond +8388607, underflow beyond -8388608) strictly verified.
- [x] `tests/test_rtl_residual_embed.py` passes and is added to `make check2`.

## Evidence (after)
- `rtl/lib/residual_add.v` implemented with streaming pipeline and saturating arithmetic.
- `rtl/lib/embed_gather.v` implemented with row addressing, dyadic scaling, and saturation into 24-bit residual stream.
- Verilator lint-only passed with 0 warnings on both modules.
- `tests/test_rtl_residual_embed.py` verified positive and negative saturation and token gather bit-for-bit against Python oracle.
- `make check2` passes 115/115 tests.
