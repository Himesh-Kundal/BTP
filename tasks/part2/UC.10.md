# UC.10 — `linear_engine` R×C with ROM Streaming and Requant Output
Status: done
Phase/Gate: UC / HC     Tags: [rtl, engine]     Estimate: 3h
Depends on: UC.1, UC.2   Spec refs: docs/part2/02 §3 (N2-3, N2-7), §6, docs/part2/04 §UC.10

## Goal
Implement `rtl/lib/linear_engine.v`:
1. Parameterized $R \times C$ MAC array consuming $C$ input elements from activation buffer and $R \times C$ weights streamed sequentially from Weight ROM.
2. Accumulates in 32-bit registers across tile columns `tc = 0 .. ceil(K/C)-1`.
3. Streams final accumulated rows through per-channel dyadic requantization $(M_i, s_i)$ from Requant ROM into output buffer.
4. Handles arbitrary non-multiple matrix shapes $(N, K)$ with zero-padding contract (per Part I and UB.4).
5. Clean lint in `iverilog` and `verilator`.

## Acceptance (executable)
- [x] `rtl/lib/linear_engine.v` implemented and lints clean in `iverilog` and `verilator`.
- [x] Passes Part I test vectors and new non-multiple shapes ($N, K$) bit-for-bit against NumPy/integer oracle.
- [x] Verifies sequential streaming from Weight ROM matching `rom_packing.py` layout.
- [x] Supports both INT8 output format and 24-bit container format (`out_is_res`).
- [x] `tests/test_rtl_linear_engine.py` passes and is added to `make check2`.
