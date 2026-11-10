# UC.5 — `softmax` Unit
Status: done
Phase/Gate: UC / HC     Tags: [rtl, numeric]     Estimate: 2h
Depends on: UC.1         Spec refs: docs/part2/02 §3 (N2-10), docs/part2/06 §A4

## Goal
Implement `rtl/lib/softmax.v` implementing causal softmax via:
1. Running maximum subtraction ($m = \max_{j} s_j$, $\Delta_j = s_j - m \le 0$).
2. Base-2 exponential decomposition ($2^{\Delta_j} = \text{LUT}[\text{frac}] \gg \text{int\_part}$).
3. Wide accumulator sum ($\Sigma = \sum 2^{\Delta_j}$).
4. Exact normalizer division into Q0.15 probabilities ($p_j = (\text{exp}_j \times 32767) / \Sigma$).
5. Verification against `numerics/decoder.py` including single-position context ($pos=0$), equal scores, extreme score spreads, and $pos = \text{MAX\_CTX}-1$.

## Acceptance (executable)
- [x] `rtl/lib/softmax.v` implemented and lints clean in `iverilog` and `verilator`.
- [x] Bit-exact match with `numerics/decoder.py` across single-position, equal scores, extreme spread, and random score vectors up to `MAX_CTX=256`.
- [x] `tests/test_rtl_softmax.py` passes and is added to `make check2`.

## Evidence (after)
- `rtl/lib/softmax.v` implemented with 3-pass pipelined architecture (Pass 1 max, Pass 2 base-2 exp & sum, Pass 3 Q0.15 normalizer division).
- 18-bit precision correctly accommodates Q1.16 exp2 values up to $2.0 \times 65536 = 131072$.
- Verilator lint-only passed with 0 warnings.
- `tests/test_rtl_softmax.py` verified bit-for-bit across all edge cases and full context sequence lengths up to 256.
- `make check2` passes 112/112 tests.
