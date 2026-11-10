# UC.11 — `attention` Datapath Integration
Status: done
Phase/Gate: UC / HC     Tags: [rtl, datapath]     Estimate: 3h
Depends on: UC.5, UC.9, UC.10   Spec refs: docs/part2/02 §3 (N2-9, N2-10), §6, docs/part2/04 §UC.11

## Goal
Implement `rtl/lib/attention.v` datapath orchestrating:
1. KV Cache Store: Writes incoming $K$ and $V$ vectors for the current token into `kv_cache` at position `pos`.
2. GQA Query-to-KV head mapping: $kv\_idx = q\_idx / (NUM\_Q\_HEADS / NUM\_KV\_HEADS)$.
3. Score Dot Product Loop: Computes $q \cdot k^T \gg 2$ across all historical positions $j \in [0, pos]$ into local scores buffer.
4. Softmax Integration: Drives `softmax.v` over historical scores $[0 \dots pos]$ to generate Q0.15 attention probabilities.
5. Context Accumulator: Computes weighted sum $\sum_{j=0}^{pos} P_j \cdot V_j$ with $\gg 15$ scaling and symmetric INT8 saturation into output buffer.
6. Bit-exact equivalence against Python integer decoder oracle across positions and head groups.

## Acceptance (executable)
- [x] `rtl/lib/attention.v` implemented and lints clean in `iverilog` and `verilator`.
- [x] Correctly stores new keys/values into `kv_cache` at position $pos$.
- [x] Evaluates causal attention over historical context $0 \dots pos$ bit-for-bit against `compiler/interpreter.py` and `numerics/decoder.py`.
- [x] Verified across multiple positions ($0, 1, 5, 10$) and GQA query heads.
- [x] `tests/test_rtl_attention.py` passes and is added to `make check2`.

## Evidence
- `tests/test_rtl_attention.py` passes 3/3 test batteries including:
  1. `test_attention_single_token`: pos = 0 boundary case where probability is 1.0.
  2. `test_attention_multi_position_causal`: causal multi-step history (pos = 3).
  3. `test_attention_gqa_mapping`: 4 query heads to 2 KV heads (2:1 group ratio) with causal history.
- Verilator `--lint-only -Wall --top-module attention` and `iverilog -g2005 -Wall` 100% clean with 0 warnings.
- Added to `Makefile` `check2`. Pass recorded in `make check2` (133/133 tests green).
