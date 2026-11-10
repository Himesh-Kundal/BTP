# UC.9 — `kv_cache` RAM and Address Generator
Status: done
Phase/Gate: UC / HC     Tags: [rtl, memory]     Estimate: 2h
Depends on: UB.6         Spec refs: docs/part2/02 §3 (N2-9), §6, docs/part2/04 §UC.9

## Goal
Implement:
1. `rtl/lib/kv_cache.v`: Parameterized dual-port synchronous RAM for INT8 keys and values sized by `[NUM_LAYERS][2][NUM_KV_HEADS][MAX_CTX][HEAD_DIM]`.
2. `rtl/lib/kv_addr_gen.v`: Parameterized address generator and GQA coordinate mapper:
   - Translates `(layer, is_v, kv_head, pos, dim)` into flat RAM address.
   - Maps query head `q_head` to `kv_head = q_head / (NUM_Q_HEADS / NUM_KV_HEADS)`.
   - Supports vector and sequence streaming modes for vector write and attention sequence read.
3. Exposes strict Verilog-2001 compliance, zero floats, and clean lint under `iverilog` and `verilator`.

## Acceptance (executable)
- [x] `rtl/lib/kv_cache.v` and `rtl/lib/kv_addr_gen.v` lint clean in `iverilog` and `verilator`.
- [x] Read/write ordering tests: verify values written at position $p$ persist and can be read back across all layers and heads.
- [x] GQA head mapping test: verify $q\_head \to kv\_head$ for 9:3 ratio (SmolLM2) and 2:1 ratio (tiny model).
- [x] Causal addressing: verify historical positions $0 \le j \le pos$ can be streamed without boundary corruption.
- [x] `tests/test_rtl_kv_cache.py` passes and is added to `make check2`.
