# Task UE.3 — Layer-0 Block Trace Equality Across Multiple Positions

## Objective
Prove bit-exact trace and token equality between the full 30-layer SmolLM2-135M RTL accelerator (`top_accel.v`) and `CompiledArtifactInterpreter` across early (`pos=0..3`) and later (`pos=8..11`) autoregressive decoding positions.

## Performance & Trace Fixes Implemented
1. **Interpreter Vectorization (`compiler/interpreter.py`):**
   - Vectorized `_get_linear_weight_matrix` via NumPy reshape/transpose with matrix caching, and vectorized per-channel dyadic requantization (`rshift_rnd`). Reduced 12-position decode time from 395.02s to 5.05s (78x speedup).
   - Added `embed_tokens_current` trace tracking so multi-token sequences always dump the current step's embedding vector in `sim/tb_generator.py`.
2. **Verilator `-O3` Optimization (`scripts/ue3_layer0_traces.py`):**
   - Compiled with `-CFLAGS -O3`, increasing RTL simulation throughput from `825.2 kcycles/s` to **`2,265.07 kcycles/s`** (~6.0s per full 30-layer token).

## Verification & Results (`results/part2/ue3_layer0_traces.json`)
- **Sequence:** 8 prompt tokens `[1, 42, 128, 512, 1024, 2048, 4096, 8192]` + 4 generated tokens (`pos=0..11`).
- **Tokens Exact:** `true` (`[1029, 1361, 44236, 631, 0, 41680, 9803, 27061, 43014, 29185, 22110, 29]`).
- **Trace Identical:** `true` (`first_divergence: null`, **216 files / 728,064 elements checked** bit-for-bit).
- **Cycles:** `165,511,344` cycles (`sim_sec = 73.07s`).

## Status
Status: complete
Commit: UE.3: Layer-0 trace and token equality across 12 positions on SmolLM2-135M
