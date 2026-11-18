# Task UE.4 — Full 30-Layer SmolLM2-135M Decode Verification

## Objective
Verify 100% bit-exact autoregressive token generation across $\ge 10$ real text prompts $\times 16$ new tokens each on the complete 30-layer SmolLM2-135M RTL accelerator (`top_accel.v`), prove sampled-layer (`L0`, `L14`, `L29`) trace equality against `CompiledArtifactInterpreter`, and measure cycles per token.

## Implementation (`scripts/ue4_full_decode.py`)
- Tokenized prompts `p01` through `p10` from `eval/prompts.json` using the pinned SmolLM2-135M tokenizer.
- Executed `CompiledArtifactInterpreter` across all 10 prompts (285 total decode steps) to produce golden tokens and `L0`/`L14`/`L29` traces for `p01`.
- Compiled and executed all 10 Verilator `-O3` simulations in parallel (4 workers).

## Verification & Results (`results/part2/full_decode.json`)
- **Prompts Verified:** `10 / 10` (`p01`..`p10`), `16` new tokens per prompt (`285` total decode steps).
- **Token Exactness:** `100%` (`all_prompts_token_exact: true`, zero mismatches across all 285 steps).
- **Sampled-Layer Trace Equality (`L0`, `L14`, `L29` on `p01`):** `sampled_trace_identical: true`, `first_divergence: null` (**1,344 trace files / 2,279,424 elements** matched bit-for-bit).
- **Cycles Simulated:** `4,018,574,895` total cycles (`mean_cycles_per_token: 14,100,262`).
- **Parallel Wall Time:** `673.53s`.

## Status
Status: complete
Commit: UE.4: full 30-layer decode exactness across 10 prompts x 16 new tokens
