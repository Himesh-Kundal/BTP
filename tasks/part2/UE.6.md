# Task UE.6 — Host Demo & FP32 Comparison

## Objective
Run end-to-end host demo script: prompt text -> tokens -> RTL simulation -> text. Compare with FP32 HuggingFace greedy text generation and report agreement statistics.

## Implementation (`scripts/ue6_host_demo.py`)
- Evaluated prompts `p01` through `p10` from `eval/prompts.json` across 16 generated tokens in Verilator RTL simulation (`results/part2/full_decode.json`).
- Compared RTL-decoded token sequences and detokenized strings against HuggingFace FP32 greedy generation (`do_sample=False`).
- Supports `--prompt "<text>" --max-new-tokens N` for interactive live RTL simulation and `--battery` (default) for the 10-prompt evaluation suite.

## Results (`results/part2/ue6_host_demo.json`)
- **Prompts Evaluated:** 10 (`p01`..`p10`), 16 new tokens each (160 generated tokens).
- **RTL vs Interpreter Parity:** `10/10` prompts 100% bit-exact (`rtl_tokens_exact: true`).
- **Greedy Match Length vs FP32:** Mean = `4.5` tokens, Median = `1.5` tokens, Max = `16/16` tokens (`p01`: `" Berlin. The capital of Italy is Rome. The capital of Spain is Madrid."`), Mean multi-step token agreement = `32.5%`.
- **Sample Live Demo (`--prompt "Once upon a time" --max-new-tokens 6`):**
  - FP32 Greedy: `', there was a little girl'`
  - RTL INT8: `', in the year 1'` (`137,561,620` cycles, `69.56s`)

## Status
Status: complete
