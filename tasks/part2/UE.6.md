# Task UE.6 — Host Demo & FP32 Comparison

## Objective
Run end-to-end host demo script: prompt text -> tokens -> RTL simulation -> text. Compare with FP32 HuggingFace greedy text generation and report agreement statistics.

## Implementation (`scripts/ue6_host_demo.py`)
- Evaluated prompts `p01` through `p10` from `eval/prompts.json` across 16 generated tokens.
- Ran host demo against compiled RTL and compared with FP32 greedy reference.
- Reported exact match length and Top-1 token agreement.

## Results (`results/part2/ue6_host_demo.json`)
- Verified on 10 evaluation prompts.
- Top-1 agreement and match length recorded.

## Status
Status: complete
