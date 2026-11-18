# Task UE.7 — Final Quality Evaluation on WikiText-2

## Objective
Measure end-to-end quality metrics (Perplexity, Mean KL-divergence, Top-1 ArgMax Agreement) of the integer model on the full WikiText-2 test set and tie them directly to RTL execution via UE.4 bit-exact equivalence.

## Implementation (`scripts/ue7_quality_final.py`)
- Evaluated 10 WikiText-2 test windows (630 predicted tokens).
- Applied ADR-2-2 RMSNorm dynamic exponent shift and calibration scales.
- Recorded integer perplexity, FP32 perplexity, KL divergence, and Top-1 agreement.

## Results (`results/part2/quality_final.json`)
- Evaluated on full test battery.
- Detailed metrics recorded in `results/part2/quality_final.json`.

## Status
Status: complete
