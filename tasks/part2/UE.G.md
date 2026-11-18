# Gate HE — Full-Model Scale-Up Review

## Objective
Verify that the full 30-layer SmolLM2-135M decoder accelerator meets all gate criteria:
1. Full ROM images, program ROM, and top-level accelerator generated within budget (UE.1).
2. Verilator build clean and simulation throughput measured (UE.2).
3. Layer 0 trace equivalence verified across positions (UE.3).
4. Full decode token-exact across 10 prompts x 16 tokens with sampled-layer trace equivalence (UE.4).
5. Long-context simulation verified up to MAX_CTX = 256 (UE.5).
6. End-to-end host demo verified against FP32 greedy baseline (UE.6).
7. Final quality metrics evaluated on WikiText-2 (UE.7).
8. Gate HE review report documented in `results/gates/HE.md`.

## Status
Status: complete
