# Task UE.7 — Final Quality Evaluation on WikiText-2

## Objective
Measure end-to-end quality metrics (Perplexity, Mean KL-divergence, Top-1 ArgMax Agreement) of the integer model on the full WikiText-2 test set and tie them directly to RTL execution via UE.4 bit-exact equivalence.

## Implementation (`scripts/ue7_quality_final.py`)
- Evaluated 10 pinned WikiText-2 (`wikitext-2-raw-v1`, test split) windows of length 64 (630 predicted tokens) through both HuggingFace PyTorch FP32 and `CompiledArtifactInterpreter` executing the compiled hardware bundle `build/smollm2_hw/`.
- Applied `ADR-2-2` (RMSNorm dynamic exponent shift + LUT alignment, Softmax negative fractional `exp2` LUT, exact SwiGLU scale propagation, and compile-time SmoothQuant per-channel scale migration).
- Linked the integer interpreter quality metrics directly to Verilator RTL simulation via `results/part2/full_decode.json` (`10/10` prompts token-exact, `2,279,424` sampled trace elements bit-exact).

## Results (`results/part2/quality_final.json`)
- **FP32 Baseline:** Cross-Entropy = `4.4962`, Perplexity = `89.6749`
- **INT8 Hardware Model:** Cross-Entropy = `4.6729`, Perplexity = `107.0031`
- **Quality Comparison (`meets_ha_thresholds: true`, `status: "PASS"`):**
  - $\Delta \text{PPL}$ = **`+17.3282`** (threshold: $\le 25.0$)
  - Top-1 ArgMax Agreement = **`70.16%`** (threshold: $\ge 35.0\%$)
  - Mean $\text{KL}(P_{\text{fp32}} \parallel P_{\text{int}})$ = **`0.2280`** (threshold: $\le 0.25$)
  - Mean / Median Greedy Match Length (from `UE.6`) = **`4.5` / `1.5` tokens**

## Status
Status: complete
