# ADR-2-1: Frozen Numeric Formats, Table Sizes, and Quality Thresholds

**Status:** Accepted (Frozen at Gate HA)  
**Date:** 2026-11-09  
**Authors:** Himesh Kundal & Antigravity  
**SHA256 Hash of `numerics/config_frozen.json`:** `71b4a591093f6b115163314a211dea5cf7ddf9172e506f7d47e8ad9797bef86b`  

---

## 1. Context & Motivation

To fulfill Golden Rule 3 ("Numerics first") and prevent downstream RTL rework, all numerical formats, bitwidths, rounding schemes, lookup table dimensions, and quality thresholds must be empirically evaluated, Pareto-optimized, and frozen before RTL implementation begins.

Tasks UA.5 (Per-Op Ablation) and UA.6 (Precision Sweeps EN2–EN7) measured quality degradation and hardware cost proxies across candidate configurations on teacher-forced evaluation sequences.

---

## 2. Empirical Pareto Evidence

From `results/part2/numerics_sweep.csv` and `results/part2/ua5_ablation.csv`:

| Parameter Explored | Candidates Evaluated | Selected Value | Quality Impact | Hardware Cost Impact | Rationale |
|---|---|---|---|---|---|
| **Residual Stream Width** | 16, 20, **24**, 32 bits | **INT24** | $\Delta PPL = 0.0$ | 24-bit adders ($1.5\times$ INT16) | 24 bits provides 8-bit dynamic headroom over 30 layers without saturation risk while keeping adders compact. |
| **Linear Projections** | INT8 vs INT16 | **INT8** | $\Delta PPL = 0.0$ | Standard $8 \times 8$ MACs | INT8 weights $[-127, 127]$ with per-channel dyadic $(M, s)$ preserve full rank without quality loss. |
| **RMSNorm rsqrt LUT** | 8, 10, 12, **16** bits | **16 bits (Q0.16)** | $\Delta PPL = 0.0$ | 512 bytes ROM ($256 \times 16$) | 16-bit mantissa eliminates rounding drift in variance normalization. |
| **RoPE Trigonometric ROM** | Q1.10, Q1.12, **Q1.14**, Q1.15 | **Q1.14 (INT16)** | $\Delta PPL = 0.0$ | 512 bytes ROM ($256 \times 16$) | Q1.14 preserves coordinate orthogonality over 256 positions. |
| **Softmax Exp2 LUT** | 6, **8**, 10 bits | **8 bits (Q0.16)** | $\Delta PPL = 0.0$ | 512 bytes ROM ($256 \times 16$) | Base-2 decomposition $2^{\text{int}} \times 2^{\text{frac}}$ accurately computes probabilities. |
| **SwiGLU SiLU LUT** | 256 vs **512** entries | **512 entries (INT8)** | $\Delta PPL = -0.027$ | 512 bytes ROM ($512 \times 8$) | 512-entry symmetric LUT covers $[-128, 127]$ domain exhaustively. |
| **ArgMax Alignment** | Raw acc vs **Dyadic Scale** | **Per-row Dyadic** | ArgMax deterministic | Zero logic (folded in scan) | Raw INT32 accumulators cannot be compared due to per-channel scales; dyadic scaling normalizes logits. |

---

## 3. The Frozen Format Contract

The formats defined in [`numerics/config_frozen.json`](file:///home/himesh/BTP/numerics/config_frozen.json) are now **LAW** for all downstream compiler (Phase UB) and RTL (Phase UC) tasks:

1. **Rounding Scheme:** `rshift_rnd(x, s) = (x + (1 << (s - 1))) >> s` for $s > 0$ (round-half-up toward $+\infty$); identity for $s = 0$.
2. **Saturation:** Symmetric INT8 $[-127, 127]$ for all activation and linear weight boundaries.
3. **Residual Stream:** Signed 24-bit saturating container $[-8388608, 8388607]$.
4. **All Weights, Tables, and Dyadic Scales in Immutable ROMs:** Zero RAM or runtime write ports permitted for model parameters.
5. **No Floating Point:** Zero `float`, `double`, or floating-point HDL operators in the inference path.

---

## 4. Frozen Quality Thresholds (Claim C2′)

Hardware integer model performance vs. FP32 reference must satisfy:
- **Maximum Permissible Perplexity Degradation ($\Delta PPL$):** $\le \mathbf{25.0}$ points.
- **Minimum Top-1 ArgMax Agreement:** $\ge \mathbf{35.0\%}$.
- **Maximum Mean KL Divergence:** $\le \mathbf{0.150}$.

---

## 5. Consequences & Merging Guard

Any proposed change to table sizes, bitwidths, rounding formulas, or thresholds requires a formal ADR and human approval.
