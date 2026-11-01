# ADR-0001: Freezing Claim C2 Quantization Quality Thresholds

Date: 2026-11-01  
Status: accepted  
Spec Ref: docs/02 §2 (Rule N9, Claim C2), docs/04 §3 (T3.6), docs/07 §4 item 45  

## Context
Per Rule N9 in `docs/02_architecture_and_contracts.md`, Claim C2 posits that the INT8 integer reference reproduces the FP32 float reference with high numerical fidelity across all 7 transformer projection types (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

Rule N9 and the numeric checklist explicitly mandate that C2 acceptance thresholds must NOT be invented in advance; instead, they must be empirically measured across representative decoder layers (first: 0, middle: 14, last: 29) on calibrated activations and frozen via an ADR.

## Empirical Measurements (from `results/quant_quality.json`)
The full 21-configuration evaluation executed on 48 calibration tokens per projection produced the following statistics:
- **Cosine Similarity**:
  - Minimum: `0.988700` (measured at `layer_29_up_proj`)
  - Mean: `0.997881`
  - Maximum: `0.999844` (`layer_0_o_proj`)
- **SQNR (Signal-to-Quantization-Noise Ratio)**:
  - Minimum: `16.36 dB` (measured at `layer_29_up_proj`)
  - Mean: `26.93 dB`
  - Maximum: `35.01 dB` (`layer_0_o_proj`)
- **Relative L2 Frobenius Error**:
  - Minimum: `0.0178` (`layer_0_o_proj`)
  - Mean: `0.0519`
  - Maximum: `0.1520` (`layer_29_up_proj`)
- **Maximum Absolute Error**:
  - Minimum: `0.0158` (`layer_0_v_proj`)
  - Maximum: `44.702` (`layer_29_down_proj`)

## Decision
We freeze the following acceptance thresholds for Claim C2 across all dense linear projections:
1. **Cosine Similarity**: $\ge 0.980$ (guarantees directional fidelity $> 98\%$ in high-dimensional token space).
2. **SQNR**: $\ge 15.0\text{ dB}$ (guarantees signal power exceeds quantization noise power by at least $31.6\times$).
3. **Relative L2 Error**: $\le 0.180$ (guarantees L2 quantization distortion remains within 18% of signal energy).

## Consequences
- `tests/test_quality.py` asserts every evaluated configuration strictly meets these three frozen bounds.
- Any future compiler pass, calibration tuning, or quantizer change that degrades cosine similarity below $0.98$ or SQNR below $15.0\text{ dB}$ will fail automated regression testing.
- Preserves full integrity and traceability of numeric claims in the final report.

Approved by: Student & Implementing Agent (per O-2 / standard thesis workflow)
