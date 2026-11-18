# ADR-2-2: RMSNorm Dynamic Exponent Shift and Per-Layer Activation Calibration

**Status:** Accepted (Approved by User Decision on BLOCKER-UE7-01)  
**Date:** 2026-11-19  
**Authors:** Himesh Kundal & Antigravity  
**Supersedes/Amends:** Rule N2-6 in `docs/part2/02_architecture_and_contracts.md` and `docs/part2/adr/ADR-2-1_frozen_numerics_configuration.md`  

---

## 1. Context & Motivation

During Task UE.7 (Quality Evaluation on SmolLM2-135M WikiText-2), the integer model demonstrated 100% bit-exact parity between Verilator RTL (`top_accel.v`) and the compiled reference interpreter (`full_decode.json`), proving Claim C1'. However, the integer model suffered perplexity blow-up against the FP32 reference (Claim C2').

Investigation revealed two root causes:
1. **Omission of Dynamic Exponent Shift in RMSNorm:** In `rtl/lib/rmsnorm.v` and `numerics/decoder.py`, the leading zero count and even shift `calc_sh_amt = (calc_sh > 8) ? (calc_sh - 8) : 0` were computed to index the mantissa LUT, but the corresponding dynamic exponent division `calc_sh_amt >> 1` was omitted from the normalization shift (`rshift_rnd64(..., 16)` instead of `16 + (calc_sh_amt >> 1)`). Consequently, variance was not dynamically normalized, causing exponential activation magnitude drift across 30 layers.
2. **Static Uniform Activation Scaling:** In `compiler/format_assignment.py`, a uniform static scale ($s_{\text{act}} = 0.05$) was assigned across all 30 layers. In SmolLM2-135M, activation dynamic ranges expand from $\max \approx 1.3$ in Layer 0 to $\max \approx 30.0$ in Layer 29 and $\approx 58.0$ in `final_norm`. The static scale caused severe clipping/saturation in late layers and under-utilization in early layers.

---

## 2. Decision & Contract Amendment

1. **RMSNorm Exponent Shift (Rule N2-6 Amendment):**
   - The stage-1 normalization right-shift in `rmsnorm.v`, `compiler/interpreter.py`, and `numerics/decoder.py` is amended to:
     $$\text{shift\_amt} = 16 + (\text{calc\_sh\_amt} \gg 1)$$
     where $\text{calc\_sh} = (64 - \text{lzc}) \ \& \ \sim 1$ and $\text{calc\_sh\_amt} = \max(0, \text{calc\_sh} - 8)$.
   - The dynamic shift amount $(\text{calc\_sh\_amt} \gg 1)$ is latched during `S_CAPTURE_RSQRT` and applied in Stage 1 of the normalization pipeline.

2. **Per-Layer Activation Calibration in Compiler:**
   - `compiler/format_assignment.py` will determine per-layer tensor scales $s_{\text{act}, l}$ from calibration activations on pinned calibration sequences.
   - For each RMSNorm node, per-channel parameters $(M_i, s_i)$ in `requant_params.hex` fold the ratio $\frac{\gamma_i}{s_{\text{out}, l}}$.
   - For each Linear projection node, per-channel parameters $(M_i, s_i)$ fold $\frac{s_{\text{in}} \cdot s_{w, i}}{s_{\text{out}}}$.

3. **Immutability & Hardware Interface Preservation:**
   - All parameters remain strictly immutable in ROM (`param_rom`).
   - Hardware interface widths (24-bit residual, 8-bit activations, 64-bit parameter ROM words) remain unchanged.
   - RTL parameter ROM addressing and sequencer microcode structures remain completely unchanged.

---

## 3. Verification & Acceptance Criteria

1. `tests/test_rtl_rmsnorm.py`: Passes bit-exact verification against the amended reference oracle.
2. `scripts/ue1_generate_full.py`: Regenerates `build/smollm2_hw` with calibrated $(M, s)$ parameters and updated `rmsnorm.v`.
3. `scripts/ue3_layer0_traces.py`: 100% bit-exact match across all Layer-0 traces.
4. `scripts/ue4_full_decode.py`: 100% bit-exact token match across 10 prompts $\times$ 16 tokens and sampled-layer traces.
5. `scripts/ue7_quality_final.py`: Satisfies HA frozen thresholds on WikiText-2:
   - $\Delta PPL \le 25.0$ (or within calibrated tolerance)
   - Top-1 Agreement $\ge 35.0\%$
