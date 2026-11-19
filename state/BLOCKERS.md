# Open Blockers

*(No active blockers)*

---

## Resolved Blockers

### BLOCKER-UE7-01: Integer Model Perplexity & Demo Quality Gap on SmolLM2-135M (Quality Threshold Violation)
- **Date Opened:** 2026-11-19
- **Date Resolved:** 2026-11-19
- **Phase / Task:** UE.6 / UE.7 / Gate HE
- **Status:** RESOLVED (`ADR-2-2` Option 4 — Pure INT8 Scaling, LUT & SmoothQuant Migration Fix)
- **Root-Cause Diagnosis (4 Concrete Arithmetic & Calibration Bugs):**
  1. **SwiGLU Scale Under-Accounting (`compiler/format_assignment.py`):** `swiglu.v` multiplies `silu_lut` (scale $s_{\text{gate}}$) by `up` (scale $s_{\text{up}}$) and shifts right by $s=6$ ($\div 64$), yielding an actual output scale $s_{\text{swiglu}} = 64 \cdot s_{\text{gate}} \cdot s_{\text{up}}$. `format_assignment.py` previously assigned $s_{\text{swiglu}} = 0.05$, under-scaling every layer's MLP output by $6.4\times$.
  2. **Incomplete Per-Layer & Per-Channel Activation Calibration (Massive Activations, Risk RP2 / EN4):** `calib_scales.json` previously only calibrated `h_attn_norm` and `h_mlp_norm` per-tensor without SmoothQuant migration, leaving `q`, `k`, `v`, `gate`, `up`, and `swiglu` at static `0.05` and allowing 2–3 massive outlier channels in `L02`, `L11`, `L28`, `L29` (`swiglu` reaching $2630$) to clip or crush the quantization resolution of the remaining 1534 channels.
  3. **Softmax Fractional LUT Sign Inversion (`numerics/ops.py`):** `build_exp2_fraction_lut` computed $2^{+k/256}$ instead of $2^{-k/256}$ for negative score differences $\Delta = \text{max\_score} - \text{score} \ge 0$, causing non-monotonic sawtooth probabilities in `softmax.v`.
  4. **RMSNorm LUT / Shift Reversion (`numerics/ops.py`):** `build_rsqrt_mantissa_lut` had been reverted to $2^{16}/\sqrt{1+k/256}$ while `rmsnorm.v` and `interpreter.py` shifted by $19 + s_i$, mis-scaling RMSNorm outputs.
- **Verified Resolution (`results/part2/quality_final.json`, `results/part2/full_decode.json`, `results/part2/ue6_host_demo.json`):**
  - Updated `build_rsqrt_mantissa_lut` and `build_exp2_fraction_lut` in `numerics/ops.py`.
  - Implemented compile-time SmoothQuant per-channel scale migration (`calibrate_and_migrate_artifacts_v2` in `frontend/importer_v2.py`) and exact non-linear output scale propagation in `compiler/format_assignment.py` and `compiler/rom_packing.py`.
  - Regenerated `build/smollm2_hw/` (`UE.1`), verified 100% bit-exact Layer-0 traces (`UE.3`: `216/216` files, `728,064` elements) and 100% bit-exact 30-layer full decode across `10/10` prompts $\times 16$ tokens (`UE.4`: `285` steps, `2,279,424` sampled trace elements).
  - Verified on 10 WikiText-2 test windows (`UE.7`, 630 tokens): **FP32 PPL = `89.6749` vs INT8 PPL = `107.0031` ($\Delta \text{PPL} = +17.3282 \le 25.0$)**, **Top-1 ArgMax Agreement = `70.16%` ($\ge 35.0\%$)**, **Mean KL = `0.2280` ($\le 0.25$)**, `meets_ha_thresholds: true`, `status: "PASS"`.
  - Verified coherent English and Python code completions in RTL simulation (`UE.6`: e.g., `p01` matches FP32 greedy 16/16 tokens `" Berlin. The capital of Italy is Rome. The capital of Spain is Madrid."`; mean greedy match length `4.5` tokens, `32.5%` multi-step greedy token agreement).
