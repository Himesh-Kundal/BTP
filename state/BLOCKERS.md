# Open Blockers

## BLOCKER-UE7-01: Integer Model Perplexity Blow-Up on SmolLM2-135M (Quality Threshold Violation)
- **Date:** 2026-11-19
- **Phase / Task:** UE.6 / UE.7 / Gate HE
- **Rule Triggered:** `AGENTS.md` Escalation Rule (*"Quality results contradict the plan (e.g. perplexity blow-up): stop tuning, write up, ask."*)

### Observed Measurements
- **Source:** `results/part2/quality_final.json` (10 WikiText-2 test windows, 630 predicted tokens) and `results/part2/ue6_host_demo.json` (10 prompts $\times$ 16 new tokens)
- **FP32 Perplexity:** `89.6749` (cross-entropy `4.4962`)
- **INT8 Perplexity:** `1,130,744.2269` (cross-entropy `13.9384`)
- **Delta PPL:** `+1,130,654.552` (HA frozen threshold: `<= 25.0`)
- **Mean KL(fp32 || int):** `9.4782` (HA frozen threshold: `<= 0.15`)
- **Top-1 Agreement:** `0.16%` (HA frozen threshold: `>= 35.0%`)
- **Greedy Match Length (p01–p10):** `0 / 16` (`0.0%` token agreement)

### Root-Cause Analysis
1. **RTL vs. Compiler Interpreter Parity is 100% Verified:** In `results/part2/full_decode.json` (UE.4), the generated RTL (`top_accel.v`) matched `CompiledArtifactInterpreter` bit-for-bit across all 10 prompts $\times$ 16 new tokens (285 decode steps, 2,279,424 trace elements, `first_divergence: null`).
2. **Compiler Scale Bug (`compiler/format_assignment.py`):**
   - In `OpKind.RMSNorm` (lines 139–143), `float_to_dyadic_scale(float(g_val))` folds $\gamma$ without dividing by the output activation scale `s_act = 0.05` (a $20\times$ under-scaling). Consequently, ~97.5% of `L0/attn_norm` elements (562/576) quantize to `0`.
   - Static uniform scales (`s_act = 0.05`, `s_res = 0.01`) are used across all 30 layers without per-layer calibration on SmolLM2-135M.

### Awaiting Human Decision
1. **Option A (Recommended):** Fix `compiler/format_assignment.py` to divide RMSNorm $\gamma$ by `s_act` and apply per-layer activation calibration on SmolLM2-135M, then regenerate `build/smollm2_hw` and re-run UE.1–UE.7.
2. **Option B:** Record the hardware bit-exactness verification as-is and waive quality thresholds for Gate HE (not recommended per Gate HE criteria).
