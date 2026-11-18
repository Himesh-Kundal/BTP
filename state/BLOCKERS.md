# Open Blockers

*None.* All blockers resolved.

---

# Resolved Blockers

## BLOCKER-UE7-01: Integer Model Perplexity Gap on SmolLM2-135M (Quality Threshold Violation)
- **Date Opened:** 2026-11-19
- **Date Resolved:** 2026-11-19
- **Phase / Task:** UE.6 / UE.7 / Gate HE
- **Resolution:**
  - User approved ADR-2-2 (`docs/part2/adr/ADR-2-2_rmsnorm_exponent_shift_and_activation_calibration.md`).
  - Implemented dynamic exponent shift `calc_sh_amt / 2` and dyadic scaling in `rtl/lib/rmsnorm.v`, `numerics/decoder.py`, and `compiler/interpreter.py`.
  - Added calibration scale factors for intermediate tensors.
  - Successfully regenerated the hardware bundle in `build/smollm2_hw/` (UE.1).
  - Verified 100% bit-exact parity across Layer 0 (UE.3) and full 30-layer decode across 10 prompts $\times$ 16 tokens (UE.4).
  - Re-evaluated host demo (UE.6) and quality suite (UE.7).
  - Closed Gate HE.
