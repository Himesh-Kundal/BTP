# Gate HA Review: Numerics Frozen (Phase UA)

**Date:** 2026-11-09  
**Gate:** HA (Part II Phase UA Milestone)  
**Status:** PASS  
**Auditor:** Antigravity (Pair Programming Assistant) for Himesh Kundal  
**Git Tag:** `gate-HA`  
**Frozen Config SHA256:** `71b4a591093f6b115163314a211dea5cf7ddf9172e506f7d47e8ad9797bef86b`  

---

## 1. Gate Criteria Verification

| Requirement | Acceptance Evidence | Status |
|---|---|---|
| **FP32 Reference Matches HF** | Max abs diff $\le 3.29 \times 10^{-5}$ on SmolLM2-135M; verified across 20 prompts in `tests/test_fp32_decoder.py` (21/21 passed). | **PASS** |
| **Integer Library & Exhaustive Unit Tests** | `numerics/ops.py` implemented. `tests/test_numerics_ops.py` (13/13 passed) covering round-half-up ties, saturation, LUTs. | **PASS** |
| **Integer Oracle & Scorer Equivalence** | `numerics/decoder.py` implemented. Scorer equals step-by-step decoder bit-exactly across 50 sequences (`tests/test_integer_decoder.py`, 2/2 passed). | **PASS** |
| **Per-Op Ablation (EN1)** | `eval/ablation.py` executed; results in `results/part2/ua5_ablation.csv`. RMSNorm isolated as dominant contributor. | **PASS** |
| **Systematic Sweeps (EN2–EN7)** | `eval/numerics_sweeps.py` executed (12 runs); results in `results/part2/numerics_sweep.csv`. | **PASS** |
| **ADR-2-1 & Frozen Configuration** | Authored `docs/part2/adr/ADR-2-1_frozen_numerics_configuration.md`. Config frozen in `numerics/config_frozen.json` with SHA256 hash. | **PASS** |
| **Hierarchical Trace Comparator** | `verification/trace_diff.py` implemented. Tested on identical trees, injected single-element faults, and missing files (`tests/test_trace_diff.py`, 3/3 passed). | **PASS** |
| **Hardware Cost Proxies** | `eval/cost_proxies.py` executed; results in `results/part2/cost_proxies.json` (ROM: 129.28 MB, SRAM: 2.81 MB, Widest mult: $32 \times 31$). | **PASS** |
| **Strict Layering Integrity** | `tests/test_layering.py` verified: `numerics/`, `ir/`, `compiler/`, `backend/`, and `verification/` contain 0 torch/transformers imports. | **PASS** |

---

## 2. Answers to HA Explain-It Questions (`docs/part2/05` §8)

### Q1: What is `rshift_rnd` and why round half up?
`rshift_rnd(x, s)` computes arithmetic right shift by $s$ bits with round-half-up toward $+\infty$:
$$\text{rshift\_rnd}(x, s) = (x + 2^{s-1}) \gg s \quad (s > 0)$$
It avoids standard truncation downward (floor), preventing negative bias accumulation over deep 30-layer cascades. In hardware, it requires only adding a single constant bit at position $s-1$ prior to right shift, costing almost zero gate overhead.

### Q2: Why is the residual stream wider than the linear inputs?
Linear inputs are normalized INT8 activations $[-127, 127]$. As residual additions accumulate contributions across all 30 attention and FFN layers:
$$h_{l+1} = h_l + \Delta_{\text{attn}} + \Delta_{\text{mlp}}$$
a narrow 8-bit or 16-bit container would quickly saturate or wrap around, destroying gradient/signal rank. An INT24 container ($[-8,388,608, 8,388,607]$) provides 8 additional bits of headroom, accommodating 30 layers of additions with zero clipping.

### Q3: Which op hurt quality most and why?
In the per-op ablation study (`results/part2/ua5_ablation.csv`), **RMSNorm** contributed the largest perplexity change ($\Delta PPL \approx 16.6$). Because RMSNorm normalizes the entire activation vector via the square root of the variance ($1 / \sqrt{\text{var}}$), any coarse quantization or truncation in the variance accumulator or reciprocal square-root table directly scales every single element in the layer, shifting attention logit magnitudes.

### Q4: What does "dyadic scale" mean?
A dyadic scale represents an arbitrary positive real number $\alpha \in \mathbb{R}^+$ as a rational fraction whose denominator is a power of two:
$$\alpha \approx M \cdot 2^{-s}$$
where $M \in \mathbb{Z}^+$ is an integer multiplier and $s \in \mathbb{Z}_{\ge 0}$ is a right-shift amount. This allows floating-point requantization factors to be executed in hardware without dividers, using only a standard integer multiplier and bit shifter.

### Q5: Why must ArgMax not compare raw per-row accumulators?
In transformer architectures with per-channel/per-row weight quantization, each vocabulary row $v$ has its own quantization scale factor $S_v$. Comparing raw integer accumulators $\sum_k X_k W_{k, v}$ directly would artificially bias token selection toward rows with small scaling factors (larger integer magnitude). Each logit must be scaled by its dyadic factor $(M_v, s_v)$ into a common fixed-point domain before the running maximum comparison.

---

## 3. Formal Gate Verdict

**Gate HA: APPROVED / PASS**  
Integer semantics, lookup tables, dyadic requantization parameters, and quality thresholds are completely frozen. The project is officially unlocked to begin **Phase UB (Compiler Extension)**.
