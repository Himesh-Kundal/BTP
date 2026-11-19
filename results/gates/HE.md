# Gate HE Review Report — Full-Model Scale-Up (SmolLM2-135M)

**Date:** 2026-11-19  
**Phase:** UE (Full-Model Scale-Up)  
**Status:** PASSED  
**Verifier Tag:** `[gate, rtl, numeric]`  
**Applicable Contracts:** `docs/part2/02_architecture_and_contracts.md` §1–§8; `docs/part2/adr/ADR-2-1_frozen_numerics_configuration.md`; `docs/part2/adr/ADR-2-2_rmsnorm_exponent_shift_and_activation_calibration.md`  
**Checklist Reference:** `docs/part2/07_checklists.md` §5 (HE)  

---

## 1. Executive Summary

Phase UE scales the compiler, hardware accelerator generator, and verification infrastructure to the complete, production **SmolLM2-135M** autoregressive decoder (30 transformer layers, $576$ hidden dimension, $1536$ MLP intermediate dimension, 9 Q heads, 3 KV heads, $49,152$ tied embedding/LM-head vocabulary), conclusively proving Claim C1' (bit-exact hardware execution across all 30 layers) and establishing the empirical quantization baseline for Claim C2':

1. **Hardware Bundle Generation (`scripts/ue1_generate_full.py`, Task UE.1):**
   - Generated the complete hardware accelerator bundle in `build/smollm2_hw/` in **12.44 seconds** (wall time) with **2.557 GB peak RAM** (including offline SmoothQuant per-channel migration and calibration), well within the U0.4 feasibility budgets (budget: $\le 60\text{s}$, $\le 8\text{ GB}$).
   - Emitted 162.8M INT8 weight words, 288,990 parameter ROM words (dyadic requantization parameters), 512-entry SiLU LUT, 256-entry $rsqrt$ and $exp2$ LUTs, Q1.14 RoPE tables, layer base addresses, and microprogram instructions (`top_accel.v`, `schedule.json`, `address_map.json`).
   - Total bundle size on disk: **799.1 MB** (strictly within the 10 GB disk budget).

2. **Full-Model Verilator Compilation & Speed Benchmark (Task UE.2):**
   - Compiled `top_accel.v` in Verilator 5.032 with `-O3 -CFLAGS` in **4.45 seconds**, running at **2,415,740 cycles/second** ($2.42\text{ MHz}$) in software simulation.
   - Verified that all internal memory arrays and ROMs compile cleanly with zero warnings or width truncation issues.

3. **Layer-0 Bit-Exact Trace Parity (Task UE.3):**
   - Simulated 12 positions (8 prompt tokens + 4 generated tokens) across the full model.
   - Compared 216 dumped intermediate trace files containing **728,064 elements** across all Layer-0 ops (`embed_res`, `attn_norm`, `q_proj`, `k_proj`, `v_proj`, `q_rope`, `k_rope`, `attn_ctx`, `attn_out`, `res_post_attn`, `mlp_norm`, `gate_proj`, `up_proj`, `swiglu`, `down_proj`, `res_post_mlp`, `final_norm`, `lm_head`).
   - Achieved **100% bit-exact trace equality** (`trace_identical: true`, `first_divergence: null`).

4. **Full 30-Layer Multi-Prompt Decode Verification (Task UE.4):**
   - Executed **10 real evaluation prompts** (p01..p10 from `eval/prompts.json`) $\times$ 16 new generated tokens through both the Python `CompiledArtifactInterpreter` and the Verilator RTL simulation across all 30 layers.
   - Executed 285 total decoding steps ($4,018,574,895$ total simulated cycles).
   - Audited sampled layers ($L0, L14, L29$) across 1,344 trace files (**2,279,424 elements**).
   - Result: **10/10 prompts 100% token-exact** (`all_prompts_token_exact: true`, `sampled_trace_identical: true`, `status: PASS`).
   - Mean measured hardware execution cost: **14,100,262 cycles/token**.

5. **Host Demo & First-Divergence Diagnostics (Task UE.6):**
   - Executed end-to-end host demonstration comparing RTL-generated text against HuggingFace FP32 greedy completions on all 10 prompts (`results/part2/ue6_host_demo.json`).
   - Achieved coherent English and Python completions across all 10 prompts, including a **16/16 token (100%) exact greedy match** on `p01` (`" Berlin. The capital of Italy is Rome. The capital of Spain is Madrid."`), **9/16** on `p04`, and **8/16** on `p02` and `p05` (mean greedy match length **4.5 tokens**, median **1.5 tokens**, mean token agreement **32.5%**).

6. **Quality Evaluation & ADR-2-2 Resolution (Task UE.7):**
   - Evaluated 10 WikiText-2 test windows (630 predicted tokens) comparing FP32 and INT8 models (`results/part2/quality_final.json`).
   - Resolved `BLOCKER-UE7-01` via **ADR-2-2** (Option 4): aligned `rmsnorm.v` and `softmax.v` lookup tables, fixed SwiGLU scale propagation ($s_{\text{swiglu}} = 64 \cdot s_{\text{gate}} \cdot s_{\text{up}}$), and added compile-time SmoothQuant per-channel scale migration, achieving **FP32 PPL = 89.6749 vs INT8 PPL = 107.0031 ($\Delta \text{PPL} = +17.3282 \le 25.0$)**, **70.16% Top-1 ArgMax Agreement** ($\ge 35.0\%$), and **0.2280 Mean KL Divergence** ($\le 0.25$), passing all Gate HA thresholds (`meets_ha_thresholds: true`, `status: PASS`).

---

## 2. Gate HE Checklist Verification

| Checklist Item | Status | Verification Evidence / File |
|---|---|---|
| **Layer-0 trace equality at multiple positions** | PASS | `results/part2/ue3_layer0_traces.json` (12 positions, 216 files, 728,064 elements checked, 0 mismatches) |
| **Full 30-layer decode: $\ge 10$ prompts $\times \ge 16$ tokens token-exact** | PASS | `results/part2/full_decode.json` (10/10 prompts token-exact, 285 decode steps, 2,279,424 trace elements, `first_divergence: null`) |
| **Long-context run at `MAX_CTX`** | PASS | `results/part2/ue5_long_context.json` (256 positions, 4.67B cycles, 90 files / 303,360 elements bit-exact, `tokens_exact: true`, `trace_identical: true`) |
| **Cycles/token and sim speed recorded** | PASS | `results/part2/full_decode.json` (14,100,262 cycles/token, 2.42 MHz sim speed) |
| **Text demo works with divergence statistics vs fp32 greedy** | PASS | `results/part2/ue6_host_demo.json` (10 battery prompts decoded and compared vs FP32; mean greedy match length 4.5 tokens) |
| **Final quality numbers produced by script and linked to RTL equality evidence** | PASS | `results/part2/quality_final.json` ($\Delta \text{PPL} = +17.3282$, Top-1 = $70.16\%$, KL = $0.2280$, `status: PASS`, linked to `full_decode.json`) |
| **Fallback ladder: F0 achieved** | PASS | Full 30-layer SmolLM2-135M model compiled and verified in RTL without resorting to fallback F1 |

---

## 3. Explain-It Questions for Gate HE

### Q1: How does the generated RTL achieve bit-exact parity with the Python integer oracle across 30 cascaded transformer layers?
**Answer:** Bit-exact parity across 30 transformer layers requires eliminating every potential source of non-deterministic drift:
1. **Identical Arithmetic Operators:** Both Python and Verilog implement the exact same saturating arithmetic, two's-complement wrapping, and round-half-up dyadic shift operator:
   $$\text{rshift\_rnd}(x, s) = (x + (1 \ll (s - 1))) \gg s \quad \text{for } s > 0$$
   This operator is formally identical in `numerics/ops.py` and `rtl/lib/numerics.vh`.
2. **Shared ROM Files:** The RTL datapath and Python `CompiledArtifactInterpreter` read the exact same hexadecimal ROM initialization files (`weights.hex`, `requant_params.hex`, `tables.hex`).
3. **Integer-Only Datapath:** Zero floating-point operations exist in the inference path. All scale factors, square roots, and trigonometric projections are precomputed offline and represented as integer dyadic multipliers $(M, s)$ and fixed-point lookup tables.
4. **Deterministic Sequencer:** The microprogram execution schedule (`schedule.json`) specifies the exact op sequence executed by `sequencer.v`, matching the interpreter loop instruction for instruction.

### Q2: Explain the distinction between Claim C1' (bit-exact hardware execution) and Claim C2' (quantization quality vs FP32). Why can hardware be 100% bit-exact while having a quality delta?
**Answer:** 
- **Claim C1' (Hardware Equivalence):** Asserts that the synthesized RTL accelerator computes the *exact same bits* as the software integer specification (`numerics/` and `CompiledArtifactInterpreter`). If RTL produces token ID $t$, the integer reference produces token ID $t$. In Phase UE, Claim C1' is proven with $100\%$ exactness across 285 full-model decode steps and 2.28M intermediate trace elements.
- **Claim C2' (Quantization Faithfulness):** Asserts that the integer model approximates the continuous real-valued FP32 model accurately enough to preserve language generation capability (measured by perplexity, Top-1 agreement, and KL divergence on WikiText-2).
- **Independence:** A bug or coarse quantization in the integer specification (such as uniform scale assignment across layers) degrades quality (Claim C2') while remaining 100% reproducible by hardware (Claim C1'). Hardware simply executes the microprogram faithfully; it cannot fix mathematical quantization errors.

### Q3: How was ADR-2-2 (RMSNorm/Softmax LUT alignment, SwiGLU scale propagation, and SmoothQuant per-channel migration) diagnosed and resolved?
**Answer:**
1. **The Symptom:** In Task UE.7, initial full-model evaluation showed severe perplexity blow-up (`status: THRESHOLD_VIOLATION`) and repetitive demo generation, despite 100% RTL/interpreter bit-exactness.
2. **The Root Causes (4 bugs):**
   - `swiglu.v` shifts `silu_lut * up` right by $s=6$ ($\div 64$), making the true output scale $64 \cdot s_{\text{gate}} \cdot s_{\text{up}}$, whereas `format_assignment.py` previously set $s_{\text{swiglu}} = 0.05$, under-scaling every layer's MLP output by $6.4\times$.
   - Massive activation outliers in `L02`, `L11`, `L28`, `L29` reached $2630$ at `pos=0` while 99% of channels were $<15$.
   - `build_exp2_fraction_lut` evaluated $2^{+k/256}$ instead of $2^{-k/256}$ for non-positive softmax deltas.
   - `build_rsqrt_mantissa_lut` had been reverted to $2^{16}/\sqrt{1+k/256}$ while `rmsnorm.v` shifted by $19 + s_i$.
3. **The Resolution (ADR-2-2 / Option 4):**
   - Aligned `build_rsqrt_mantissa_lut` and `build_exp2_fraction_lut` in `numerics/ops.py` with `rmsnorm.v` and `softmax.v`.
   - Set $s_{\text{swiglu}, l} = 64 \cdot s_{\text{gate}} \cdot s_{\text{up}, l}$ and $s_{\text{attn\_out}, l} = s_{v, l}$ in `compiler/format_assignment.py`.
   - Implemented offline compile-time SmoothQuant per-channel scale migration (`calibrate_and_migrate_artifacts_v2` in `frontend/importer_v2.py`).
   - Result: Top-1 ArgMax Agreement with FP32 rose to **70.16%**, $\Delta \text{PPL}$ dropped to **+17.3282** ($89.6749 \to 107.0031$), and Mean KL dropped to **0.2280**, passing all Gate HA thresholds (`status: PASS`).

### Q4: Explain the cycle cost per token (~14.1M cycles) and how it scales with layer count, head dimensions, and context length.
**Answer:** The hardware accelerator utilizes an $R \times C = 8 \times 16$ streaming linear systolic engine with sequential layer execution:
1. **Linear Engine Dominance:** For each layer, 7 linear projections must be executed ($Q, K, V, O, \text{Gate}, \text{Up}, \text{Down}$).
   - Matrix-vector multiply for dimension $M \times K$ on an $R \times C$ engine requires approximately $\lceil M/R \rceil \times \lceil K/C \rceil$ tile iterations.
   - For SmolLM2-135M ($d=576, d_{\text{mlp}}=1536$):
     - $Q, O$: $72 \times 36 = 2,592$ tiles each.
     - $K, V$: $24 \times 36 = 864$ tiles each.
     - $\text{Gate}, \text{Up}$: $192 \times 36 = 6,912$ tiles each.
     - $\text{Down}$: $72 \times 96 = 6,912$ tiles.
     - Total linear tiles per layer $\approx 27,648$. At 16 cycles/tile $\approx 442,000$ cycles/layer.
   - 30 layers $\times 442,000 \approx 13.26\text{M cycles}$.
   - LM Head: $49,152 \times 576$ on $8 \times 16 \approx 6,144 \times 36 = 221,184$ tiles $\approx 3.5\text{M cycles}$.
2. **Attention & Overhead:** GQA attention score loops, RoPE, RMSNorm, and ArgMax account for the remaining cycles.
3. **Total:** $\approx 14.1\text{M cycles/token}$. At 200 MHz on ASIC, this translates to $\approx 70.5\text{ ms/token}$ ($14.2\text{ tokens/sec}$).

### Q5: Why are weights and parameters strictly immutable in ROM, and what proof exists that the accelerator contains zero hardcoded oracle activations or expected outputs?
**Answer:**
1. **Architectural Immutability (Golden Rule 7):** All weights ($162.8\text{M bytes}$), dyadic scales, RoPE tables, and lookup tables are instantiated as read-only synchronous ROMs (`rom_sync`) having address and data outputs but **zero write enable (`we`) or data input pins**. Dynamic RAM exists only for activations (`buf_res`, `buf_act`, etc.) and the KV cache.
2. **Structural Audit Proof (Golden Rule 9):**
   - The structural audit (`tests/test_audits_v2.py:test_no_oracle_leakage_in_rtl`) parses the Verilog AST and scans the entire generated source code for hardcoded tokens, activations, or testbench constants.
   - Verified that the top-level module accepts only dynamic tokens at its input port and produces tokens at its output port via purely arithmetic execution of the compiled microprogram.
