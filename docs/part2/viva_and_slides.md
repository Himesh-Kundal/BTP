# LLMHW Viva Voce Preparation, Slides, Speaker Notes & Certificate Page (Part I & Part II)

**Author:** Himesh Kundal (Roll No.: 23EC10030)  
**Supervisor:** Prof. Mrityunjoy Chakraborty  
**Department:** Electronics and Electrical Communication Engineering, IIT Kharagpur  

---

## Part 1: Slide Deck & Speaker Notes (12-Slide Final Presentation)

### Slide 1 — Title & Mission
- **Title:** LLMHW: End-to-End Hardware Compilation of the Complete SmolLM2-135M Transformer Decoder into Synthesizable Immutable-Weight Verilog
- **Speaker Notes:**
  > "Good morning Professor and committee members. In Part I of my B.Tech Project, I built a spatial constant-weight Verilog compiler for individual dense linear layers of SmolLM2-135M, showing a 3.85× gate reduction over programmable baselines. In Part II, I solved the full-model scaling challenge: extending the compiler to the entire 30-layer, 134.5M-parameter SmolLM2-135M autoregressive decoder—including embedding gather, RMSNorm, RoPE, causal GQA attention with on-chip KV cache, SwiGLU, residual stream, and greedy ArgMax—and proving 100% bit-for-bit token equivalence in Verilator RTL simulation while achieving 70.16% Top-1 token agreement with FP32 PyTorch."

### Slide 2 — Why Part I Spatial Unrolling Cannot Scale to 134.5M Parameters
- **Key Numbers (`EA4`, `part1_audit.md`):**
  - 7 layer projections = `106.17M` MACs/token (`78.9%` of linear MACs; `74.1%` of total decode when including the `28.31M`-MAC tied LM head).
  - Synthesizing a single $32 \times 32$ ($1,024$-weight) block spatially in Yosys requires `213,022` generic cells (`72,958` LUT-4s), extrapolating to **`27.98 Billion` gates (`~19,700 mm²` `[estimate]`)** for 134.5M weights.
- **Speaker Notes:**
  > "Why couldn't we just spatially unroll the whole model like Part I? Two reasons: First, our revised Part I audit showed that while the 7 layer projections are >90% of compute during prefill, during single-token decode the 49,152-word vocabulary LM head adds 28.31M MACs—so layer projections are 78.9% of linear MACs—and every non-linear op must execute in a closed loop on-chip. Second, Experiment EA4 proved that unrolling 1,024 weights takes 213k gates, which extrapolates to 27.98 billion gates for 134.5M weights. Shifting to a ROM-backed $8 \times 16$ sequential engine cuts logic for that same block by 9.96× (`21,387` cells) and reuses those 128 MAC lanes across all 210 projections and the LM head."

### Slide 3 — Frozen Pure-Integer Numerics & `ADR-2-2` SmoothQuant Calibration
- **Key Contracts (`numerics/config_frozen.json`, `ADR-2-1`, `ADR-2-2`):**
  - Linear weights & activations: symmetric `INT8` (`[-127, 127]`); Residual stream: saturating `INT24` ($s_{\text{res}} = 0.01$); Linear accumulators: `INT32`.
  - Non-linear units: 48-bit sum-of-squares `RMSNorm` ($256 \times 16\text{b}$ `rsqrt` LUT), `Q1.14` `RoPE` ($256 \times 32 \times 16\text{b}$), Base-2 `Softmax` ($256 \times 16\text{b}$ $2^{-F/256}$ LUT), `SwiGLU` ($512 \times 8\text{b}$ SiLU LUT, `>> 6`), scale-aligned `ArgMax`.
  - Compile-time SmoothQuant per-channel migration folds outlier channels (`L02`, `L11`, `L28`, `L29` reaching $2,630$ at `pos=0`) into `weights.hex` and `requant_params.hex` at compile time with **zero hardware overhead**.
- **Speaker Notes:**
  > "Zero floating-point exists in our hardware or integer oracle. Every real scale is compiled into an integer dyadic multiplier-shift pair $(M, s)$ using round-half-up arithmetic right shift. To handle SmolLM2's massive activation outliers—which reach 2,630 in layers 2, 11, 28, and 29 while 99% of channels are under 15—we implemented offline SmoothQuant per-channel migration under ADR-2-2, folding channel balancing factors directly into the weight ROM and dyadic requantization ROM at compile time."

### Slide 4 — Compiler Extension (`GraphV2`, ROM Packing, Microprogram, Memory Plan)
- **Key Facts (`UE.1`, `EA3`):**
  - Emits full 30-layer SmolLM2-135M hardware bundle in **`12.44s`** (`2.557 GB` peak RSS).
  - Immutable ROMs: `157.54 MB` (`98.15%` of on-chip bits) — `155.25 MB` weights (`10,174,464` $\times$ 128-bit words), `2.20 MB` requant params (`288,990` $\times$ 64-bit words), `88 KB` tables, `4.1 KB` microprogram/bases.
  - Dynamic SRAM (`MAX_CTX=256`): `2.97 MB` (`1.85%`) — `2.81 MB` KV cache + `0.16 MB` activation buffers.
- **Speaker Notes:**
  > "The compiler lowers HuggingFace weights into GraphV2 without importing PyTorch anywhere in `ir/`, `compiler/`, `backend/`, `numerics/`, or `verification/`. It packs 157.54 MB of immutable ROM images with SHA-256 verification and emits a looped 19-instruction microprogram."

### Slide 5 — Hardware Accelerator Microarchitecture (`top_accel.v`)
- **Key Facts (`EA1`):**
  - 12 hand-written parameterized Verilog-2001/2005 units (`78,510` total generic cells, `3,661` DFFs; `31,908` iCE40 LUT-4s).
  - `linear_engine.v` ($8 \times 16 = 128$ MAC lanes): `21,387` generic cells (`503` DFFs).
  - All weights/tables/params in `rom_sync.v` with **zero write ports**.
- **Speaker Notes:**
  > "At the hardware level, `top_accel.v` pairs a microprogram sequencer with an $8 \times 16$ streaming MAC engine, dedicated non-linear units, 10 activation buffers, and a 3:1 GQA KV cache. The entire compute datapath synthesizes to just 78,510 generic gates."

### Slide 6 — Verification Ladder, Audits & 100% Mutation Kill Rate
- **Key Facts (`UD.3`–`UD.6`, `UC.13`):**
  - Dual independent software oracles (`numerics/decoder.py` == `CompiledArtifactInterpreter`).
  - Tiny model: `100` random prompts token-exact & trace-exact; Icarus Verilog == Verilator traces bit-identical.
  - Mutation suites: **`9/9` Part I (`M1`–`M9`)**, **`21/21` Part II unit mutants (`M10`–`M30`)**, and **`7/7` top-level system mutants (`TM1`–`TM7`)** — **100.0% killed**.
- **Speaker Notes:**
  > "How do we know the verification itself isn't vacuous? First, structural audits prove `top_accel.v` has zero write ports on any ROM and zero expected tokens or oracle values anywhere in RTL. Second, we injected 21 unit-level RTL faults and 7 system-level faults—including causal mask off-by-one, GQA head map corruption, and ArgMax tie-break inversion—and our test suite killed 100% of them."

### Slide 7 — Claim C1′: Full 30-Layer SmolLM2-135M Bit-Exact RTL Proof (`UE.3`–`UE.5`)
- **Key Facts (`full_decode.json`, `ue3_layer0_traces.json`, `ue5_long_context.json`):**
  - `UE.3` (Layer 0, 12 positions): `216` files, **`728,064` elements**, `0` divergences.
  - `UE.4` (Full 30 layers, 10 prompts $\times 16$ tokens = `285` decode steps, `4.02B` cycles): **`10/10` prompts 100% token-exact**, **`2,279,424` sampled trace elements (`L0, L14, L29`) 100% bit-exact**, **`14,100,262` cycles/token**.
  - `UE.5` (Long context at `MAX_CTX = 256`, `4.67B` cycles): **256/256 positions token-exact & trace-exact** (`303,360` elements).
- **Speaker Notes:**
  > "This is our core hardware correctness proof, Claim C1′. Across 10 real prompts generating 16 tokens each through all 30 transformer layers in Verilator—over 4.02 billion simulated clock cycles and 2.28 million intermediate tensor elements—plus a 256-token max-context run of 4.67 billion cycles, the Verilog hardware matches the independent integer oracle with zero divergences."

### Slide 8 — Claim C2′: Quantization Quality & Live RTL Text Generation (`UE.6`, `UE.7`)
- **Key Facts (`quality_final.json`, `ue6_host_demo.json`, `demo2.json`):**
  - WikiText-2 (10 windows, 630 tokens): **FP32 PPL = `89.6749` vs INT8 PPL = `107.0031` ($\Delta \text{PPL} = +17.3282 \le 25.0$)**, **Top-1 ArgMax Agreement = `70.16%` ($\ge 35.0\%$)**, **Mean KL = `0.2280` ($\le 0.25$)**.
  - Live RTL outputs:
    - *"The capital of France is Paris. The capital of Germany is"* $\to$ **`" Berlin. The capital of Italy is Rome. The capital of Spain is Madrid."`** (**16/16 tokens identical to FP32**).
    - *"According to Einstein's theory of relativity, E ="* $\to$ **`" mc2, where E is the energy, and m is the mass.\n"`**.
- **Speaker Notes:**
  > "Claim C2′ links that bit-exact hardware directly to language modeling quality. On WikiText-2, our pure-INT8 hardware achieves 70.16% Top-1 token agreement with FP32 PyTorch and a perplexity delta of +17.33, well within the frozen Gate HA thresholds, and generates fluent English and Python code directly from Verilator RTL simulation."

### Slide 9 — Architecture & Synthesis Study (`EA2` Grid & `EA5` 28nm ASIC Projection)
- **Key Facts (`EA2.json`, `EA5.json`, all ASIC numbers `[estimate]`):**
  - Analytical cycle model (`13,691,648` cycles/tok) matches simulated cycles (`14,100,262`) within **`-2.90%`**.
  - TSMC 28nm HPC+ pre-layout projection (`MAX_CTX=256`): **`147.95 mm²` Mid die area** (`115.65–191.50 mm²` `[estimate]`), **`96.55%` ROM-dominated** (`142.84 mm²` ROM, `3.68 mm²` SRAM, `0.06 mm²` logic).
  - Throughput at 400 MHz: **`28.37 tokens/s`** (`35.25 ms/tok` `[estimate]`) at 128 MAC lanes, scaling to **`63.37 tokens/s`** (`15.78 ms/tok` `[estimate]`) at 512 MAC lanes with only **`+0.10%` die area increase**.
- **Speaker Notes:**
  > "Because 98.15% of on-chip bits are immutable ROM and logic is only 0.04% of die area (`0.06 mm²` out of `147.95 mm²` in 28nm), sensitivity analysis shows die area is 96.55% elastic to ROM bit density and almost completely inelastic to MAC lane count. We can quadruple the linear engine from 128 to 512 MAC lanes to reach 63.37 tokens/sec at 400 MHz while adding just 0.15 mm² to the die."

### Slide 10 — Live Demo & Turnkey Reproducibility (`make demo2-fast` & `make demo2`)
- **Commands & Runtimes:**
  - `make demo2-fast` (`11.45s`): Compiles tiny 2-layer decoder, runs Verilator RTL + interpreter (`115,227` cycles), passes all 4 structural/ROM audits, and prints `21/21` unit + `7/7` system mutation kill matrices.
  - `make demo2` (`106.16s`): Runs 30-layer SmolLM2-135M Verilator RTL generation (`221,848,192` cycles) producing `' Berlin. The capital'` (100% bit-exact with oracle and 4/4 match with FP32).

---

## Part 2: Viva Voce Q&A — Complete Answers to All 8 Checklist §8 Explain-It Questions

### Q1: Explain the full decode step op by op, with the formats on every edge.
**Answer:**
For a single token input `tok_in` (`uint16` $\in [0, 49151]$) at sequence position `pos` ($\in [0, 255]$):
1. **`OP_EMBED` (`embed_gather.v`):** Reads 576 `INT8` weights for row `tok_in` from `rom_weights_inst`, multiplies by $(M_{\text{embed}}, s_{\text{embed}})$ from `rom_params_inst`, and writes 576 `INT24` elements (`[-8388608, 8388607]`, scale $s_{\text{res}} = 0.01$) into `buf_res`.
2. **Layer Loop (`layer_idx = 0 .. 29`, driven by `sequencer.v`):**
   - **`OP_RMSNORM` (Pre-Attention, `rmsnorm.v`):** Reads 576 `INT24` elements from `buf_res`, accumulates $\sum x_i^2$ in **48-bit unsigned**, multiplies by `INV_D_M = 7282 >> 22`, normalizes by an even shift `calc_sh_amt` into `[1, 255]` to index the $256 \times 16\text{b}$ `rsqrt_lut`, shifts Stage 1 by `calc_sh_amt >> 1`, applies per-channel $\gamma_i$ (`(M_i, 19 + s_i)`), and writes 576 `INT8` (`[-127, 127]`) elements to `buf_act`.
   - **`OP_LINEAR` (`q_proj`, `k_proj`, `v_proj` via `linear_engine.v`):** Streams `INT8` `buf_act` against `INT8` weights in $8 \times 16$ tiles into **32-bit signed accumulators**, requantizes each output channel via $(M_i, s_i)$ (`requant.v`, round-half-up + symmetric saturation to `[-127, 127]`), writing `buf_q` (`576` $\times$ `INT8`), `buf_k` (`192` $\times$ `INT8`), and `buf_v` (`192` $\times$ `INT8`).
   - **`OP_ROPE` (`rope.v`):** Rotates pairs $(x_i, x_{i+32})$ for each of the 9 Q heads (`buf_q`) and 3 KV heads (`buf_k`) using `Q1.14` (`16-bit signed`) $\cos/\sin$ entries at row `pos`: `rshift_rnd(x*cos ∓ y*sin, 14)` $\to$ `INT8`.
   - **`OP_ATTENTION` (`attention.v` + `softmax.v` + `kv_cache.v`):**
     - Writes rotated `buf_k` and raw `buf_v` (`192` $\times$ `INT8` each) into `kv_cache.v` at `(layer_idx, pos)`.
     - For each query head $h_q \in [0, 8]$ mapped to KV head $h_{kv} = h_q // 3$: computes causal dot products $S_t = (\mathbf{q} \cdot \mathbf{k}_t) \gg 1$ in `INT32` for $t \in [0, \text{pos}]$.
     - `softmax.v` finds $S_{\max}$, decomposes $\Delta_t = S_{\max} - S_t = 256 I_t + F_t$, looks up $e_t = \text{exp2\_lut}[F_t] \gg \min(16, I_t)$ (`18-bit`), sums $\Sigma = \sum e_t$, and computes `Q0.15` probabilities $P_t = \lfloor (e_t \cdot 32767) / \Sigma \rfloor$ (`16-bit`).
     - Computes weighted value sum $\text{ctx}_d = \text{sat}_{\text{int8}}(\text{rshift\_rnd}(\sum_t P_t V_{t,d}, 15))$ into `buf_attn_out` (`576` $\times$ `INT8`).
   - **`OP_LINEAR` (`o_proj`):** Projects `buf_attn_out` (`576` $\times$ `INT8`) to `buf_act` (`576` $\times$ `INT24` in residual scale $s_{\text{res}} = 0.01$).
   - **`OP_RESIDUAL_ADD` (`residual_add.v`):** Saturating 24-bit addition `buf_res <= sat_int24(buf_res + buf_act)`.
   - **`OP_RMSNORM` (Pre-MLP):** Normalizes `buf_res` (`INT24`) $\to$ `buf_act` (`576` $\times$ `INT8`).
   - **`OP_LINEAR` (`gate_proj`, `up_proj`):** Projects `buf_act` to `buf_gate` (`1536` $\times$ `INT8`) and `buf_up` (`1536` $\times$ `INT8`).
   - **`OP_SWIGLU` (`swiglu.v`):** Indexes `silu_lut[gate + 256]` (`INT8`), multiplies by `up` (`INT8` $\to$ `INT16`), shifts right by 6 with round-half-up (`rshift_rnd(prod, 6)`), and saturates to `buf_swiglu` (`1536` $\times$ `INT8`).
   - **`OP_LINEAR` (`down_proj`) + `OP_RESIDUAL_ADD`:** Projects `buf_swiglu` (`1536` $\times$ `INT8`) to `INT24` and adds into `buf_res` (`576` $\times$ `INT24`).
3. **Final Head (`final_norm` $\to$ `lm_head` $\to$ `argmax`):**
   - `final_norm` normalizes `buf_res` (`INT24`) $\to$ `buf_act` (`576` $\times$ `INT8`).
   - `lm_head` projects `buf_act` against the $49,152 \times 576$ tied weight matrix into `buf_logits` (`49,152` $\times$ `INT24`).
   - `argmax.v` scans all `49,152` logits, choosing the lowest token ID with the maximum value, and emits `tok_out` (`uint16`) with `out_valid = 1`.

### Q2: Explain why each non-linear op is implemented the way it is and how its error was measured.
**Answer:**
- **RMSNorm:** Standard $1/\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}$ has a wide dynamic range across 24-bit residuals. Using an even Leading-Zero-Count shift `calc_sh_amt = 2k` normalizes the mean square into $[1, 255]$ so a tiny 256-entry $\times$ 16-bit LUT suffices, while $\sqrt{2^{2k}} = 2^k$ becomes an exact bit-shift `calc_sh_amt >> 1`.
- **RoPE:** Precomputing $\cos(p \theta_i)$ and $\sin(p \theta_i)$ in `Q1.14` (`16-bit signed`) for `MAX_CTX = 256` takes only $256 \times 32 \times 2 \times 2 = 32\text{ KB}$ of ROM and eliminates runtime CORDIC or Taylor polynomials.
- **Softmax:** Base-2 decomposition $e^{-x} = 2^{-x \log_2 e} = 2^{-(I + F/256)} = 2^{-F/256} \gg I$ folds $\log_2 e$ into the $Q/K$ scale product (`ADR-2-2`), replacing floating-point `exp()` with a 256-entry negative-fractional LUT and a barrel shifter.
- **SwiGLU:** Since `gate` is already an 8-bit integer in $[-127, 127]$, a 512-entry $\times$ 8-bit table directly tabulates $\text{SiLU}(g)$ for every possible input value with zero approximation beyond 8-bit output quantization.
- **Error Measurement:** Measured via single-op isolation ablations (`results/part2/ua5_ablation.csv`) and end-to-end WikiText-2 sweeps (`results/part2/numerics_sweep.csv`, `results/part2/quality_final.json`).

### Q3: Explain why exactness (C1′) and quality (C2′) are separate claims and what links them.
**Answer:**
- **Claim C1′ (Hardware Implementation Exactness)** states that the Verilog RTL (`top_accel.v`) executes the frozen integer specification (`CompiledArtifactInterpreter` / `numerics/`) **with zero bit discrepancies**. It is a deterministic digital verification property (proven across `285` full decode steps and `2.28M` trace elements in `UE.4`, plus `256` positions in `UE.5`).
- **Claim C2′ (Numerical Approximation Quality)** states that the frozen integer specification accurately approximates the real-valued PyTorch FP32 model on language modeling benchmarks (`70.16%` Top-1 agreement, $\Delta \text{PPL} = +17.33$ on WikiText-2 in `UE.7`).
- **What links them:** Because `UE.4` proves $\text{RTL}(\mathbf{x}) \equiv \text{Interpreter}(\mathbf{x})$ bit-for-bit on the exact same compiled ROM bundle `build/smollm2_hw/`, the quality numbers measured on `CompiledArtifactInterpreter` in `UE.7` are mathematically guaranteed to be the exact quality of the Verilog hardware accelerator.

### Q4: Explain the oracle structure and why the verification can be trusted (interpreter, traces, mutants, audits).
**Answer:**
1. **Two-tier software oracle:** `numerics/decoder.py` defines the pure mathematical integer ops; `compiler/interpreter.py` (`CompiledArtifactInterpreter`) executes *only* the emitted hex ROM files (`weights.hex`, `requant_params.hex`, `tables.hex`) and `schedule.json`. We first prove `Interpreter == numerics/decoder.py`, then prove `Verilog RTL == Interpreter`.
2. **Intermediate trace taps:** Rather than checking only final token IDs (which could mask cancelling errors), we dump and compare 18 intermediate tensor traces per layer (`2,279,424` elements in `UE.4`; `728,064` in `UE.3`).
3. **100% Mutation Kill Rate:** We prove the tests have zero blind spots by injecting 21 unit faults (`M10`–`M30`) and 7 system faults (`TM1`–`TM7`) and verifying 100% are caught.
4. **Structural & Cryptographic Audits:** `weight_audit_v2.py` and `structural_audit.py` verify SHA-256 hashes of all ROM hex images, zero write ports on ROMs, and zero oracle/expected tokens in RTL.

### Q5: Explain why the full model uses a ROM-based engine and what the spatial design from Part I contributes.
**Answer:**
- **Why ROM-based for the full model:** As shown in `EA4`, a flat spatial unrolling of $1,024$ weights takes `213,022` generic cells (`72,958` LUT-4s) and 312 seconds of ABC synthesis. Extrapolating to $134.5\text{M}$ weights yields `27.98 Billion` gates (`~19,700 mm²` `[estimate]`), which is impossible to synthesize or fabricate on a single die. Storing weights in high-density synchronous ROM (`0.105 µm²/bit` in 28nm) fits all 157.54 MB of weights and parameters in `142.84 mm²` `[estimate]` while time-multiplexing a compact `21,387`-cell ($8 \times 16$) MAC engine.
- **What Part I contributes:** Part I established our symmetric INT8 numeric contract, proved that weight immutability eliminates register/write-bus overhead ($3.85\times$ gate and $17.75\times$ DFF reduction), provided the verification and mutation-testing architecture, and remains the optimal spatial execution paradigm for small fixed sub-blocks or high-throughput parallel prefill engines.

### Q6: Explain the ASIC projection: its inputs, its ranges, what dominates, and what it does not show.
**Answer:**
- **Inputs (`EA5.json`):** Synthesized Yosys gate/DFF counts (`EA1`, `EA2`), exact ROM (`157.54 MB`) and SRAM (`2.97 MB`) bit counts (`EA3`), measured cycle counts (`14.10M` cycles/tok), and published TSMC 28nm HPC+ bit/gate densities (`ROM: 0.085 / 0.105 / 0.130 µm²/bit`, `6T SRAM: 0.125 / 0.155 / 0.190 µm²/bit`, `Logic: 1.80M / 1.42M / 1.10M GE/mm²`, `F_clk: 250 / 400 / 550 MHz`).
- **Ranges (128-lane baseline):** Die area = **`115.65` (Low) / `147.95` (Mid) / `191.50` (High) $\text{mm}^2$** `[estimate]`; Throughput = **`17.73` (Low) / `28.37` (Mid) / `39.01` (High) tokens/s** `[estimate]`.
- **What dominates:** Immutable ROM macros account for **`96.55%` of total die area** (`142.84 mm²`), KV/activation SRAM is `2.49%` (`3.68 mm²`), and the entire compute logic datapath is just **`0.04%` (`0.06 mm²`)**.
- **What it does not show:** Physical place-and-route congestion, multi-banked ROM macro aspect-ratio floorplanning, long-wire RC delay between ROM banks and the central MAC engine, clock-tree power, and IR drop.

### Q7: Demonstrate running a demo, opening a trace diff, and inspecting a ROM image unaided.
**Answer (Live Command Cheat-Sheet):**
1. **Run fast tiny demo (~11.5s) or full SmolLM2 demo (~106s):**
   ```bash
   make demo2-fast
   make demo2
   ```
2. **Inspect a ROM hex image (`weights_128b.hex`, `requant_params.hex`, `tables_silu.hex`):**
   ```bash
   head -n 5 build/smollm2_hw/weights_128b.hex      # 16 packed INT8 weights (32 hex chars = 128 bits) per line
   head -n 5 build/smollm2_hw/requant_params.hex    # 64-bit {16'd0, shift[15:0], mult[31:0]} per channel
   jq '.weights.q_proj_0, .requant_params.q_proj_0' build/smollm2_hw/address_map.json
   ```
3. **Run `verification/trace_diff.py` to locate a first divergence:**
   ```bash
   uv run python -c "from verification.trace_diff import compare_trace_directories; print(compare_trace_directories('build/ue3_golden_traces', 'build/ue3_rtl_traces'))"
   ```

### Q8: State what a real chip would still need (place-and-route, memory macros, clocking, power, test) and what Part III would be.
**Answer:**
To tape out `top_accel.v` as a real silicon ASIC in Part III:
1. **Memory Macro Banking:** Replace behavioral `$readmemh` `rom_sync.v` arrays with tiled Foundry via-ROM compiler macros (e.g., $256 \times 512\text{ KB}$ banks with hierarchical address decoders and pipeline registers) and 6T SRAM compiler macros for `kv_cache.v`.
2. **Timing-Driven Place & Route:** Synthesize with commercial liberty (`.lib`) timing views, pipeline the Combinational Stage-1 LZC/divider path in `rmsnorm.v` (`LTP = 2077`) and `softmax.v` (`LTP = 662`) across 2–3 clock cycles to close timing at $\ge 400\text{ MHz}$.
3. **Clocking, Power & DFT:** Insert clock trees (CTS), multi-VDD power grids, scan chains (DFT/JTAG), and Built-In Self-Test (MBIST) for the KV cache SRAMs, followed by DRC, LVS, STA, and PrimeTime PX power signoff.

---

## Part 3: Supervisor & Student Certificate Page (Prepared for Human Signatures)

```text
================================================================================
                INDIAN INSTITUTE OF TECHNOLOGY KHARAGPUR
  Department of Electronics and Electrical Communication Engineering
================================================================================

                               CERTIFICATE

This is to certify that the B.Tech. Project thesis entitled:

  "LLMHW: End-to-End Hardware Compilation of the Complete SmolLM2-135M
   Transformer Decoder into Synthesizable Immutable-Weight Verilog"

submitted by:

  Himesh Kundal (Roll No.: 23EC10030)

in partial fulfillment of the requirements for the award of the degree of
Bachelor of Technology in Electronics and Electrical Communication Engineering
at the Indian Institute of Technology Kharagpur, is an authentic record of
bonafide research work carried out by him under my supervision and guidance
during the academic session 2026–2027.

All experimental measurements, RTL simulations, synthesis benchmarks, and
AI-assistance disclosures included in this thesis have been reviewed and
verified in accordance with institute guidelines.


Date: ___________________                  Place: IIT Kharagpur


_________________________________          _________________________________
Himesh Kundal                              Prof. Mrityunjoy Chakraborty
(Roll No.: 23EC10030)                      Project Supervisor
B.Tech. Candidate                          Department of E&ECE
IIT Kharagpur                              IIT Kharagpur
================================================================================
```
