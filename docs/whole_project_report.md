# LLMHW: End-to-End Hardware Compilation of the Complete SmolLM2-135M Transformer Decoder into Synthesizable Immutable-Weight Verilog

**Author:** Himesh Kundal (Roll No.: 23EC10030)  
**Supervisor:** Prof. Mrityunjoy Chakraborty  
**Department:** Department of Electronics and Electrical Communication Engineering, Indian Institute of Technology Kharagpur  
**Project:** B.Tech Project (Combined Part I & Part II Final Report)  
**Date:** November 19, 2026  
**Target Model:** `HuggingFaceTB/SmolLM2-135M` (`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`)  
**Traceability Table:** [`docs/report_traceability.md`](report_traceability.md)  

---

## Abstract

Autoregressive inference in modern Large Language Models (LLMs) is fundamentally memory-bandwidth bound ($O(1)$ operations per weight byte fetched), spending the majority of its silicon area and dynamic energy moving stationary parameters across programmable memory hierarchies. In embedded, edge-robotics, and dedicated appliance deployments where a foundation model remains fixed post-training, general-purpose programmable NPUs pay severe area and routing penalties for runtime weight write buses, writable register files, and floating-point datapaths.

This two-part thesis presents **LLMHW**, an end-to-end model-specific hardware compiler targeting **SmolLM2-135M** ($134,515,008$ parameters, $30$ transformer layers, $d_{\text{model}} = 576$, $d_{\text{ffn}} = 1536$, GQA $9:3$ attention, SwiGLU MLP, $49,152$ tied vocabulary).

In **Part I**, we developed a spatial constant-weight compiler for individual dense linear projections (**Design F**) that embeds symmetric INT8 weights directly into combinational shift-add networks with zero runtime weight input ports (Invariant **I3**). Across Yosys synthesis benchmarks (`E1`–`E5`), Design F achieved a **3.75× to 3.85× standard-cell reduction**, up to **17.75× flip-flop reduction**, and **20% to 31% topological critical-path shortening** over a width-matched programmable baseline (**Design P**), while proving **100% bit-exact equivalence** (**Claim C1**) across 124 regression suites and a **100% mutation kill rate** (`9/9` mutants `M1`–`M9`).

In **Part II**, we re-evaluated the full-model scaling boundary: because the 7 layer projections account for **78.9% of linear MACs** (`74.1%` of total decode arithmetic when including the $28.31\text{M}$-MAC tied LM head, and `>90%` only during multi-token prefill), and because flat spatial unrolling of all $134.5\text{M}$ weights extrapolates to **`27.98 Billion` logic gates** ($\approx 19,698\,\text{mm}^2$ `[estimate]`), a full-model single-die accelerator requires a **ROM-backed time-multiplexed architecture** coupled with integer-only non-linear units. We extended LLMHW to compile the **complete 30-layer SmolLM2-135M decoder**—embedding gather, 48-bit accumulator RMSNorm, Q1.14 rotate-half RoPE, causal GQA attention with base-2 LUT softmax and on-chip KV cache SRAM, SwiGLU, 24-bit saturating residual stream, tied LM head, and scale-aligned greedy ArgMax—into a self-contained Verilog accelerator (`top_accel.v`) whose weights and dyadic requantization parameters reside in immutable synchronous ROMs (`rom_sync.v`) with zero write ports.

We prove two core scientific claims in Part II:
1. **Claim C1′ (Full-Decoder Bit-Exact RTL Equivalence):** Across 10 evaluation prompts $\times 16$ generated tokens (`285` full 30-layer decode steps, `4,018,574,895` cycles) and a long-context run up to `MAX_CTX = 256` (`4,669,315,072` cycles), the Verilator-simulated Verilog hardware produces **100% bit-identical output tokens and intermediate layer traces** (`2,279,424` sampled elements in `UE.4`, `728,064` Layer-0 elements in `UE.3`, `303,360` long-context elements in `UE.5`; **0 divergences**) to an independent pure-integer reference oracle (`CompiledArtifactInterpreter`). The verification suite kills **100.0% of 21 unit mutants (`M10`–`M30`)** and **100.0% of 7 system mutants (`TM1`–`TM7`)**.
2. **Claim C2′ (Integer Model Quality vs. FP32):** Using compile-time SmoothQuant per-channel scale migration and aligned non-linear LUTs (`ADR-2-2`), the 100% integer W8A8/INT24 hardware model achieves **70.16% Top-1 ArgMax token agreement** (threshold $\ge 35.0\%$), a perplexity delta of **$\Delta \text{PPL} = +17.3282$** (`89.6749` FP32 vs. `107.0031` INT8; threshold $\le 25.0$), and **0.2280 mean KL divergence** across 10 WikiText-2 test windows (`630` tokens), generating coherent English text and Python code in RTL simulation.

Finally, our architecture and synthesis study (`EA1`–`EA5`) demonstrates that the entire 12-unit compute datapath synthesizes to just **`78,510` generic logic cells** (`3,661` DFFs), while on-chip storage at `MAX_CTX = 256` is **98.15% immutable ROM (`157.54 MB`)** and **1.85% dynamic SRAM (`2.97 MB`)**. An analytical pre-layout projection to **TSMC 28nm HPC+** (`EA5`) estimates a **`147.95 mm²` die area** (`[estimate]`, $96.55\%$ ROM-dominated) delivering **`28.37 tokens/s` at 400 MHz** (`[estimate]`) for the baseline 128-MAC engine ($14,100,262$ measured cycles/token; analytical model error **`-2.90%`**) and scaling to **`63.37 tokens/s` at 512 MAC lanes** (`[estimate]`) with only a `+0.10%` die area increase.

---

## 1. Introduction and Motivation

Autoregressive language generation computes $P(x_t \mid x_{<t})$ one token at a time. Because each decode step reuses the entire $134.5\text{M}$-parameter weight state for a single vector activation ($B=1$), conventional processors suffer from the von Neumann bottleneck: fetching weights from off-chip DRAM or multi-ported on-chip SRAM consumes orders of magnitude more energy and area than the arithmetic itself.

This project asks a foundational computer architecture question across two phases:
- **Part I Question:** *When a linear projection's weights are frozen at compile time, how much logic area, sequential storage, and critical-path delay can be eliminated by compiling weights directly into hardwired constant-coefficient Verilog?*
- **Part II Question:** *How do we scale model-specific hardware compilation from isolated linear tiles to a complete, self-contained 30-layer transformer decoder (including every non-linear operator, KV cache, and vocabulary head) that fits on a realistic die, executes 100% in fixed-point integer arithmetic, and generates bit-exact tokens verified in RTL simulation?*

---

## 2. Background

### 2.1 Transformer Decoder Architecture (`SmolLM2-135M`)
`HuggingFaceTB/SmolLM2-135M` (pinned revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`) is a Llama-family decoder-only transformer with the following architectural parameters (`docs/model_facts.md`, `numerics/config_frozen.json`):
- **Layers ($L$):** `30`
- **Hidden Dimension ($d_{\text{model}}$):** `576`
- **Attention Heads:** `9` query heads ($n_q = 9$), `3` key-value heads ($n_{kv} = 3$, Grouped-Query Attention ratio $3:1$), head dimension $d_h = 64$.
- **Rotary Position Embedding (RoPE):** Rotate-half convention (`[0 .. 31]` paired with `[32 .. 63]`), base $\theta = 100,000.0$.
- **MLP (SwiGLU):** Intermediate dimension $d_{\text{ffn}} = 1536$, zero bias across all linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Normalization:** Pre-attention, pre-MLP, and final RMSNorm with $\epsilon = 10^{-5}$.
- **Vocabulary & Tied Embeddings:** `vocab_size = 49,152`; the input embedding table (`embed_tokens`, $49,152 \times 576$) is tied to the output language modeling head (`lm_head`).
- **Total Parameters:** `134,515,008` ($106,168,320$ across the 210 layer projections + $28,311,552$ in the tied embedding/LM head + $35,136$ in the 61 RMSNorm $\gamma$ vectors).

### 2.2 Pure-Integer Inference and Dyadic Scaling
To eliminate IEEE-754 floating-point units completely from the hardware inference path (Golden Rule 8), all real-valued scales $s \in \mathbb{R}^+$ are compiled offline into integer dyadic multiplier-shift pairs $(M, s_{\text{shift}})$ such that $s \approx M \cdot 2^{-s_{\text{shift}}}$, evaluated in hardware via round-half-up arithmetic right shift (`numerics.vh`):
$$\text{rshift\_rnd}(x, s) = \begin{cases} x & \text{if } s \le 0 \\ (x + (1 \ll (s - 1))) \gg s & \text{if } s > 0 \end{cases}$$
followed by symmetric saturation into $[-127, 127]$ (`INT8`) or $[-8388608, 8388607]$ (`INT24`).

---

## 3. Problem Statement, Scope, and Non-Goals

### 3.1 Formal Problem Statement
Design, implement, and verify an automated Python-to-Verilog hardware compiler (`llmhw`) that ingests `SmolLM2-135M`, quantizes and calibrates all linear and non-linear operators into a frozen pure-integer specification (`numerics/`), packs all weights, lookup tables, and dyadic scales into immutable ROM images, schedules a deterministic microprogram, and emits synthesizable Verilog-2001/2005 (`top_accel.v`) that executes end-to-end token-in to token-out autoregressive decoding (`tok_in` $\to$ `tok_out`).

### 3.2 Explicit Non-Goals
- **Floating-Point Hardware:** Zero floating-point operators exist in RTL or the integer inference oracle.
- **Runtime Weight Writes:** All weight, table, and parameter memories are strictly read-only ROMs with zero write ports (Golden Rule 7).
- **Host Tokenization & Sampling:** BPE string tokenization runs on the host CPU; decoding is strictly deterministic greedy argmax (`batch_size = 1`).
- **Physical Silicon Tapeout:** Area and timing in 28nm CMOS are pre-layout analytical projections clearly labeled as `[estimate]`.

---

## 4. Part I: Spatial Linear-Layer Compiler (Revised & Audited)

### 4.1 Spatial Constant-Weight Compilation (Design F vs. Design P)
In Part I, individual linear layers $Y = X W^T$ were tiled into $(N_t \times K_t)$ sub-matrices and compiled into combinational shift-add reduction trees (**Design F**) where zero weights ($w = 0$) are pruned and non-zero weights ($w \in [-127, 127]$) are emitted as signed literals (`8'sd...`). We compared Design F against a width-matched programmable baseline (**Design P**) containing an $N_t \times K_t \times 8$-bit writable weight register file ($W_{\text{reg}}$) and runtime Baugh-Wooley multipliers under a strict fairness contract ($L=1$ cycle latency, identical bitwidths, Yosys hierarchy flattening).

### 4.2 Part I Experimental Summary (`E1`–`E5`)
| Tile Shape $(N_t \times K_t)$ | Design F Cells | Design P Cells | Cell Reduction | Design F DFFs | Design P DFFs | DFF Reduction | LTP Depth (F vs P) |
|---|---|---|---|---|---|---|---|
| $(4 \times 16)$ | `5,690` | `21,399` | **3.76×** | `114` | `641` | **5.62×** | `41` vs `59` (**0.69**) |
| $(8 \times 16)$ | `22,764` | `85,283` | **3.75×** | `226` | `1,293` | **5.72×** | `57` vs `77` (**0.74**) |
| $(16 \times 16)$ | `38,912` | `149,768` | **3.85×** | `442` | `2,561` | **5.79×** | `69` vs `92` (**0.75**) |
| $(16 \times 64)$ | `155,648` | `599,085` | **3.85×** | `491` | `8,705` | **17.73×** | `89` vs `111` (**0.80**) |
| $(32 \times 64)$ | `311,296` | `1,198,170` | **3.85×** | `981` | `17,409` | **17.75×** | `89` vs `111` (**0.80**) |

Real trained SmolLM2 weights synthesized to **23.0% fewer cells** (`22,764`) than uniform random INT8 weights (`28,006 ± 48`, `E2`) due to weight clustering near zero. Across all 210 linear projections (`results/quant_quality.json`), per-channel INT8 quantization achieved a mean cosine similarity of **`0.9996`** (min `0.9984`) and mean relative Frobenius error of **`3.81%`** (max `5.60%`).

### 4.3 Revised Part I Arithmetic Share & Scaling Audit (Task `U0.1`)
During the Part II bootstrap audit (`results/part2/part1_audit.md` §5.4), we re-derived the exact per-token arithmetic share of the 7 layer projections in SmolLM2-135M:
- **30 Layers (7 projections: $Q, K, V, O, \text{Gate}, \text{Up}, \text{Down}$):** $30 \times 3,538,944 = \mathbf{106,168,320\text{ MACs/token}}$.
- **Tied LM Head ($49,152 \times 576$):** $\mathbf{28,311,552\text{ MACs/token}}$.
- **Causal Attention Dot-Products (at $S = 256$):** $\approx \mathbf{8,847,360\text{ ops/token}}$.
- **Revised Arithmetic Share:**
  1. For single-token autoregressive decoding ($T = 1$) including the vocabulary LM head, the 7 layer projections represent **78.9% of all linear MACs** ($106.17\text{M} / 134.48\text{M}$) and **74.1% of total decode arithmetic** ($106.17\text{M} / 143.33\text{M}$).
  2. The `>90%` (`>99%` of non-vocab layer compute) share holds **only during prompt prefill** where large sequence lengths amortize or omit intermediate vocabulary projections, or when the LM head is excluded.
- Furthermore, as proven in Experiment `EA4`, spatially unrolling all $134.5\text{M}$ weights into flat combinational logic would require **`27.98 Billion` generic gates** (`[estimate]`), motivating the Part II ROM-backed sequential engine.

---

## 5. Part II: Frozen Integer Numerics, Non-Linear Operators, and Quality (`numerics/`)

### 5.1 Frozen Integer Formats (`numerics/config_frozen.json`, `ADR-2-1`, `ADR-2-2`)
| Signal / Table | Bitwidth & Sign | Format / Range | Quantization & Rounding Rule |
|---|---|---|---|
| **Linear Weights ($W$)** | `8`-bit signed | `[-127, 127]` | Per-output-channel symmetric INT8, round-half-even |
| **Linear Activations ($X$)** | `8`-bit signed | `[-127, 127]` | Static symmetric INT8 via dyadic $(M_i, s_i)$, round-half-up |
| **Residual Stream ($r$)** | `24`-bit signed | `[-8388608, 8388607]` | Fixed residual scale $s_{\text{res}} = 0.01$, saturating addition |
| **Linear Accumulator** | `32`-bit signed | `INT32` | Exact $\sum_{k} x_k w_{i,k}$ before dyadic requantization |
| **RMSNorm Internals** | `48`-bit sum-of-squares | $256 \times 16\text{b}$ `rsqrt` LUT | `INV_D_M = 7282, INV_D_S = 22`; even LZC shift + Stage-2 $\gamma_i$ |
| **RoPE Tables** | `16`-bit signed | `Q1.14` (`[-16384, 16384]`) | $256 \times 32$ $\cos/\sin$ tables; `(x*cos ∓ y*sin) >> 14` |
| **Softmax Internals** | `32`-bit score $\to$ `16`-bit prob | `Q0.15` (`[0, 32767]`) | Base-2 split $\Delta = 256 I + F$; $256 \times 16\text{b}$ $2^{-F/256}$ LUT |
| **SwiGLU Internals** | `8`-bit gate/up $\to$ `8`-bit out | $512 \times 8\text{b}$ SiLU LUT | `(silu_lut[gate + 256] * up) >> 6` with round-half-up |
| **Greedy ArgMax** | `24`-bit logit $\to$ `16`-bit ID | `[0, 49151]` | Per-row dyadic scale alignment; lowest token ID tie-break |

### 5.2 Non-Linear Operator Designs and `ADR-2-2` Resolution
During full-model quality testing (`BLOCKER-UE7-01`), we diagnosed and resolved four critical numeric interactions in 30-layer deep INT8 transformers (`ADR-2-2`):
1. **RMSNorm Dynamic Exponent & Mantissa LUT Alignment (`Rule N2-6`):** `rmsnorm.v` normalizes the 32-bit mean square by an even bit-shift `calc_sh_amt` into `[1, 255]`, indexes `rsqrt_lut[k] = min(65535, round(2^19 / sqrt(k)))`, shifts Stage 1 by `exp_sh = calc_sh_amt >> 1`, and applies per-channel $\gamma_i$ (`(M_i, 19 + s_i)`) in Stage 2.
2. **Monotonic Base-2 Softmax LUT (`Rule N2-10`):** Attention scores $S_j = (\mathbf{q} \cdot \mathbf{k}_j) \gg 1$ subtract $S_{\max}$ to form $\Delta_j = S_{\max} - S_j = 256 I_j + F_j \ge 0$. `build_exp2_fraction_lut` evaluates $2^{-F/256}$ in Q0.16 (`[32857, 65536]`) so `exp2_lut[F] >> I` is strictly monotonic across integer boundaries.
3. **Exact SwiGLU Scale Propagation (`Rule N2-11`):** Because `swiglu.v` computes `(silu_lut[g] * up) >> 6`, its true output physical scale is $s_{\text{swiglu}, l} = 64 \cdot s_{\text{gate}} \cdot s_{\text{up}, l}$, which the compiler folds directly into `down_proj`'s dyadic $(M_i, s_i)$ parameters.
4. **Compile-Time SmoothQuant Per-Channel Migration (`Hypothesis H4 / EN4`):** In SmolLM2-135M, 2–3 massive outlier channels in `L02`, `L11`, `L28`, and `L29` reach magnitudes of $2,630$ at `pos = 0` while 99% of channels remain $< 15$. `calibrate_and_migrate_artifacts_v2` (`frontend/importer_v2.py`) folds per-channel SmoothQuant migration factors offline into $\gamma_{\text{attn}} \leftrightarrow W_{q,k,v}$, $\gamma_{\text{mlp}} \leftrightarrow W_{\text{gate},\text{up}}$, $\gamma_{\text{final}} \leftrightarrow W_{\text{lm\_head}}$, RoPE-pair-invariant $W_q \leftrightarrow W_k$, and $(W_{\text{gate}}, W_{\text{up}}) \leftrightarrow W_{\text{down}}$ with **zero runtime hardware overhead**.

---

## 6. Part II: Compiler Extension (`ir/`, `compiler/`, `backend/`)

1. **Block-Structured Graph IR (`ir/graph_v2.py`):** Represents the full decoder graph (`EMBED`, `RMSNORM`, `LINEAR`, `ROPE`, `ATTENTION`, `RESIDUAL_ADD`, `SWIGLU`, `ARGMAX`) and enforces Invariants `I1`–`I7` with lossless JSON round-trip serialization.
2. **Format & Dyadic Scale Pass (`compiler/format_assignment.py`):** Assigns edge formats and computes 64-bit packed requantization words `{16'd0, shift[15:0], mult[31:0]}` for every channel of every linear and RMSNorm operator.
3. **Immutable ROM Packer (`compiler/rom_packing.py` & `backend/top_generator.py`):** Packs all 162,791,424 weight bytes (padded to 16-byte / 128-bit engine rows in `weights_128b.hex`), 288,990 requantization parameters (`requant_params.hex`), lookup tables (`tables_*.hex`), and layer base pointers (`layer_bases.hex`) with SHA-256 integrity hashes in `address_map.json`.
4. **Microprogram Scheduler & Memory Planner (`compiler/schedule.py`, `compiler/memory_planner.py`):** Emits a 19-instruction looped microprogram (`program.hex`, `schedule.json`) and allocates 10 live activation buffers (`167,040` bytes) plus the `[30][2][3][256][64]` INT8 KV cache (`2,949,120` bytes).
5. **Compiled-Artifact Interpreter (`compiler/interpreter.py`):** Reads *only* the emitted hex and JSON files in `build/smollm2_hw/` to serve as the independent hardware-level golden oracle.

---

## 7. Part II: Hardware Accelerator Architecture (`rtl/lib/`)

The generated top-level accelerator `top_accel.v` integrates 12 parameterized Verilog-2001/2005 modules:
- **`rom_sync.v`**: Synchronous read-only memory (`clk`, `addr`, `data_out`) initialized via `$readmemh` with zero write ports.
- **`sequencer.v`**: Fetches 128-bit microinstructions from `prog_rom_inst`, loops `layer_idx` from `0` to `29`, resolves per-layer weight/parameter base addresses from `layer_bases_rom_inst`, and coordinates `start`/`busy`/`done` handshakes across all functional units.
- **`linear_engine.v`**: Streaming $R \times C = 8 \times 16$ (128 MAC lanes) matrix-vector engine reading 128-bit packed weight words per cycle, accumulating $R$ rows in parallel into 32-bit signed accumulators, and pipelining through `requant.v` into either 8-bit activation buffers or 24-bit residual/logit buffers.
- **`embed_gather.v`, `rmsnorm.v`, `rope.v`, `attention.v`, `softmax.v`, `kv_cache.v`, `swiglu.v`, `residual_add.v`, `argmax.v`**: Dedicated fixed-point streaming units implementing the frozen `numerics/` contracts bit-for-bit.

---

## 8. Verification Methodology

1. **Dual Independent Oracles (Golden Rule 2):** `numerics/decoder.py` and `compiler/interpreter.py` agree bit-for-bit, and RTL is verified against `CompiledArtifactInterpreter` across every intermediate tensor trace tap (`18` taps per layer).
2. **First-Divergence Diagnostic Tool (`verification/trace_diff.py`):** Compares golden and RTL trace trees in topological execution order (`seq -> tok -> layer -> op -> elem`) and isolates the exact first differing byte.
3. **Tiny-Model Verification Ladder (Phase `UD`):**
   - Verified **100 random prompts** (including sequences up to `MAX_CTX = 32`) with 100% token and trace exactness (`tests/test_tiny_model_end_to_end.py`).
   - Verified **cross-simulator trace equality** between Icarus Verilog (`iverilog`) and Verilator (`tests/test_cross_simulator.py`).
4. **Mutation Testing Suites (`M10`–`M30` & `TM1`–`TM7`):**
   - **21 Unit Mutants (`verification/mutate2.py`):** Faults injected into `requant.v`, `numerics.vh`, `rope.v`, `attention.v`, `kv_addr_gen.v`, `softmax.v`, `rmsnorm.v`, `swiglu.v`, `residual_add.v`, `argmax.v`, `sequencer.v`, `kv_cache.v`, `embed_gather.v`, and `linear_engine.v` — **21/21 killed (100.0%)** (`results/mutation_matrix_v2.json`).
   - **7 System Mutants (`tests/test_top_mutation.py`):** Causal mask off-by-one, GQA head map bug, position counter freeze, RoPE table shift, residual scale corruption, ArgMax tie-break inversion, and ROM word offset — **7/7 killed (100.0%)** (`results/top_mutation_matrix.json`).
5. **Structural, ROM Immutability, Determinism & Multi-Tool Lint Audits (`tests/test_audits_v2.py`):**
   - Verified zero write ports on any ROM, zero unauthorized writable arrays, zero expected/oracle tokens in RTL, byte-identical codegen determinism, and clean lint across `iverilog -Wall`, `verilator --lint-only -Wall`, and `yosys check -assert`.

---

## 9. Experimental Results

### 9.1 Claim C1′: Full 30-Layer SmolLM2-135M Bit-Exact RTL Verification (`UE.1`–`UE.5`)
- **Compilation Scalability (`UE.1`, `results/part2/ue1_generation.json`):** Full SmolLM2-135M hardware bundle generated in **`12.44` seconds** (`2.557` GB peak RSS, `799.1` MB on disk).
- **Verilator Build & Speed (`UE.2`, `results/part2/ue2_verilator_build.json`):** Built with `-O3` in **`4.45` seconds**, simulating at **`2,415,740` cycles/s (`2.42 MHz`)**.
- **Layer-0 Multi-Position Trace Parity (`UE.3`, `results/part2/ue3_layer0_traces.json`):** Across 12 positions (`165,511,344` cycles), all `216` trace files (**`728,064` elements**) matched `0` divergences.
- **Full 30-Layer Multi-Prompt Decode (`UE.4`, `results/part2/full_decode.json`):** Across **10 prompts (`p01`–`p10`) $\times 16$ new tokens** (`285` total decode steps, `4,018,574,895` cycles), **10/10 prompts achieved 100% token exactness** and **100% sampled-layer (`L0`, `L14`, `L29`) trace equality** across `1,344` files (**`2,279,424` elements**, `first_divergence: null`). Mean execution latency is **`14,100,262` cycles/token**.
- **Long-Context Verification at `MAX_CTX = 256` (`UE.5`, `results/part2/ue5_long_context.json`):** Simulated a full 256-token sequence (`240` prompt + `16` generated tokens, `4,669,315,072` cycles) with **100% token and trace exactness** (`90` files, `303,360` elements checked).

### 9.2 Claim C2′: Quantization Quality & Host RTL Text Generation (`UE.6`, `UE.7`)
Evaluated across 10 pinned WikiText-2 (`wikitext-2-raw-v1`, test split) windows of length 64 (`630` predicted tokens, `results/part2/quality_final.json`):

| Metric | PyTorch FP32 Baseline | LLMHW Pure-INT8 Hardware (`top_accel.v`) | Delta / Agreement | Gate HA Frozen Threshold | Status |
|---|---|---|---|---|---|
| **Cross-Entropy Loss** | `4.4962` | `4.6729` | `+0.1767` nats | — | — |
| **Perplexity (PPL)** | `89.6749` | `107.0031` | **`+17.3282`** | $\le 25.0$ | **PASS** |
| **Top-1 ArgMax Agreement** | `100.0%` | **`70.16%`** | `442 / 630` exact top-1 | $\ge 35.0\%$ | **PASS** |
| **Mean $\text{KL}(P_{\text{fp32}} \parallel P_{\text{int}})$** | `0.0000` | **`0.2280`** | `0.2280` nats | $\le 0.25$ | **PASS** |

Sample 16-token completions generated by the 30-layer Verilog RTL simulation (`results/part2/ue6_host_demo.json`, `results/part2/full_decode.json`):
- **`p01`** (*"The capital of France is Paris. The capital of Germany is"*):
  - **FP32 Greedy:** `" Berlin. The capital of Italy is Rome. The capital of Spain is Madrid."`
  - **RTL INT8:** `" Berlin. The capital of Italy is Rome. The capital of Spain is Madrid."` (**16/16 tokens identical**)
- **`p04`** (*"Isaac Newton published the Principia Mathematica in"*):
  - **FP32 Greedy:** `" Natural Philosophy, which was the first book on physics and astronomy"`
  - **RTL INT8:** `" Natural Philosophy, which was the first book to be published in the English language."` (**9/16 prefix match**)
- **`p05`** (*"She went to the library to read a book about"*):
  - **FP32 Greedy:** `" the history of the city. She found a book about the history"`
  - **RTL INT8:** `" the history of the city. She found the book and read it. She was"` (**8/16 prefix match**)
- **`p07`** (*"The Earth orbits around a star that is"*):
  - **RTL INT8:** `" called the Sun. The planet farthest from the Sun is called the Moon.\n"`
- **`p10`** (*"According to Einstein's theory of relativity, E ="*):
  - **RTL INT8:** `" mc2, where E is the energy, and m is the mass.\n"`

### 9.3 Architecture & Synthesis Study (`EA1`–`EA5`)

#### EA1: Unit-Level Logic Synthesis (`results/part2/EA1.csv`)
| RTL Unit | Generic Total Cells | Generic Comb Cells | Generic DFFs | Generic LTP Depth | iCE40 Total Cells | iCE40 LUT-4 | iCE40 DFFs |
|---|---|---|---|---|---|---|---|
| `linear_engine` ($8 \times 16$) | `21,387` | `20,884` | `503` | `96` | `8,376` | `7,260` | `503` |
| `rmsnorm` | `51,656` | `51,294` | `362` | `2,077` | `70,127` | `22,742` | `362` |
| `attention` (incl. `softmax`) | `3,389` | `1,150` | `2,239` | `78` | `1,045` | `898` | `2,238` |
| `softmax` | `2,690` | `2,535` | `155` | `662` | `4,225` | `1,528` | `155` |
| `rope` | `892` | `732` | `160` | `46` | `495` | `428` | `160` |
| `argmax` | `370` | `248` | `122` | `37` | `265` | `230` | `122` |
| `swiglu` | `253` | `171` | `82` | `27` | `153` | `146` | `82` |
| `sequencer` | `191` | `78` | `113` | `6` | `151` | `121` | `113` |
| `requant` | `159` | `159` | `0` | `28` | `84` | `57` | `0` |
| `residual_add` | `124` | `72` | `52` | `28` | `46` | `45` | `52` |
| `embed_gather` | `76` | `48` | `28` | `20` | `44` | `38` | `28` |
| `kv_addr_gen` | `13` | `13` | `0` | `2` | `1` | `1` | `0` |
| **Total Compute Datapath** | **`78,510`** | **`74,849`** | **`3,661`** | — | **`80,762`** | **`31,908`** | **`3,660`** |

#### EA2: Systolic Engine Grid $(R, C)$ Exploration & Cycle Validation (`results/part2/EA2.csv`, `EA2.json`)
Our closed-form analytical cycle model predicts **`13,691,648` cycles/token** for the baseline $8 \times 16$ engine versus **`14,100,262` measured cycles/token** in Verilator (`UE.4`), a relative error of just **`-2.90%`**.

| Grid $(R \times C)$ | MAC Lanes | Generic Cells | DFFs | iCE40 LUT-4 | LTP Depth | Analytical Cycles/Token | Speedup vs $4 \times 4$ |
|---|---|---|---|---|---|---|---|
| `4x4` | `16` | `2,477` | `235` | `853` | `78` | `89,383,488` | `1.00×` |
| `4x8` | `32` | `5,152` | `235` | `1,797` | `87` | `46,129,728` | `1.94×` |
| `8x8` | `64` | `10,331` | `447` | `3,573` | `87` | `24,502,848` | `3.65×` |
| **`8x16` (Baseline)** | **`128`** | **`21,387`** | **`503`** | **`7,260`** | **`96`** | **`13,691,648`** (`14.10M` sim) | **`6.53×`** |
| `16x16` | `256` | `39,553` | `775` | `13,820` | `96` | `8,284,928` | `10.79×` |
| `16x32` | `512` | `81,969` | `775` | `27,900` | `105` | `6,133,824` | `14.57×` |

#### EA3: On-Chip Memory Accounting at `MAX_CTX = 256` (`results/part2/EA3.json`)
- **Immutable ROM Total:** **`157.5435 MB`** (`165,196,496` bytes, **`98.15%`** of on-chip bits) — comprising `155.25 MB` weights (`162,791,424` B), `2.20 MB` requantization parameters (`2,311,920` B), `88.0 KB` tables, and `4.14 KB` microprogram/layer bases.
- **Dynamic Writable SRAM Total:** **`2.9718 MB`** (`3,116,160` bytes, **`1.85%`** of on-chip bits) — comprising `2.8125 MB` KV cache (`2,949,120` B) and `0.1593 MB` activation buffers (`167,040` B).

#### EA4: Spatial (Part I) vs. ROM-Sequential (Part II) on $32 \times 32$ Block (`results/part2/EA4.csv`, `EA4.json`)
Synthesizing an identical $32 \times 32$ ($1,024$-weight) block in Yosys:
- **Part I Flat Spatial (`Design F`):** `213,022` generic cells (`72,958` iCE40 LUT-4s), `1` cycle latency, `60.94` s synthesis time (`312.33` s for iCE40). Extrapolating $208.0$ cells/weight to $134.5\text{M}$ weights yields **`27.98 Billion` logic gates** (`[estimate]`).
- **Part II ROM-Sequential ($8 \times 16$ Engine):** `21,387` generic cells (`7,260` iCE40 LUT-4s), `138` cycles latency (**`9.96×` fewer gates** on a single $32 \times 32$ block, and **reused across all 210 projections + LM head**).

#### EA5: TSMC 28nm HPC+ Pre-Layout ASIC Projection (`results/part2/EA5.json`, All Numbers `[estimate]`)
| Engine Grid (MAC Lanes) | Low Die Area ($\text{mm}^2$) | **Mid Die Area ($\text{mm}^2$)** | High Die Area ($\text{mm}^2$) | Low Tok/s (250 MHz) | **Mid Tok/s (400 MHz)** | High Tok/s (550 MHz) | Mid Latency (ms/tok) |
|---|---|---|---|---|---|---|---|
| `4x4` (`16` lanes) | `115.61` `[estimate]` | **`147.90`** `[estimate]` | `191.43` `[estimate]` | `2.72` `[estimate]` | **`4.35`** `[estimate]` | `5.98` `[estimate]` | `230.10` `[estimate]` |
| `8x8` (`64` lanes) | `115.63` `[estimate]` | **`147.92`** `[estimate]` | `191.46` `[estimate]` | `9.92` `[estimate]` | **`15.86`** `[estimate]` | `21.81` `[estimate]` | `63.03` `[estimate]` |
| **`8x16` (`128` lanes)** | **`115.65`** `[estimate]` | **`147.95`** `[estimate]` | **`191.50`** `[estimate]` | **`17.73`** `[estimate]` | **`28.37`** `[estimate]` | **`39.01`** `[estimate]` | **`35.25`** `[estimate]` |
| `16x16` (`256` lanes) | `115.69` `[estimate]` | **`148.00`** `[estimate]` | `191.56` `[estimate]` | `28.59` `[estimate]` | **`45.75`** `[estimate]` | `62.90` `[estimate]` | `21.86` `[estimate]` |
| `16x32` (`512` lanes) | `115.77` `[estimate]` | **`148.10`** `[estimate]` | `191.70` `[estimate]` | `39.61` `[estimate]` | **`63.37`** `[estimate]` | `87.13` `[estimate]` | `15.78` `[estimate]` |

Sensitivity elasticity analysis (`EA5.json`) proves that total die area is **96.55% elastic to ROM bit density** ($\epsilon_{\text{ROM}} = 0.9655$), `8.46%` to routing overhead, `2.49%` to SRAM density, and only **`0.10%` to logic gate density** ($\epsilon_{\text{logic}} = 0.0010$). Consequently, scaling compute from 128 to 512 MAC lanes boosts throughput by **$2.23\times$ (`28.37` $\to$ `63.37 tok/s` `[estimate]`)** while increasing die area by just **`0.15 mm²` (`+0.10%`)**.

---

## 10. Discussion, Context Comparison, and Limitations

As detailed in [`docs/part2/context_comparison.md`](part2/context_comparison.md) and [`docs/limitations.md`](limitations.md):
- Compared against commercial spatial/SRAM-centric accelerators (Groq LPU, Etched Sohu, SambaNova RDU, Tenstorrent Grayskull, Apple Neural Engine), LLMHW occupies a unique embedded design point: **single-chip weight-in-ROM specialization** for sub-200M models without external HBM/LPDDR DRAM.
- Key limitations are explicitly documented: (1) greedy `batch_size = 1` decoding, (2) host-CPU BPE tokenization, (3) `MAX_CTX = 256` on-chip KV SRAM sizing ($2.81\text{ MB}$), (4) sequential prompt prefill, (5) multi-step greedy trajectory divergence after a mean prefix match of `4.5` tokens despite `70.16%` single-step Top-1 agreement, and (6) pre-layout synthesis and analytical 28nm area/frequency assumptions.

---

## 11. Future Work (Part III Roadmap)

1. **Physical Design & Foundry Macro Integration (Part III):** Push `top_accel.v` through OpenLane / Cadence Innovus with commercial 28nm multi-banked ROM and 6T SRAM compiler macros for post-route timing, DRC/LVS, and power signoff.
2. **Hybrid Spatial-ROM Co-Processors:** Combine Part I unrolled spatial constant trees for hot bottleneck projections or parallel prompt-prefill engines with Part II ROM-backed sequential decoding.
3. **Sub-Byte Weight Compression (W4 / W6):** Reducing weight ROM width from 8 bits to 4 bits would directly halve the $142.84\,\text{mm}^2$ ROM macro area to $\sim 75\,\text{mm}^2$ `[estimate]`.

---

## 12. Conclusion

Across Part I and Part II, **LLMHW** demonstrates that a complete, production 30-layer, 134.5M-parameter transformer decoder (`SmolLM2-135M`) can be automatically compiled into a pure-integer, immutable-ROM Verilog hardware accelerator (`78,510` logic cells + `157.54 MB` ROM + `2.97 MB` SRAM) that generates **bit-for-bit identical tokens (`10/10` prompts, `2.28M` trace elements, `0` divergences)** to an independent software integer reference while retaining **70.16% Top-1 token agreement** and **$\Delta \text{PPL} = +17.33$** against FP32 PyTorch.

---

## Appendices

### Appendix A: Reproducibility from a Fresh Clone
```bash
make env-check      # Verify Python, PyTorch, Icarus Verilog, Verilator, Yosys
make check          # Run full Part I + Part II test suite
make check2         # Run Part II lint + 168 unit, RTL, mutation, audit & CLI tests
make demo2-fast     # Tiny model end-to-end RTL sim + audits + mutation kill matrix (~11.5s)
make demo2          # Full 30-layer SmolLM2-135M Verilator RTL generation demo (~106s)
```
Clean-directory reproduction verified in `results/part2/ug3_clean_repro.json` (`status: PASS`, `54.86s`).

### Appendix B: Repository Map
- `numerics/` — Frozen pure-integer operator library (`ops.py`), decoder oracle (`decoder.py`), and `config_frozen.json`.
- `frontend/` — Model loader, linear extractor (Part I), full-decoder artifact dumper & SmoothQuant calibrator (`importer_v2.py`), and `tiny_model.py`.
- `ir/` — Part I linear IR and Part II block-structured `graph_v2.py` (Invariants `I1`–`I7`).
- `compiler/` — Format & dyadic scale assignment, ROM packing, microprogram scheduler, memory planner, and `CompiledArtifactInterpreter`.
- `rtl/lib/` — 12 hand-written parameterized Verilog-2001/2005 hardware modules (`top_accel.v`, `linear_engine.v`, `rmsnorm.v`, `rope.v`, `attention.v`, `softmax.v`, `kv_cache.v`, `swiglu.v`, `residual_add.v`, `embed_gather.v`, `argmax.v`, `sequencer.v`, `rom_sync.v`).
- `backend/` & `sim/` — Top-level bundle generator (`top_generator.py`) and Verilator/Icarus testbench generator (`tb_generator.py`).
- `verification/` — Independent INT64 reference, trace diff engine (`trace_diff.py`), weight/ROM audit v2, structural audit, and mutation engines (`mutate.py`, `mutate2.py`).
- `eval/` & `synth/` — WikiText-2 evaluation harness (`harness.py`) and Yosys synthesis runner (`yosys_runner.py`).

### Appendix C: Architecture Decision Records (ADR Index)
- **Part I:** `ADR-0001` (Linear quantization thresholds), `ADR-0002` (Symmetric INT8 $[-127, 127]$), `ADR-0003` (Pipeline latency $L=1$).
- **Part II:**
  - [`ADR-2-0`](part2/adr/ADR-2-0_retrospective_and_process_trim.md): Part I retrospective and workflow streamlining.
  - [`ADR-2-1`](part2/adr/ADR-2-1_frozen_numerics_configuration.md): Frozen W8A8/INT24 dyadic numerics contract and LUT sizes.
  - [`ADR-2-2`](part2/adr/ADR-2-2_rmsnorm_exponent_shift_and_activation_calibration.md): RMSNorm/Softmax LUT alignment, exact SwiGLU scale propagation, and offline SmoothQuant per-channel scale migration (`BLOCKER-UE7-01` resolution).

### Appendix D: Academic Integrity & AI-Assistance Disclosure
In compliance with the Indian Institute of Technology Kharagpur academic integrity guidelines:
1. **Nature of AI Assistance:** This B.Tech Project utilized an agentic AI coding assistant (Google DeepMind Antigravity) in a pair-programming workflow for Python/Verilog code generation, automated testbench scaffolding, Yosys/Verilator script automation, root-cause debugging of quantization scaling (`ADR-2-2`), and Markdown/LaTeX documentation formatting.
2. **Human Verification & Ownership:** Every architectural specification (`docs/part2/02_architecture_and_contracts.md`), Architecture Decision Record (`ADR-2-0`, `ADR-2-1`, `ADR-2-2`), mathematical derivation, hardware bitwidth contract, and experimental number in `results/` was executed, verified, and reviewed by the student author (Himesh Kundal, 23EC10030) under the supervision of Prof. Mrityunjoy Chakraborty. Zero numbers in this report were fabricated or unverified; every metric traces to an executable script in [`docs/report_traceability.md`](report_traceability.md).
