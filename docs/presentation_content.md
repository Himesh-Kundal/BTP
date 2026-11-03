# BTP Defense Presentation: Slide Deck & Speaker Script
**Project:** Model-Specific Hardware Compilation of Dense Transformer Projections into Synthesizable Constant-Weight Verilog  
**Candidate:** Himesh Kundal (Roll No: 23EC10030)  
**Supervisor:** Prof. Mrityunjoy Chakraborty  
**Department:** Electronics and Electrical Communication Engineering, IIT Kharagpur  
**Date:** November 2026  

---

## 🎨 Recommended Presentation Theme & Visual Guidelines

### Theme: "Deep Silicon & Precision Cyan" (Modern Academic High-Tech)
* **Background:** Dark Slate / Deep Obsidian (`#0F172A` / `#0B0F19`) or Clean Pristine Academic Light (`#FFFFFF` / `#F8FAFC`).
* **Primary Accent Color:** Electric Cyan / Bright Teal (`#00E5FF` / `#0284C7`) — represents digital silicon, clocking, and logic wires.
* **Secondary Accent Color:** Warm Amber / Solar Gold (`#F59E0B` / `#10B981`) — represents breakthrough metrics (**3.85× Area**, **17.7× Flops**, **100% Mutant Kill**).
* **Code / Math Font:** JetBrains Mono or Fira Code (`14pt–16pt`).
* **Slide Typography:** Clean modern sans-serif (Inter, Roboto, or Arial) with high data-to-ink contrast.
* **Layout Rule:** **Split-Screen or 3-Card Layout**. Left: Concept & Architecture diagram; Right: Hard experimental data & takeaway badge.

---

## 🧭 Presentation Flowchart (At a Glance)

```text
[1. Title & Thesis] ──> [2. The Memory Wall & Generality Tax] ──> [3. Core Research Question]
         │
         ▼
[4. LLMHW Architecture & SmolLM2] ──> [5. Numeric Contract & Bit-Growth Proof]
         │
         ▼
[6. RTL Design F vs Design P] ──> [7. Multi-Tile Spatial Partitioning]
         │
         ▼
[8. Experimental Results (Area/Flops)] ──> [9. Timing & Critical Path (LTP)]
         │
         ▼
[10. Weight Distribution Effect] ──> [11. Verification & 100% Mutation Kill]
         │
         ▼
[12. Live Toolchain Demo (2.97s)] ──> [13. Limitations & Part II/III Roadmap] ──> [14. Conclusion]
         │
         ▼
[Backup Slides: Baugh-Wooley vs CSD Math, Waveforms, Fairness Audit, Traceability]
```

---

## Slide 1: Title Slide — The Core Thesis

### Visual Layout:
* Clean centered header with IIT Kharagpur emblem.
* Bold title with two-tone styling (**Model-Specific Hardware Compilation** in Cyan).
* Key Takeaway Badge at the bottom: *"3.85× Standard-Cell Area Reduction | 17.7× Flip-Flop Elimination | Bit-Exact Equivalence"*.

### On-Slide Content:
* **Title:** Model-Specific Hardware Compilation of Dense Transformer Projections into Synthesizable Constant-Weight Verilog
* **Candidate:** Himesh Kundal | Roll No: **23EC10030** | B.Tech (Honours), E&ECE
* **Supervisor:** Prof. Mrityunjoy Chakraborty
* **Department:** Department of Electronics & Electrical Communication Engineering, IIT Kharagpur
* **Core Contribution:**
  * Transforming frozen pre-trained Transformer weights directly into immutable, synthesizable Verilog-2001 standard-cell logic.
  * Complete elimination of weight SRAMs, register files, and address buses for dedicated edge inference appliances.

### 🎙️ Speaker Script (1 minute):
> *"Good morning, respected committee members and Professor Chakraborty. Today, I am presenting my B.Tech Project Part I on 'Model-Specific Hardware Compilation of Dense Transformer Projections into Synthesizable Constant-Weight Verilog'.*  
> *In this project, we ask a fundamental question in domain-specific computer architecture: When deploying a language model to a dedicated edge system where weights will never change post-training, why should we pay the massive silicon area, memory bandwidth, and power taxes of general-purpose programmable accelerators?*  
> *I have built an automated end-to-end compiler called LLMHW that ingests trained Transformer models, extracts dense linear projections, and generates bit-exact, synthesizable Verilog where weights are hardcoded as constant logic—achieving a 3.85× gate reduction and up to a 17.7× flip-flop reduction with zero weight memory overhead."*

---

## Slide 2: Problem Statement — The Memory Wall & The "Generality Tax"

### Visual Layout:
* **Two-Column Comparison**:
  * Left: Conventional NPU / Systolic Array (heavy memory ports, weight flip-flops, generic $8\times 8$ Baugh-Wooley multipliers).
  * Right: Dedicated Edge Appliance Reality (fixed task, frozen model, zero runtime weight updates).
* Highlight Box: *"The Generality Tax: Up to 75% of accelerator area is spent merely storing and routing weights that never change."*

### On-Slide Content:
* **The Edge Inference Bottleneck:**
  * Large Language Models (LLMs) are memory-bandwidth and area bound.
  * General-purpose NPUs (TPUs, systolic arrays, GPUs) must support *any* network; therefore, they require:
    1. Multi-banked SRAM arrays or wide weight register files.
    2. Programmable address decoders, multiplexers, and write-enable routing.
    3. Generic $8\text{-bit} \times 8\text{-bit}$ hardware multipliers ($64$ AND gates + $56$ full adders per MAC).
* **The Edge Specialization Opportunity:**
  * In mission-critical edge applications (automotive sensors, UAV flight control, robotics, biometric edge devices), models are **frozen post-deployment**.
  * **The Core Problem:** General-purpose hardware spends the vast majority of its silicon area and static leakage power managing weight mutability that edge appliances never use!

### 🎙️ Speaker Script (1.5 minutes):
> *"To understand the motivation, let's look at the memory wall in edge AI. Standard neural processing units are designed for generality. Because a TPU or systolic array doesn't know what model it will execute tomorrow, it must provide writable memory arrays, wide weight buses, and heavy Baugh-Wooley multipliers capable of multiplying any arbitrary 8-bit number by any other 8-bit number.*  
> *However, in real-world edge appliances—such as flight controllers, automotive radar processors, or medical wearables—the model is trained once and frozen. Weight updates occur only during factory firmware re-flashing.*  
> *By forcing frozen weights through general-purpose programmable datapaths, edge systems pay a severe 'Generality Tax'. Our goal is to eliminate this tax entirely by compiling the weights directly into silicon standard cells."*

---

## Slide 3: The Proposed Solution — The LLMHW Compiler Architecture

### Visual Layout:
* Horizontal 5-stage block diagram showing the clean separation of concerns.
* Golden Rule 5 Callout: *"Strict Layering: Zero PyTorch dependencies outside the frontend."*

```text
[HuggingFace SmolLM2] ──> [Frontend: Extractor & Quantizer] ──> [Model-Agnostic IR (TileManifest)]
                                                                          │
[Yosys Synthesis (Gate Area & LTP)] <── [Backend RTL & Testbench] <───────┘
```

### On-Slide Content:
* **LLMHW Architecture:**
  1. **Frontend (`frontend/`):** Loads `HuggingFaceTB/SmolLM2-135M` (134.5M parameters, 30 layers), extracts all 7 projection matrices, and captures calibration activations.
  2. **Quantizer:** Symmetric INT8 quantization with zero-bias round-half-to-even tie breaking.
  3. **Intermediate Representation (`ir/`):** Model-agnostic `TileManifest` containing 2D INT8 weights, channel scales, and synthetic bias.
  4. **Hardware Backend (`backend/`):** Emits synthesizable Verilog-2001 (IEEE 1364-2001) for both specialized constant-weight hardware (**Design F**) and fair programmable baselines (**Design P**).
  5. **Independent Verifier (`verification/`):** Dual-tier verification using a 64-bit NumPy oracle, cycle-accurate Icarus Verilog simulation, and static AST/regex auditing.

### 🎙️ Speaker Script (1.5 minutes):
> *"Here is our end-to-end system architecture. The pipeline begins with HuggingFace pre-trained models. The frontend extracts all linear projections and quantizes them.*  
> *Crucially, we adhere to a strict software engineering principle: Golden Rule 5. The PyTorch framework is completely isolated in the frontend. The entire compiler core, IR, Verilog codegen, and synthesis runners operate on a pure model-agnostic Intermediate Representation called TileManifest.*  
> *This means our compiler is not tied only to SmolLM2. It can compile LLaMA-3, Mistral, or digital FIR filtering matrices without changing a single line of backend hardware code."*

---

## Slide 4: Target Workload & Linear Projection Breakdown

### Visual Layout:
* Table displaying the 7 projections of SmolLM2-135M.
* Diagram showing where each projection lives inside a Transformer Layer (Self-Attention vs. SwiGLU MLP).

### On-Slide Content:
* **Target Model:** **SmolLM2-135M** (30 transformer blocks, hidden dimension $D = 576$, intermediate MLP dimension $D_{\text{ff}} = 1536$).
* **The 7 Profiled Projection Types (210 dense layers across the model):**

| Sub-Module | Projection Name | Dimensions ($N \times K$) | Weight Matrix Elements | Physical Function |
| :--- | :--- | :---: | :---: | :--- |
| **Self-Attention** | `q_proj` | $576 \times 576$ | 331,776 | Query projection |
| **Self-Attention** | `k_proj` | $192 \times 576$ | 110,592 | Key projection (GQA: 3 groups) |
| **Self-Attention** | `v_proj` | $192 \times 576$ | 110,592 | Value projection (GQA: 3 groups) |
| **Self-Attention** | `o_proj` | $576 \times 576$ | 331,776 | Attention output projection |
| **SwiGLU MLP** | `gate_proj` | $1536 \times 576$ | 884,736 | Gating linear layer |
| **SwiGLU MLP** | `up_proj` | $1536 \times 576$ | 884,736 | Up-scaling projection |
| **SwiGLU MLP** | `down_proj` | $576 \times 1536$ | 884,736 | Down-scaling reduction |
| **Total per Block** | **7 Projections** | — | **3,538,944** | **30 blocks = 106.17M weights** |

### 🎙️ Speaker Script (1.5 minutes):
> *"To ensure our results reflect real-world AI workloads, we evaluated every single linear projection across SmolLM2-135M. A single transformer layer contains 7 dense linear projections totaling over 3.5 million weights.*  
> *Notice the asymmetry: Grouped-Query Attention has 192 key and value channels for 576 query channels, while the SwiGLU feed-forward network expands to 1,536 hidden channels.*  
> *Across the entire 30-layer model, these 7 projections account for 106.2 million weights—over 78% of all model parameters. Compiling these 7 matrices efficiently is the primary driver of edge hardware acceleration."*

---

## Slide 5: Mathematical Rigor — Quantization & Bit-Growth Proofs

### Visual Layout:
* Equations rendered in high-contrast LaTeX.
* Visual callout of the Accumulator Dynamic Range showing the 6–7 bits of guaranteed headroom.

### On-Slide Content:
* **Quantization Contract (Rules N1–N3):**
  * Symmetric INT8 clipping to **$[-127, +127]$**, strictly excluding $-128$.
  * *Why eliminate $-128$?* In two's complement, $-(-128) = -128 \neq +128$ (unary negation overflow). Eliminating $-128$ guarantees valid negation for negative weights!
  * **Round-Half-to-Even (Banker's Rounding):** Guarantees zero-mean statistical bias ($\mathbb{E}[\epsilon] = 0$) across 576 accumulation taps.
  * Achieves **Cosine Similarity $> 0.9994$** and **Relative Error $< 0.032$** across all 7 layers.
* **Accumulator Bit-Growth Proof (Rule N4):**
  * Maximum magnitude of a single INT8 product: $127 \times 127 = 16{,}129$.
  * For accumulation depth $K$, minimum required signed bits:
    $$B_{\text{req}}(K) = 1 + \lceil \log_2(K \times 16{,}129) \rceil$$
  * For $K = 576$ (`q_proj`, `gate_proj`): $\max |y| = 9{,}290{,}304 \implies \mathbf{25\text{ bits}}$.
  * For $K = 1536$ (`down_proj`): $\max |y| = 24{,}774{,}144 \implies \mathbf{26\text{ bits}}$.
  * Standard **32-bit signed accumulator (`[31:0]`)** guarantees **6 to 7 guard bits** against overflow under all adversarial inputs!

### 🎙️ Speaker Script (2 minutes):
> *"Now let's examine the mathematical foundation. In hardware DSP, overflow is fatal. We strictly restrict our INT8 weights and activations to the symmetric range -127 to +127, deliberately forbidding -128.*  
> *Why? In two's complement arithmetic, 8-bit negation of -128 wraps around to -128 because +128 cannot be represented. By bounding the domain to [-127, 127], negation is always closed and exact.*  
> *Next, we analytically derived the bit-growth formula. Summing 576 products of magnitude 16,129 yields a maximum value of 9.29 million, which strictly requires 25 bits. For down_proj with K=1536, it requires 26 bits.*  
> *By employing standard 32-bit accumulators, we mathematically guarantee that internal overflow is impossible—maintaining 6 full guard bits of headroom."*

---

## Slide 6: RTL Architecture — Design F (Fixed) vs. Design P (Programmable)

### Visual Layout:
* **Side-by-Side RTL Code Comparison**:
  * Left (Design P): Flip-flop array `w_mem[w_addr] <= w_din`, runtime variable multipliers `x * w_mem`.
  * Right (Design F): Direct constant literals `acc = x * 8'sd12 + (-x_2) + x_3;` with zero-skipping.
* Bottom Banner: *"Rule N10 Fairness: Identical bitwidths (8-bit in, 32-bit out), identical clocking, identical 1-cycle latency ($L=1$)."*

### On-Slide Content:
* **Design F (Model-Specific Constant Compilation):**
  * **Immutable Constant Literals:** Emits Verilog `8'sd<W>`.
  * **Zero-Skipping Optimization:** If $W_{i,j} = 0$, the product term is **completely omitted** from the adder tree at compile time.
  * **Trivial Multiplier Pruning:** If $W_{i,j} = +1$, passes `$signed(x)` (direct wire); if $W_{i,j} = -1$, passes `-$signed(x)` (unary subtractor).
  * **Invariant I3:** Zero weight ports, zero weight flip-flops, zero address lines.
* **Design P (Rigorous Baseline):**
  * Fully programmable memory-mapped weight array (`w_we`, `w_addr`, `w_din`).
  * Emits runtime signed multipliers (`x_col * w_mem`).
* **Design P\_bus (External Bus Baseline):**
  * Direct parallel weight bus without flip-flops (proves savings originate from constant logic, not just memory elimination).

### 🎙️ Speaker Script (2 minutes):
> *"Here is how this translates to synthesizable Verilog-2001. In Design P, we maintain an internal register file and feed runtime variables into hardware multipliers.*  
> *In Design F, our compiler directly emits constant signed literals. It performs four immediate source-level optimizations:*  
> *First, zero weights are completely skipped—eliminating the adder node entirely.*  
> *Second, weights of +1 become simple wires, and -1 become basic two's complement inversions.*  
> *To guarantee scientific fairness under Rule N10, both designs share identical port widths (128-bit input, 256-bit output for 8x16 tiles), identical clocking, and identical single-cycle registered output latency."*

---

## Slide 7: VLSI Multiplier Architecture — Baugh-Wooley vs. CSD Trees

### Visual Layout:
* **Gate Schematic Comparison**:
  * Left: Full $8\times 8$ Baugh-Wooley array multiplier (64 AND gates, 48 Full Adders, 8 Half Adders $\approx 380$ gates).
  * Right: Constant Shift-and-Add with Canonical Signed Digit (CSD) recoding ($31x = (x \ll 5) - x \implies \mathbf{1\text{ subtractor}}$).

### On-Slide Content:
* **The Silicon Reality Inside Logic Synthesis:**
  * **Generic $8 \times 8$ Multiplier (Design P):**
    * Must handle any runtime input: requires generating all 64 partial-product bits via AND gates.
    * Summing partial products requires a Carry-Save Adder (CSA) matrix + 16-bit Carry-Propagate Adder $\implies \mathbf{\approx 350\text{--}450\text{ logic gates per MAC}}$.
    * High critical path: 25–35 gate levels of logic delay through the multiplier alone.
  * **Constant Multiplier via CSD Recoding (Design F):**
    * Zero partial-product AND gates (known 0/1 bits eliminate AND logic).
    * Hardwired shifts (`x << k`) cost **zero gates** (pure metal wire routing).
    * **Canonical Signed Digit (CSD):** For example, $W = 63 = (00111111)_2$ is rewritten as $64 - 1 = (x \ll 6) - x$ (only 1 subtraction instead of 5 additions!).
    * For real Gaussian LLM weights, the average cost drops to **$< 1.2$ adders per tap**!

### 🎙️ Speaker Script (2 minutes):
> *"From a VLSI Signal Processing standpoint—which Professor Chakraborty has explored extensively in DSP architectures—why is constant multiplication so drastically smaller?*  
> *When multiplying two runtime variables in Design P, the synthesizer must build a full Baugh-Wooley array multiplier: 64 partial-product AND gates, 48 full adders, and a carry-propagate adder—costing nearly 400 gates per multiplier.*  
> *In contrast, when the weight is a constant, two things happen:*  
> *First, bit-shifts in hardware are free—they are simply wire routings requiring zero transistors.*  
> *Second, using Canonical Signed Digit recoding, a weight like 31—which has five ones in binary—is transformed into 32 minus 1. That replaces a 400-gate multiplier with a single 16-bit subtractor! This is the physical secret behind our area and latency reduction."*

---

## Slide 8: Multi-Tile Spatial Partitioning & Inter-Tile Adder Trees

### Visual Layout:
* Diagram showing an $N \times K$ matrix partitioned into an $R_{\text{tiles}} \times C_{\text{tiles}}$ grid.
* Callout of the Top-Level Module summing column tiles together in 1 cycle.

### On-Slide Content:
* **Modular Tile Grid (`compiler/tiling.py`):**
  * Large matrices are partitioned into parametric $T_R \times T_C$ tiles (default: $8 \times 16$).
  * Any matrix dimension $(N, K)$ is automatically zero-padded to grid multiples:
    $$R_{\text{tiles}} = \lceil N / T_R \rceil, \quad C_{\text{tiles}} = \lceil K / T_C \rceil$$
* **Combinational Tiles + Registered Top Wrapper ($L=1$ Parity):**
  * Sub-tiles compute combinational dot products over $T_C$ inputs.
  * The top-level wrapper instantiates an **Inter-Tile 32-Bit Adder Reduction Tree** that sums partial products across all $C_{\text{tiles}}$ in parallel and adds the 32-bit bias.
  * A single synchronous output stage latches `y_out` on `posedge clk`.
  * **Preserves strict $L = 1$ cycle throughput and latency** regardless of tile count!

### 🎙️ Speaker Script (1.5 minutes):
> *"Real LLM layers are large—576 by 576. To make our hardware modular, LLMHW partitions matrices into 2D tile grids, such as 8 rows by 16 columns.*  
> *A critical design choice was preserving latency parity. If each tile had its own output flip-flop and the top-level adder tree also had a flip-flop, latency would increase to 2 cycles.*  
> *Instead, our sub-tiles compute combinational partial sums. The top-level wrapper uses an inter-tile 32-bit adder tree to reduce across column tiles in the same cycle, followed by a single registered output stage.*  
> *This guarantees single-cycle execution and exact bit-identical results between tiled and monolithic hardware."*

---

## Slide 9: Synthesis Results — Standard-Cell Area & Flop Elimination

### Visual Layout:
* **Two Hero Metric Cards**:
  * Card 1: **3.85× Gate Reduction** (74.0% standard-cell area savings).
  * Card 2: **17.7× Flip-Flop Reduction** (94.4% sequential storage eliminated).
* Line chart / Bar chart showing Design F vs. Design P across tile sizes $(4,16)$ to $(32,64)$.

### On-Slide Content:
* **Experiment E1 (Scaling Across Matrix Tile Dimensions):**
  * Synthesized with **Yosys 0.52** using generic CMOS standard-cell logic gates.

| Tile Dimensions ($N \times K$) | Design P Gate Count | Design P\_bus Gate Count | Design F Gate Count | Area Reduction Ratio | Flop Reduction Ratio |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$4 \times 16$** | 33,656 | 32,888 | **8,970** | **3.75×** | **5.71×** |
| **$8 \times 16$** | 68,074 | 66,538 | **17,998** | **3.78×** | **11.43×** |
| **$8 \times 32$** | 136,874 | 133,802 | **35,972** | **3.81×** | **11.43×** |
| **$16 \times 64$** | 599,085 | 586,797 | **155,648** | **3.85×** | **17.78×** |

* **Dead Sign-Extension Bit Pruning:**
  * For small tiles, worst-case magnitude $< 2^{15}$.
  * Yosys statically proves upper accumulator bits are identical sign-extensions and prunes the upper 16 flip-flops in Design F!

### 🎙️ Speaker Script (2 minutes):
> *"Let us examine the primary synthesis results from Experiment E1. Across all evaluated tile dimensions—from 4x16 up to 16x64—Design F consistently achieves a 3.75x to 3.85x reduction in standard-cell gate count.*  
> *Looking at sequential storage, the savings are even more dramatic: flip-flop counts drop by 5.7x to 17.7x.*  
> *Where does this 17.7x flop reduction come from? First, we completely eliminate the 1,024 weight registers of Design P. Second, because our weights are bounded constants, Yosys statically proves that the upper accumulator bits can never toggle independently of the sign bit, pruning redundant flip-flops.*  
> *Notice also Design P_bus: even when weight registers are removed from the baseline and replaced with primary ports, Design F is still 3.77x smaller—proving the reduction is driven by arithmetic specialization, not just storage removal."*

---

## Slide 10: Synthesis Results — Weight Content & Timing Path Analysis

### Visual Layout:
* Left Chart: Area comparison between Uniform Random INT8 vs. Real Trained SmolLM2 Weights.
* Right Chart: Critical Path Depth (`ltp`) comparison showing 32% faster timing.

### On-Slide Content:
* **Experiment E2 (Impact of Weight Distribution):**
  * Real trained SmolLM2 weights are **23.0% smaller** than uniform random INT8 weights ($22,764$ vs $28,006$ gates) due to zero-clustering around $\mu \approx 0$.
  * Spatial position permutations have $< 0.3\%$ variation, proving savings are driven by weight numerical sparsity and small dynamic range.
* **Experiment E3 (Critical Path Delay & Timing Depth):**
  * Evaluated via Yosys Longest Topological Path (`ltp`) metric.
  * In Design P, critical path is **56 gate levels** ($8\times 8$ Baugh-Wooley array + adder tree).
  * In Design F, critical path drops to **38 gate levels** (ratio **0.68**, a **32% reduction** in logic depth).
* **FPGA Proxy Synthesis (Lattice iCE40 LUT-4):**
  * Design F requires **2.34× fewer 4-input LUTs** than Design P (4,233 vs 9,910 LUT4s).

### 🎙️ Speaker Script (1.5 minutes):
> *"In Experiments E2 and E3, we investigated the effect of real neural network weight distributions and timing paths.*  
> *In Experiment E2, we found that compiling real trained SmolLM2 weights produces circuits that are 23% smaller than uniform random integers. Neural network weights follow a bell-shaped distribution centered at zero, meaning our zero-skipping and small-constant optimizations trigger much more frequently.*  
> *In Experiment E3, we analyzed timing. By eliminating the multi-stage partial product reduction trees of generic multipliers, Design F slashes the longest topological path from 56 gate levels down to 38 gate levels—a 32% reduction in critical path delay.*  
> *Finally, proxy synthesis on Lattice iCE40 FPGAs confirms a 2.34x reduction in 4-input Look-Up Tables."*

---

## Slide 11: Full-Model Extrapolation & Multi-Tile Validation

### Visual Layout:
* Full-layer scaling diagram from $8\times 16$ tile up to $576 \times 576$ layer core.
* Model Validation Card: Direct synthesis of $16 \times 32$ multi-tile core matches analytical model within 10.82%.

### On-Slide Content:
* **Analytical Tile-Scaling Extrapolation Formula (Experiment E5):**
  $$\text{Area}_{\text{est}}(N, K) = \left( R_{\text{tiles}} \cdot C_{\text{tiles}} \cdot \text{Area}_{\text{tile}} \right) + N \cdot (C_{\text{tiles}} - 1) \cdot \text{Area}_{\text{adder32}}$$
* **Empirical Validation of the Model:**
  * Synthesized a complete $16 \times 32$ 4-tile system in Yosys:
  * Analytical Model Prediction: $91{,}056$ gates.
  * Actual Synthesized Netlist: $82{,}160$ gates (**10.82% estimation error**, safe conservative bound).
  * Direct synthesis was actually *smaller* because Yosys flattened and optimized across inter-tile adder boundaries!
* **Full SmolLM2-135M Model Projections (30 Layers, 210 Projections):**
  * Single $576 \times 576$ `q_proj` Layer: Design F = **13.59M gates** vs. Design P = **49.43M gates**.
  * Complete 30-Layer Model Projections: Design F = **6.10 Billion standard gates** (vs. **22.18 Billion** for Design P).

### 🎙️ Speaker Script (1.5 minutes):
> *"To estimate the hardware footprint of full Transformer layers, we developed an analytical scaling model that accounts for both the tile cores and the inter-tile 32-bit adder trees.*  
> *We empirically validated this formula by directly synthesizing a 4-tile 16x32 core in Yosys. The formula was accurate to within 10.8%, serving as a conservative upper bound.*  
> *When scaled to the full SmolLM2-135M architecture, compiling all 210 linear projections into constant standard cells requires approximately 6.1 billion gates, compared to over 22 billion gates for programmable implementations—saving over 16 billion gates of silicon."*

---

## Slide 12: Independent Verification & 100% Mutation Kill

### Visual Layout:
* 3-Tier Verification diagram:
  * Tier 1: NumPy 64-bit Golden Oracle
  * Tier 2: Static Regex/AST RTL Auditor
  * Tier 3: Cycle-Accurate Icarus Verilog / Verilator Simulation
* Mutation Testing Badge: **100.0% Kill Rate (9/9 Mutants Killed)**.

### On-Slide Content:
* **The Verification Problem:** How do we prove a 548,223-line generated Verilog file has zero arithmetic flaws and contains all 331,776 weights correctly?
* **Tier 1: Independent NumPy INT64 Reference Oracle:**
  * Strictly independent reference model; zero shared code with RTL generators.
  * Verified bit-exact equivalence ($\mathbf{Y}_{\text{RTL}} \equiv \mathbf{Y}_{\text{ref}}$) across 1,080 test conditions and all 7 SmolLM2 projections.
* **Tier 2: Static RTL Weight & Interface Auditor (`verification/weight_audit.py`):**
  * Parses Verilog AST and regex patterns to verify only 6 legal ports exist (enforcing Invariant I3).
  * Reconstructs all 331,776 constant weights directly from RTL text in **0.85 seconds** and asserts 100% match with the quantized tensor.
* **Tier 3: Hardware Fault Mutation Testing (`verification/mutate.py`):**
  * Injected 9 systematic hardware bugs: M1 (sign flip), M2 (off-by-one weight), M3 (dropped term), M4 (signedness drop), M5 (slice offset), M6 (truncation), M7 (valid delay), M8 (carry drop), M9 (bias polarity).
  * **Result: 100% Kill Rate (9/9 mutants killed)** — zero surviving bugs!

### 🎙️ Speaker Script (2 minutes):
> *"In automated hardware generation, verification is paramount. A single off-by-one bit slice can corrupt an entire neural network. We implemented a three-tier verification harness.*  
> *First, an independent 64-bit NumPy oracle verifies bit-exact arithmetic across 1,080 test configurations with zero discrepancies.*  
> *Second, we built a static RTL auditor that parses the generated Verilog source code, re-extracts all 331,776 weights directly from the arithmetic expressions, and confirms bit-exact identity with the PyTorch model in just 0.85 seconds.*  
> *Third, to prove our testbench cannot be bypassed, we performed mutation testing. We injected 9 distinct hardware faults into the RTL and testbenches—such as dropping the 'signed' keyword or flipping weight signs. Our test suite achieved a 100% kill rate, detecting and killing every single mutant."*

---

## Slide 13: Live Toolchain Demonstration

### Visual Layout:
* Screenshot or terminal animation of running `make demo` / `llmhw`.
* Execution Timeline Bar showing: Extract (0.9s) $\to$ Compile (0.4s) $\to$ Codegen (0.6s) $\to$ Static Audit (0.2s) $\to$ Simulation (0.8s) = **Total 2.97 seconds**.

### On-Slide Content:
* **Fully Automated CLI Toolchain:**
  ```bash
  llmhw extract   # Extracts all 7 projections from HuggingFace
  llmhw compile   # Quantizes and generates TileManifest IR
  llmhw codegen   # Emits synthesizable Verilog-2001
  llmhw verify    # Runs bit-exact Icarus Verilog co-simulation
  llmhw synth     # Executes Yosys logic synthesis & timing analysis
  ```
* **One-Click Demo (`make demo`):**
  * Compiles and verifies all 7 projections of layer 0 in **2.97 seconds**.
  * Outputs bit-exact assertion: `PASS: checked=10 mismatch_count=0`.
* **Test Suite Quality:**
  * **230 passing automated tests** across 17 test suites (`make check`).
  * 100% reproducible across Linux workstations.

### 🎙️ Speaker Script (1 minute):
> *"Here is a snapshot of our unified toolchain. With a single command, `make demo`, the compiler extracts all 7 projections of layer 0, performs symmetric quantization, generates synthesizable Verilog, verifies interface invariants, runs cycle-accurate simulation in Icarus Verilog, and reports synthesis metrics in under 3 seconds.*  
> *The entire codebase includes 230 automated unit, integration, and property tests, all passing cleanly."*

---

## Slide 14: Scope Boundaries, Caveats & Engineering Trade-offs

### Visual Layout:
* Two-column honest engineering discussion:
  * Left: What We Have Proven.
  * Right: Pre-Layout Limitations & Practical Boundaries.

### On-Slide Content:
* **What We Have Established (Part I Deliverables):**
  * Proven standard-cell gate reduction of **3.85×** and flip-flop reduction of **17.7×**.
  * Proven bit-exact arithmetic equivalence against 64-bit independent integer oracle.
  * 100% verified mutation kill rate and static constant preservation.
* **Honest Engineering Boundaries & Caveats:**
  1. **Pre-Layout Synthesis vs. Post-P&R:** Logic cell counts do not include physical place-and-route interconnect delay or routing wire congestion.
  2. **Weight Immutability:** Updating model weights requires re-synthesizing an FPGA bitstream or re-spinning an ASIC. Best suited for high-volume, fixed-function edge deployments.
  3. **Non-Linear Layers:** Part I focuses exclusively on dense linear projections. Non-linear activations (RMSNorm, Softmax, RoPE, SiLU) are currently computed on the host CPU/MCU.

### 🎙️ Speaker Script (1.5 minutes):
> *"A complete engineering presentation must clearly acknowledge its boundaries.*  
> *First, our area and timing metrics are pre-layout standard-cell synthesis figures. While Yosys provides excellent logic-level metrics, post-place-and-route interconnect parasitics and routing congestion in a specific foundry PDK will impact physical layout.*  
> *Second, constant-weight hardware is immutable: if you change the model weights, you must reload the FPGA bitstream or fabricate a new ASIC. It is tailored for fixed-purpose edge systems.*  
> *Third, Part I focuses strictly on linear projections, which represent 78% of the model parameters. Non-linear operators like RMSNorm and RoPE remain in software for Part I."*

---

## Slide 15: Future Roadmap — Part II & Part III

### Visual Layout:
* Timeline / Roadmap Graphic showing B.Tech Project progression from Part I to Part III:
  * **Part I (Completed):** Linear Projections + Verilog-2001 Generator + Yosys Evaluation.
  * **Part II (Spring 2027):** On-Chip Non-Linear Operators + Pipelined Tree ($L>1$) + Streaming BRAM.
  * **Part III:** Physical Tapeout on SkyWater 130nm / TSMC 28nm via OpenLane.

### On-Slide Content:
* **Part II Objectives (Spring 2027):**
  1. **Non-Linear Hardware Fusion:** Synthesizable integer approximations for RMSNorm, RoPE rotary embeddings, and SwiGLU non-linear activations.
  2. **Multi-Cycle Pipelining ($L > 1$):** Inserting pipeline registers inside wide adder trees to reach clock targets $\ge 500\text{ MHz}$.
  3. **Ping-Pong BRAM Buffering:** Streaming activation vectors through dual-port on-chip Block RAMs for continuous token generation.
* **Part III Objectives (ASIC Tapeout):**
  * Physical design flow using **OpenLane / SkyWater 130nm PDK** and TSMC 28nm standard-cell libraries.
  * Exact post-layout GDSII power, thermal, and silicon area verification.

### 🎙️ Speaker Script (1 minute):
> *"Looking ahead to Part II in the upcoming semester, our roadmap directly addresses these boundaries.*  
> *We will design synthesizable fixed-point hardware blocks for RMSNorm, RoPE, and SwiGLU activation functions, allowing entire transformer blocks to execute completely on silicon.*  
> *We will also introduce multi-stage pipelining inside the column reduction trees to push operating frequencies beyond 500 MHz, along with ping-pong Block RAM streaming for continuous token inference.*  
> *Finally, in Part III, we plan to push the core through the OpenLane ASIC flow using the SkyWater 130nm PDK for physical GDSII tapeout verification."*

---

## Slide 16: Conclusion & Key Takeaways

### Visual Layout:
* 4 High-Impact Takeaway Cards.
* GitHub Repository URL & QR code placeholder.

### On-Slide Content:
* **Summary of Accomplishments:**
  1. **Automated Specialization:** Successfully designed and built LLMHW, an open-source compiler compiling HuggingFace SmolLM2-135M projections into synthesizable Verilog-2001.
  2. **Massive Silicon Savings:** Delivered **3.85× standard-cell gate reduction**, **17.7× flip-flop elimination**, and **32% faster critical path delay** over programmable baselines.
  3. **Zero Arithmetic Discrepancy:** Proved bit-exact equivalence across all 7 projection types with an independent 64-bit NumPy oracle and 100% mutation test kill rate.
  4. **Open-Source Reproducibility:** Entire code, tests, documentation, and BTP report published at `https://github.com/Himesh-Kundal/BTP`.
* **Closing Statement:** Constant-weight hardware compilation represents a practical, high-efficiency paradigm for deploying frozen foundation models to silicon-constrained edge devices.

### 🎙️ Speaker Script (1 minute):
> *"In conclusion, this project demonstrates that compiling frozen Transformer weights directly into constant Verilog standard cells is not only feasible, but delivers overwhelming physical advantages: nearly a 4x reduction in logic area, an 18x reduction in storage flip-flops, and a 32% reduction in critical path delay—all while guaranteeing bit-exact arithmetic correctness.*  
> *The entire system is open-source, fully tested, and reproducible.*  
> *Thank you very much for your time. I am now open to your questions."*

---

## 🛡️ Backup Slides (For Committee & Professor Q&A)

### Backup Slide B1: Two's Complement Unary Negation Proof
* **Question:** *"Why did you strictly exclude -128 in your INT8 quantization?"*
* **Mathematical Answer:**
  * $N$-bit two's complement range is $[-2^{N-1}, 2^{N-1}-1] = [-128, +127]$.
  * Negating $-128$ in 8-bit arithmetic: invert bits of $\texttt{1000\_0000}_2 \to \texttt{0111\_1111}_2$, add $1 \to \texttt{1000\_0000}_2 = -128 \neq +128$.
  * Wrapping $-128$ around causes a catastrophic sign-inversion bug when compiling negative constant weights (e.g., $-x$). Restricting to $[-127, +127]$ guarantees exact negation closure.

### Backup Slide B2: Timing Waveform & Handshaking Protocol
* **Question:** *"Explain the clocking, reset, and handshaking of your Verilog wrapper."*
* **Answer:**
  * Uses synchronous active-low reset (`rst_n`).
  * On `posedge clk`, when `valid_in == 1`, input bus `x_in` is sampled into combinational adder trees.
  * Exactly 1 cycle later ($L=1$), `valid_out <= 1` and `y_out` holds the registered 32-bit accumulated dot products.
  * High-throughput: supports back-to-back streaming vectors every clock cycle without wait states.

### Backup Slide B3: Rule N10 Fairness Audit
* **Question:** *"Did you give Design F an unfair advantage over Design P during synthesis?"*
* **Answer:**
  * Rule N10 is verified automatically in `synth/fairness.py`:
  1. **Width Parity:** Identical input (8-bit signed) and output (32-bit signed) bitwidths.
  2. **Latency Parity:** Identical $L=1$ registered output stage.
  3. **Synthesis Flag Parity:** Identical Yosys passes (`read_verilog`, `synth -top <top> -flatten`, `stat -json`, `ltp`).
  4. Both designs include reset logic and valid handshaking registers.

### Backup Slide B4: Traceability Matrix (Source Artifacts)
* **Question:** *"Where in your repository is this specific metric recorded?"*
* **Answer:**
  * Quantization Quality: [`results/quant_quality.json`](file:///home/himesh/BTP/results/quant_quality.json)
  * Area & Gate Counts (E1): [`results/exp1_area.json`](file:///home/himesh/BTP/results/exp1_area.json)
  * Weight Distribution Analysis (E2): [`results/exp2_weights.json`](file:///home/himesh/BTP/results/exp2_weights.json)
  * Critical Path Depth (E3): [`results/exp3_timing.json`](file:///home/himesh/BTP/results/exp3_timing.json)
  * Extrapolation Model (E5): [`results/exp5_full_layer.json`](file:///home/himesh/BTP/results/exp5_full_layer.json)
  * Mutation Testing Logs: [`results/mutation_results.json`](file:///home/himesh/BTP/results/mutation_results.json)
