# BTP Defense: Complete Word-for-Word Speaker Script
**Project:** Model-Specific Hardware Compilation of Dense Transformer Projections into Synthesizable Constant-Weight Verilog  
**Presenter:** Himesh Kundal (Roll No: 23EC10030)  
**Supervisor:** Prof. Mrityunjoy Chakraborty  
**Department:** Electronics and Electrical Communication Engineering, IIT Kharagpur  
**Total Target Timing:** 18–20 Minutes Presentation + 10 Minutes Viva Q&A  

---

## 🎙️ General Presentation Delivery Tips for Defense

1. **Pacing:** Speak at a measured, calm pace (~120–130 words per minute). Don't rush through equations—give the committee 2–3 seconds to look at a chart before explaining it.
2. **Body Language & Pointers:** When mentioning numbers like **3.85×** or **17.7×**, point directly to the corresponding column or badge on the slide.
3. **Handling Interruptions:** If Prof. Chakraborty interrupts with a technical question (e.g., regarding two's complement overflow or CSD recoding), say: *"That is a very pertinent point, sir. Let me address that directly..."* and use the exact derivation from the script.
4. **Pronunciation & Acronyms:**
   - **RTL:** "R-T-L" (Register-Transfer Level)
   - **GQA:** "G-Q-A" (Grouped-Query Attention)
   - **SwiGLU:** "Swi-GLU" (Swish-Gated Linear Unit)
   - **CSD:** "C-S-D" (Canonical Signed Digit)
   - **LTP:** "L-T-P" (Longest Topological Path)
   - **Yosys:** "Yo-sis"

---

## Slide 1: Title Slide — The Core Thesis
* **Target Timing:** 0:00 – 1:15 (1 min 15 sec)
* **Visual Cue:** Look at the committee members, make eye contact, and introduce yourself confidently.

### 🗣️ Spoken Script:
> "Good morning, respected committee members and Professor Chakraborty.
>
> My name is Himesh Kundal, Roll Number 23EC10030, a final-year B.Tech student in the Department of Electronics and Electrical Communication Engineering. Today, I am presenting my B.Tech Project Part I titled: **'Model-Specific Hardware Compilation of Dense Transformer Projections into Synthesizable Constant-Weight Verilog'**, conducted under the supervision of Prof. Mrityunjoy Chakraborty.
>
> In this project, we explore a fundamental architectural opportunity in edge computing:
> When deploying Large Language Models to specialized, fixed-purpose edge appliances—such as flight controllers, automotive radar systems, or biomedical edge processors—the trained weights are frozen and will never change during deployment.
>
> Yet today, we force these models to run on general-purpose hardware accelerators that dedicate huge portions of their silicon area and power to memory interfaces, address decoders, and generic multipliers.
>
> In this project, I have designed and implemented an end-to-end automated compiler called **LLMHW** that eliminates weight memories completely by baking pre-trained weights directly into immutable standard-cell logic. Our synthesized circuits achieve a **3.85× standard-cell area reduction**, up to a **17.7× flip-flop reduction**, and a **32% faster critical path**, all while guaranteeing **bit-exact arithmetic equivalence** with the original model."

---

## Slide 2: Problem Statement — The Memory Wall & The Generality Tax
* **Target Timing:** 1:15 – 2:45 (1 min 30 sec)
* **Visual Cue:** Point to the left diagram (General NPU) and then contrast with the right diagram (Fixed Appliance).

### 🗣️ Spoken Script:
> "To motivate this work, let us examine the fundamental bottleneck of running Transformer models at the edge: the Memory Wall.
>
> Standard Neural Processing Units, such as GPUs, TPUs, or systolic arrays, are designed for general-purpose programmability. Because a programmable accelerator does not know what model it will execute tomorrow, it must incorporate wide weight register files, multi-banked SRAM arrays, write-enable control logic, and full 8-by-8 Baugh-Wooley multipliers capable of multiplying any two runtime variables.
>
> We call this the **'Generality Tax'**. In silicon, this generality is extremely expensive. Up to 75% of the total chip area in a programmable inference core is spent merely storing, buffering, and routing weights.
>
> However, in many critical edge appliances, this generality is completely unnecessary. Once a model is validated and deployed, its weights remain frozen. If the model is ever updated, it is done through a firmware re-flash or bitstream update.
>
> This raises our core research question:
> *If weights are constant, what happens to silicon area, timing, and storage if we strip away every weight memory port and compile the weights directly into the silicon logic gates?*"

---

## Slide 3: System Overview — The LLMHW Compiler Architecture
* **Target Timing:** 2:45 – 4:15 (1 min 30 sec)
* **Visual Cue:** Trace the horizontal block diagram from left to right with your pointer.

### 🗣️ Spoken Script:
> "To answer this question rigorously, I developed **LLMHW**, a fully automated hardware compilation framework.
>
> The flow begins at the top left with pre-trained PyTorch models from HuggingFace. The frontend extracts every linear projection matrix and captures real activation distributions using forward execution hooks.
>
> Next, our quantizer performs symmetric INT8 quantization.
>
> At this boundary, we enforce a critical software engineering principle—**Golden Rule 5: Strict Architectural Layering**. The PyTorch framework is strictly isolated to the frontend. All downstream compiler passes, intermediate representations, Verilog generators, and verifiers operate on a model-agnostic representation called **TileManifest**.
>
> The backend compiler ingests this manifest and generates synthesizable Verilog-2001 code for two distinct targets:
> First, **Design F**, our specialized constant-weight architecture where weights are embedded directly as Verilog literals.
> Second, **Design P**, a strictly fair programmable baseline with memory-mapped weight registers.
>
> Finally, both designs undergo formal interface audits, dual-tier cycle-accurate verification against an independent 64-bit NumPy oracle, and logic synthesis using Yosys."

---

## Slide 4: Target Workload — SmolLM2-135M Projection Breakdown
* **Target Timing:** 4:15 – 5:45 (1 min 30 sec)
* **Visual Cue:** Direct the committee's attention to the table showing the 7 projections.

### 🗣️ Spoken Script:
> "To ground our experiments in a state-of-the-art open-source architecture, we evaluated **SmolLM2-135M**, a 134.5-million-parameter compact language model featuring 30 Transformer layers.
>
> Inside each Transformer block, there are 7 dense linear projections:
> In the Self-Attention mechanism, we have the query projection `q_proj` of size 576-by-576, the grouped-query key and value projections `k_proj` and `v_proj` of size 192-by-576, and the output projection `o_proj`.
> In the SwiGLU Feed-Forward Network, we have `gate_proj` and `up_proj` expanding the hidden state to 1,536 dimensions, followed by `down_proj` reducing it back to 576.
>
> In total, a single layer contains over 3.5 million weights, and across all 30 layers, these 7 linear projections represent **106.2 million parameters**—which is more than 78% of the entire model's parameter footprint.
>
> By evaluating all 7 distinct projection geometries, our compilation and synthesis results reflect the exact structural characteristics of real-world Transformer inference."

---

## Slide 5: Mathematical Rigor — Quantization & Bit-Growth Proofs
* **Target Timing:** 5:45 – 7:45 (2 min 00 sec)
* **Visual Cue:** Point to the two's complement negation formula and the bit-growth derivation equation.

### 🗣️ Spoken Script:
> "Now, let us delve into the numeric contracts and hardware arithmetic.
>
> For weights and activations, we utilize symmetric INT8 quantization. Crucially, as defined in Rule N2, we clip all integer values strictly to the range **[-127, +127]**, deliberately forbidding the value -128.
>
> Why must -128 be forbidden?
> In standard 8-bit two's complement arithmetic, the representable range is -128 to +127. If an activation or weight takes the value -128, and our constant optimizer performs a hardware unary negation—meaning zero minus the value—inverting the bits of binary 1000_0000 and adding 1 produces 1000_0000, which is still -128!
> Because +128 cannot be represented, two's complement negation wraps around, causing a catastrophic sign error. By bounding the domain to [-127, +127], unary negation is mathematically guaranteed to be closed and exact.
>
> Furthermore, we employ **Round-Half-to-Even (Banker's Rounding)**. Unlike round-half-up, which introduces a positive DC bias when summing hundreds of terms, round-half-to-even distributes rounding errors symmetrically, guaranteeing zero-mean statistical bias. Across all 7 layers, our quantization preserves a cosine similarity exceeding 0.9994.
>
> Next, look at the accumulator bit-growth derivation on the slide:
> The maximum product of two INT8 integers is 127 times 127, which equals 16,129.
> When accumulating over a vector length $K$, the analytical worst-case sum is bounded by $K$ times 16,129.
> For $K=576$, the maximum sum is 9.29 million, requiring 25 signed bits. For down_proj with $K=1536$, it requires 26 signed bits.
> By allocating a standard 32-bit signed accumulator (`[31:0]`), we mathematically guarantee **6 to 7 bits of overflow headroom** even under the most adversarial pathological inputs."

---

## Slide 6: RTL Architecture — Design F vs. Design P
* **Target Timing:** 7:45 – 9:30 (1 min 45 sec)
* **Visual Cue:** Compare the Verilog code blocks: show the flip-flop array in Design P vs. the constant expressions in Design F.

### 🗣️ Spoken Script:
> "Now let us examine how this translates into synthesizable Verilog-2001 RTL.
>
> In **Design P**, our programmable baseline, we instantiate an internal register file addressable through write-enable and address ports. In the datapath, runtime variable wires `x` and `w` are multiplied together, forcing the synthesis tool to infer full generic multipliers.
>
> In contrast, in **Design F**, our specialized architecture, all weight ports and registers are completely eliminated. Our compiler directly emits Verilog constant literals of the form `8'sd<W>`.
>
> This enables four immediate source-level optimizations in `compiler/weight_encoder.py`:
> 1. **Zero-skipping:** Whenever a weight is zero, the term is omitted from the adder tree entirely.
> 2. **Trivial multiplier replacement:** If a weight is +1, it is emitted as a bare signed wire `$signed(x)`. If -1, it becomes a simple negation `-$signed(x)`, requiring only an inverter and carry-in.
> 3. **Parenthesized signed literals:** We emit `(-8'sd42)` to ensure strict IEEE 1364-2001 signed evaluation.
>
> To ensure absolute scientific fairness under Rule N10, both designs maintain identical bus widths, identical reset logic, and an identical single-cycle registered output latency ($L=1$)."

---

## Slide 7: VLSI Multiplier Architecture — Baugh-Wooley vs. CSD Trees
* **Target Timing:** 9:30 – 11:15 (1 min 45 sec)
* **Visual Cue:** Emphasize the gate count differences between the array multiplier diagram and the shift-add tree.

### 🗣️ Spoken Script:
> "From a VLSI Signal Processing perspective, what happens physically inside the silicon when we replace a variable multiplier with a constant multiplier?
>
> In Design P, to multiply an 8-bit signed input by an 8-bit signed variable, the synthesizer must construct an 8-by-8 **Baugh-Wooley array multiplier**.
> This requires generating 64 partial-product bits using AND gates, followed by a Carry-Save Adder matrix containing roughly 48 Full Adders and 8 Half Adders, ending in a Carry-Propagate Adder. A single generic multiplier costs **350 to 450 logic gates** and introduces 25 to 35 gate levels of logic delay!
>
> In Design F, when the multiplier is a compile-time constant, the silicon changes completely:
> First, zero partial-product AND gates are needed because the bits of the constant are known statically.
> Second, bit-shifts in hardware cost **zero transistors**—a shift like `x << 3` is merely a physical wire routing!
> Third, using **Canonical Signed Digit (CSD) recoding**, any constant is rewritten to minimize non-zero digits. For instance, multiplying by 63, which has six ones in binary, is rewritten as $(x \ll 6) - x$—reducing the entire 400-gate multiplier down to **a single 16-bit subtractor**!
>
> Because real LLM weights follow a Gaussian distribution centered near zero, the average cost per weight in Design F drops to **less than 1.2 adders per tap**."

---

## Slide 8: Multi-Tile Spatial Partitioning & Inter-Tile Adder Trees
* **Target Timing:** 11:15 – 12:30 (1 min 15 sec)
* **Visual Cue:** Show how the 2D matrix is sliced into tiles and reduced at the top level.

### 🗣️ Spoken Script:
> "To scale our architecture to full Transformer dimensions without hitting single-module complexity limits, LLMHW partitions large matrices into a 2D grid of modular tiles, such as 8 rows by 16 columns.
>
> An essential architectural requirement was preserving single-cycle latency parity ($L=1$).
> If each sub-tile contained its own output flip-flop, and the top-level adder tree also had a flip-flop, the total pipeline latency would become 2 cycles.
>
> Instead, our compiler generates **combinational sub-tiles**. Each tile computes partial dot products over its 16 columns.
> At the top level, an **Inter-Tile 32-Bit Adder Reduction Tree** sums the partial outputs across all column tiles in parallel and adds the bias.
>
> A single synchronous register stage at the boundary latches `y_out` and `valid_out` on the rising clock edge.
> This guarantees strict single-cycle throughput and latency, while making tiled hardware bit-for-bit identical to monolithic RTL."

---

## Slide 9: Synthesis Results — Standard-Cell Area & Flop Elimination
* **Target Timing:** 12:30 – 14:15 (1 min 45 sec)
* **Visual Cue:** Point to the 3.85× area ratio and the 17.78× flip-flop ratio in the table.

### 🗣️ Spoken Script:
> "Now let us examine the synthesis results. We synthesized our designs using **Yosys 0.52** targeting generic CMOS standard-cell gates.
>
> Look at the results of Experiment E1 in the table:
> Across all tile sizes, from 4-by-16 up to 16-by-64, Design F achieves a **3.75× to 3.85× reduction in standard-cell gate count** compared to Design P. At 16-by-64, gate count drops from nearly 600,000 gates down to 155,000 gates—a **74% silicon area savings**.
>
> When we look at sequential storage, the results are even more striking:
> Flip-flop counts drop by **5.7× to 17.7×**.
>
> There are two reasons for this massive flip-flop reduction:
> First, we completely eliminate the 1,024 weight storage flip-flops of Design P.
> Second, because our weights are bounded constants, Yosys statically proves that the upper bits of the 32-bit accumulator can never toggle independently of the sign bit for a given tile. Yosys's optimization pass automatically prunes the upper 16 flip-flops, tying them directly to the sign bit at zero cost!
>
> Notice also Design P_bus: even when weight registers are removed from the baseline and weights are fed via primary ports, Design F is still 3.77× smaller. This proves that the savings are driven by arithmetic constant specialization, not merely the removal of memory."

---

## Slide 10: Synthesis Results — Weight Content & Timing Path Analysis
* **Target Timing:** 14:15 – 15:45 (1 min 30 sec)
* **Visual Cue:** Point to the 23% sparsity savings bar and the LTP reduction from 56 to 38.

### 🗣️ Spoken Script:
> "In Experiments E2 and E3, we investigated two deeper questions: Does the real weight distribution of an LLM matter, and what happens to the clock frequency?
>
> In Experiment E2, we synthesized tiles using uniform random INT8 weights versus real trained weights from SmolLM2-135M.
> Compiling real trained weights produced circuits that were **23.0% smaller** than uniform random weights—22,764 gates versus 28,006 gates!
> Why? Because pre-trained neural network weights are Gaussian-distributed and clustered tightly around zero. This causes our zero-skipping and small-constant CSD optimizations to trigger far more frequently.
>
> In Experiment E3, we evaluated the Longest Topological Path (`ltp`) through the circuits.
> In Design P, signals must propagate through the full Baugh-Wooley multiplier array and the adder tree, resulting in a critical path of **56 gate levels**.
> In Design F, because constant shift-and-add trees eliminate the deep multiplier stages, the critical path drops to **38 gate levels**—a **32% reduction in logic depth**.
>
> Finally, FPGA proxy synthesis on Lattice iCE40 devices demonstrates a **2.34× reduction in 4-input LUTs**."

---

## Slide 11: Full-Model Extrapolation & Multi-Tile Validation
* **Target Timing:** 15:45 – 17:00 (1 min 15 sec)
* **Visual Cue:** Highlight the 10.82% validation accuracy card and the full-model 6.1-billion gate figure.

### 🗣️ Spoken Script:
> "To estimate the area of a full Transformer layer, we developed an analytical scaling model in Experiment E5. The model accounts for the sum of all individual tile cores plus the $N \times (C_{\text{tiles}}-1)$ thirty-two-bit inter-tile adders.
>
> To validate this formula, we synthesized a complete 4-tile, 16-by-32 core in Yosys. The analytical model predicted 91,056 gates, while direct synthesis produced 82,160 gates—an estimation error of **10.82%**, confirming that our analytical model acts as a safe, conservative upper bound.
>
> Extrapolating to the full SmolLM2-135M architecture:
> A single 576-by-576 `q_proj` layer requires **13.59 million gates** in Design F, compared to 49.43 million in Design P.
> Across all 210 projections in the 30-layer model, compiling weights into constant standard cells requires approximately **6.10 billion gates**, compared to **22.18 billion gates** for a programmable design—saving over 16 billion gates of silicon area."

---

## Slide 12: Independent Verification & 100% Mutation Kill
* **Target Timing:** 17:00 – 18:30 (1 min 30 sec)
* **Visual Cue:** Walk through the 3 tiers on the slide and emphasize the 100% kill badge.

### 🗣️ Spoken Script:
> "When compiling hardware automatically, verification is of paramount importance. A single inverted bit slice would corrupt the entire model. We established a three-tier verification harness:
>
> **Tier 1:** An independent NumPy 64-bit reference oracle. We verified bit-exact equivalence across 1,080 test conditions and all 7 SmolLM2 projections with zero mismatches.
>
> **Tier 2:** A static RTL weight and interface auditor. This tool parses the generated Verilog source code, checks that no forbidden weight ports exist, and extracts all 331,776 constant literals from the arithmetic expressions, confirming 100% bit-exact identity with the PyTorch tensor in **0.85 seconds**.
>
> **Tier 3:** Mutation testing. To prove our verification suite is robust, we injected 9 deliberate hardware fault mutants—including weight sign inversions, dropping signed keywords, and altering adder carry paths.
> Our test suite achieved a **100% kill rate (9 out of 9 mutants killed)**, with zero false positives or undetected bugs."

---

## Slide 13: Live Toolchain Demonstration
* **Target Timing:** 18:30 – 19:15 (0 min 45 sec)
* **Visual Cue:** Point to the 2.97-second execution timeline.

### 🗣️ Spoken Script:
> "The entire compilation and verification process is completely automated under the `llmhw` command-line tool.
>
> With a single command, `make demo`, the framework extracts all 7 projections of layer 0 from HuggingFace, performs quantization, generates synthesizable Verilog, verifies interface invariants, executes cycle-accurate Icarus Verilog simulations, and produces a complete synthesis summary in just **2.97 seconds**.
>
> The repository includes **230 automated unit, integration, and property tests**, all passing cleanly."

---

## Slide 14: Scope Boundaries, Limitations & Engineering Trade-offs
* **Target Timing:** 19:15 – 20:15 (1 min 00 sec)
* **Visual Cue:** Use an open, honest academic tone to discuss boundaries.

### 🗣️ Spoken Script:
> "To maintain complete academic rigor, we must clearly articulate the engineering boundaries of Part I:
>
> First, our metrics reflect pre-layout standard-cell synthesis. While Yosys provides highly accurate logic gate counts and topological path depths, physical place-and-route interconnect parasitics and routing congestion in a specific foundry PDK will introduce secondary effects.
>
> Second, constant-weight hardware is physically immutable. If the model weights are changed, an FPGA must be re-flashed or a new ASIC fabricated. Therefore, this paradigm is specifically targeted at high-volume, fixed-purpose edge deployments.
>
> Third, Part I focuses strictly on linear matrix projections, which represent over 78% of the model parameters. Non-linear operations—specifically RMSNorm, RoPE, and SwiGLU activation functions—are currently computed on the host processor."

---

## Slide 15: Future Roadmap — Part II & Part III
* **Target Timing:** 20:15 – 21:00 (0 min 45 sec)
* **Visual Cue:** Walk through the roadmap phases for the upcoming semester.

### 🗣️ Spoken Script:
> "Our roadmap for Part II and Part III directly addresses these limitations:
>
> In **Part II (Spring 2027)**, we will design dedicated synthesizable hardware blocks for RMSNorm, SwiGLU non-linearities, and RoPE rotary embeddings, enabling full Transformer blocks to execute completely on silicon without host roundtrips.
> We will also introduce multi-stage pipelining inside the column adder trees to achieve clock frequencies of 500 MHz and above, alongside ping-pong Block RAM streaming for continuous token generation.
>
> In **Part III**, we aim to take the verified core through physical ASIC tapeout using the open-source **OpenLane flow and SkyWater 130nm PDK**, producing final GDSII layouts and tapeout-ready silicon sign-off."

---

## Slide 16: Conclusion & Summary
* **Target Timing:** 21:00 – 21:45 (0 min 45 sec)
* **Visual Cue:** Conclude warmly and invite questions.

### 🗣️ Spoken Script:
> "In conclusion:
> This project demonstrates that compiling frozen Transformer weights directly into constant Verilog standard cells is practical, fast, and delivers substantial silicon advantages:
> - A **3.85× reduction in standard-cell logic area**,
> - A **17.7× reduction in flip-flop storage**,
> - A **32% reduction in critical path delay**,
> - And **100% bit-exact mathematical equivalence**.
>
> The entire framework is open-source, fully documented, and backed by a comprehensive 14-page project report.
>
> Thank you very much for your time and attention. I am now delighted to take your questions."

---

## 🛡️ Backup Viva Q&A Defense Scripts

Use these exact prepared responses if Professor Chakraborty or the committee asks these common questions during the viva:

### Q1: "Why did you forbid -128 in your INT8 quantization? Isn't -128 a valid 8-bit number?"
> *"Yes, sir. In standard two's complement, an 8-bit signed wire represents integers from -128 to +127.
> However, -128 is an asymmetric outlier because its positive counterpart, +128, does not exist in 8 bits.
> When our compiler optimizes negative weights—for example, multiplying by -1—it emits a unary negation, which is zero minus the input.
> If an input were -128, computing 0 minus -128 requires inverting 1000_0000 to 0111_1111 and adding 1, which results in 1000_0000—which is still -128!
> The negation wraps around to a negative number, creating a severe sign inversion error. By strictly clipping both weights and activations to the symmetric range [-127, +127], unary negation is mathematically guaranteed to be closed and exact."*

### Q2: "Can your 32-bit accumulator ever overflow under extreme inputs?"
> *"No, sir, it cannot. We derived an analytical proof for the maximum accumulator dynamic range:
> The maximum product of any two inputs in [-127, +127] is 127 times 127, which is 16,129.
> For an inner product of length $K$, the worst-case accumulation occurs when every single input is +127 or -127 matching the sign of the weight.
> For our largest attention projection, $K=576$, the maximum sum is $576 \times 16{,}129 = 9{,}290{,}304$, which requires only 25 signed bits.
> For down_proj with $K=1536$, the maximum sum is $24{,}774{,}144$, which requires 26 signed bits.
> Since our accumulator is 32 bits wide, the maximum representable signed value is 2.14 billion. This leaves **6 to 7 full guard bits**, guaranteeing zero overflow under all possible inputs."*

### Q3: "Why is Design F 32% faster in critical path if the adder tree is still summing 16 or 32 inputs?"
> *"Because in Design P, the signal must first pass through a full 8-by-8 Baugh-Wooley multiplier before reaching the adder tree. That multiplier alone contains 6 to 8 stages of Full Adders plus a carry-propagate adder, contributing roughly 25 to 30 gate levels of delay.
> In Design F, the constant multiplier uses Canonical Signed Digit recoding. Bit-shifts are zero-delay wire connections, and most weights require only 0, 1, or at most 2 adder stages before entering the column reduction tree.
> By eliminating the deep partial-product reduction stages of generic multipliers, the total Longest Topological Path measured by Yosys drops from 56 gate levels down to 38 gate levels—a 32% reduction."*

### Q4: "How do you ensure you didn't give Design F an unfair advantage over Design P?"
> *"We formalized Rule N10 in our specification, which is automatically enforced by `synth/fairness.py`:
> 1. Both designs take identical 8-bit signed inputs and produce identical 32-bit signed outputs.
> 2. Both designs have an identical registered output stage with a latency of $L=1$ cycle.
> 3. Both designs are synthesized using the exact same Yosys commands: `read_verilog`, `synth -flatten`, `stat -json`, and `ltp`.
> 4. Furthermore, we even created Design P_bus, where weight registers were eliminated and replaced with primary ports. Even against Design P_bus, Design F is still 3.77× smaller, proving that the area savings come from constant logic specialization rather than storage removal."*
