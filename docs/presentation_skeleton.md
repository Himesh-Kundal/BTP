# LLMHW BTP Defense: Presentation Skeleton & Slide Outline

**Recommended Timing:** 20 minutes presentation + 10 minutes viva Q&A  
**Target Audience:** Hardware Architecture & Embedded Systems Evaluation Committee  

---

## Slide 1: Title & Overview (1 min)
- **Title:** Model-Specific Hardware Compilation of Dense Transformer Projections into Synthesizable Constant-Weight Verilog
- **Author:** Himesh (B.Tech Candidate)
- **Supervisor:** Faculty Advisor / Evaluation Committee
- **Core Thesis:** *Hardcoding trained transformer weights as compile-time constants eliminates weight storage registers and memory ports, unlocking a 3.85× standard-cell area reduction and 17.7× flop reduction while guaranteeing bit-exact arithmetic equivalence.*

---

## Slide 2: Motivation — The Memory Wall in Edge Transformers (1.5 min)
- **Problem:** Transformer inference is severely memory-bandwidth and area bound.
- **Overhead of Generality:** Standard NPUs and systolic arrays maintain writable SRAM/register files, address decoders, and generic $8 \times 8$ multiplier trees to support arbitrary models.
- **The Specialization Opportunity:** In fixed-purpose edge appliances (robotics, aerospace, automotive sensors), weights are frozen post-training.
- **Key Question:** What are the exact physical and numeric trade-offs if we eliminate all weight ports and bake weights directly into silicon standard cells?

---

## Slide 3: System Overview & Core Claims (1.5 min)
- **Pipeline Architecture:** HuggingFace PyTorch Model $\to$ Model-Agnostic IR $\to$ Quantization Passes $\to$ Parametric Verilog Backend $\to$ Independent Oracle Verification $\to$ Yosys Logic Synthesis.
- **Claim C1 (Equivalence):** Bit-exact match ($\mathbf{Y}_{\text{RTL}} \equiv \mathbf{Y}_{\text{ref}}$) across all 7 projection types of SmolLM2-135M.
- **Claim C2 (Fidelity):** INT8 quantization satisfies frozen thresholds (mean cosine similarity 0.9996, relative error 3.81%).
- **Invariant I3:** Zero weight ports, zero weight storage flip-flops, zero writable memory arrays.

---

## Slide 4: Target Workload & Architectural Layering (1.5 min)
- **Workload:** SmolLM2-135M (30 layers, 134.5M parameters, 210 dense linear projections).
- **Projections Profiled:** MHA/GQA (`q_proj`, `k_proj`, `v_proj`, `o_proj`) + SwiGLU FFN (`gate_proj`, `up_proj`, `down_proj`).
- **Architectural Boundary (Golden Rule 5):**
  * `frontend/`: inspects PyTorch model, dumps `.npy` weights and calibration activations.
  * `ir/`, `compiler/`, `backend/`, `verification/`: strictly model-agnostic, zero torch dependencies, zero hardcoded model shapes.

---

## Slide 5: Numeric Contract & Quantization Pipeline (2 min)
- **Rule N2 (Weights):** Per-channel symmetric INT8 into $[-127, 127]$ with round-half-to-even tie breaking. Eliminates $-128$ to prevent two's complement negation overflow.
- **Rule N3 (Activations):** Static per-tensor symmetric INT8 calibration via forward hooks on fixed prompts (seed 42).
- **Rule N4 (Accumulator Bound):** Proves analytical worst-case dynamic range $K \times 127 \times 127 + |b_q|_{\max} < 2^{31}$ to guarantee 32-bit overflow-free accumulation.
- **Rule N5 (Bias):** Synthetic 32-bit bias scaled by $s_b = s_x \cdot s_w$.

---

## Slide 6: Backend RTL Generation — Design F vs Design P (2.5 min)
- **Design F (Fixed-Weight Specialization):**
  * Parametric $N_t \times K_t$ tiles ($8 \times 16$).
  * Verilog constant literals (`8'sd...`); zero-skipping prunes zero products; $\pm 1$ becomes wire reconnection.
  * Column adder trees reduce intermediate sums; synchronous $L=1$ registered wrapper.
- **Design P (Programmable Baseline):**
  * Internal $W_{\text{reg}}$ array ($N \times K \times 8$ bits) addressable via write port (`w_we`, `w_addr`, `w_data`).
  * Runtime multipliers ($x_k \times w_{nk}$) with identical 16-bit intermediate products and 32-bit accumulators.
- **Strict Fairness (Rule N10):** Identical widths, identical clocking, identical registered interface.

---

## Slide 7: Verification Methodology & Independent Oracle (2 min)
- **Golden Rule 2:** Correctness governed by independent NumPy INT64 reference oracle (`verification/reference.py`). Never computed from RTL or inside testbenches.
- **Verification Hierarchy:**
  * **Level L1:** 1,080-condition test matrix (10 shapes, 6 weight classes, 6 input stimulus classes).
  * **Level L2:** Real layer slices across all 7 types at layers {0, 14, 29} with real captured activations.
  * **Level L3:** Multi-tile cores testing inter-tile adder trees.
- **Cross-Simulator Parity:** Icarus Verilog and Verilator byte-identical outputs.
- **Static Weight Audit:** RTL parser verifies all compile-time literals match $W_q$ bit-for-bit.

---

## Slide 8: Mutation Testing Hardening (1.5 min)
- **Principle:** A test suite is only as trustworthy as the bugs it can kill.
- **9 Fault Models (M1–M9):** Sign flips, weight $\pm 1$, dropped adder terms, bit slicing off-by-one, accumulator truncation, latency disruption, and carry drops.
- **Results:** **100.0% Kill Rate** (9/9 mutants killed). Zero mutant survival.

---

## Slide 9: Synthesis Evaluation — Area & Storage Elimination (2.5 min)
- **Tool:** Yosys Open SYnthesis Suite (Generic standard gates & Lattice iCE40 FPGA LUT-4 proxy).
- **Experiment E1 (Area Scaling):**
  * Cell reduction: **3.75× to 3.85×** across tile grid $(4,16)$ to $(32,64)$.
  * At $16 \times 64$: Design F requires 155,648 cells vs 599,085 for Design P.
- **Storage Elimination (Flip-Flops):**
  * Flop reduction: **5.7× to 17.7×**.
  * Pruning of dead upper accumulator bits and 100% elimination of weight registers.
- **FPGA Density:** 2.34× LUT-4 reduction on iCE40.

---

## Slide 10: Synthesis Evaluation — Weight Content & Timing (2 min)
- **Experiment E2 (Weight Distribution Effect):**
  * Real trained weights are **23.0% smaller** than uniform random INT8 ($22,764$ vs $28,006$ cells) due to Gaussian zero-clustering.
  * Spatial position permutation has $<0.3\%$ effect.
  * Zero-skipping prunes 99.99% of dead datapath terms.
- **Experiment E3 (Critical Path Depth):**
  * Constant shift-add eliminates partial-product reduction stages.
  * Topological critical path depth (`ltp`) drops by **20% to 32%** (ratio 0.68–0.74).

---

## Slide 11: Full-Layer Extrapolation Model & Validation (1.5 min)
- **Extrapolation Model (E5):** Tile-sum linear model for full layers.
- **Empirical Validation:** Directly synthesized $16 \times 32$ multi-tile core ($4$ tiles of $8 \times 16$):
  * Direct synthesis: 82,160 cells vs Tile-sum: 91,056 cells (**10.82% estimation error**, conservative overestimate).
- **Full SmolLM2 Layer 0 Estimation (3,456 tiles):**
  * Design F: **570.6M cells** [estimate] vs Design P: **2.075B cells** [estimate] (**3.64× Area Reduction**).

---

## Slide 12: Discussion, Caveats, & Scope Boundaries (1.5 min)
- **What the Data Proves:**
  * Constant-weight specialization provides drastic logic and flop reduction.
  * Independent oracle and mutation testing prove 100% functional correctness.
- **Pre-Layout Caveats:**
  * Cell counts reflect logic synthesis. Post-P&R interconnect delay and routing congestion depend on physical PDKs.
  * Weight immutability requires FPGA bitstream reload or ASIC respins for updated weights.
- **Part I Scope:** Linear projections only. Non-linear operators (RMSNorm, Softmax, RoPE, SiLU) evaluated in host software.

---

## Slide 13: Live Demonstration Walkthrough (1 min)
- `make demo`: Compiles and verifies all 7 projections of layer 0 in **3.58 seconds**.
- Single command shows: extract $\to$ compile $\to$ RTL codegen $\to$ static audit PASS $\to$ bit-exact simulation PASS $\to$ synthesis summary.

---

## Slide 14: Roadmap to Part II & Part III (0.5 min)
- **Part II:** Fused RMSNorm, SwiGLU hardware multipliers, pipelined adder trees ($L > 1$), and ping-pong activation BRAM streaming.
- **Part III:** Physical design tapeout on SkyWater 130nm / TSMC 28nm using OpenLane/Cadence.

---

## Slide 15: Conclusion (0.5 min)
- Automated hardware compilation of LLM linear layers into constant-weight Verilog is viable, highly scalable (0.54s for $576 \times 576$), bit-exact, and delivers a **3.85× area savings** and **17.7× flop reduction**.
- Fully reproducible open-source release with 226 passing tests.

---

## Backup Slides (For Committee Q&A)
- **Slide B1: Waveform Walkthrough**: Inspection of clock, active-low reset, valid handshaking, and 1-cycle latency pipeline transaction.
- **Slide B2: Mutation Testing Kill Matrix**: Breakout of killed mutants M1–M9 across test stages.
- **Slide B3: Fairness Auditor Rules**: Automated verification of identical bitwidths, synthesis commands, and flattening.
- **Slide B4: Numerical Traceability Table**: Instant mapping of any committee inquiry to source artifact in `results/`.
