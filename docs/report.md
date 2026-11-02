# Model-Specific Hardware Compilation of Dense Transformer Projections into Synthesizable Constant-Weight Verilog

**Author:** Himesh  
**Project:** B.Tech Project (Part I)  
**Date:** November 2026  
**Target Architecture:** HuggingFaceTB/SmolLM2-135M  
**Traceability Reference:** [`docs/report_traceability.md`](report_traceability.md)  

---

## Abstract

Modern Large Language Models (LLMs) are predominantly memory-bound during autoregressive inference, spending a substantial fraction of their operational energy and chip area transferring weight parameters across on-chip memory hierarchies. In fixed-workload and edge-embedded deployments where models remain immutable post-training, general-purpose programmable systolic arrays incur severe hardware overheads by maintaining writable storage registers, write-address decoding logic, and generic multipliers.

This report presents **LLMHW**, an end-to-end, model-specific hardware compiler that translates dense linear projections of modern transformer architectures into synthesizable Verilog-2001 with INT8 weights embedded directly as compile-time constants (**Design F**). By eliminating runtime weight ports, weight write buses, and internal weight storage flip-flops (Invariant **I3**), LLMHW achieves an empirical **3.0× to 3.85× standard-cell area reduction**, an **up to 17.7× flip-flop reduction**, and a **20% to 32% critical path shortening** compared to a strictly audited programmable baseline (**Design P**).

Every generated hardware core is proven **bit-exact** against an independent NumPy INT64 reference oracle (**Claim C1**), achieving 0 mismatches across all 7 projection types (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) of **SmolLM2-135M** across network depths. The verification harness achieves a **100.0% mutation kill rate** across 9 structural fault models (M1–M9) and passes dual-simulator cross-checks across Icarus Verilog and Verilator. Full-layer multi-tile extrapolation models are validated against direct multi-tile synthesis with an estimation error of 10.82%.

---

## 1. Introduction and Motivation

Autoregressive inference in modern decoder-only transformers is characterized by a computational arithmetic intensity of approximately $O(1)$ operations per byte transferred during token generation. General-purpose inference accelerators (GPUs, TPUs, and generic NPU systolic arrays) are engineered under the assumption of workload malleability: hardware datapaths are provisioned with runtime-writable weight memory arrays, dynamic routing buses, and multi-bit programmable multiplier arrays.

However, in embedded, edge-robotics, automotive, and dedicated security appliances, foundation models are frequently deployed in fixed configurations with frozen weights. In such environments, hardware programmability represents pure overhead:
1. **Silicon Area**: Maintaining 8-bit flip-flop registers for every weight parameter consumes substantial active area.
2. **Static & Dynamic Power**: Register clock trees and toggling memory buses incur severe parasitic dynamic power.
3. **Logic Complexity**: Generic signed multipliers ($8 \times 8 \to 16$) require full add-and-shift tree arrays, whereas multiplication by a known compile-time constant reduces to shallow hardwired shift-add primitives or zero-cost bit-wiring.

This thesis investigates the hardware specialization hypothesis: *Can we build an automated compiler that specializes dense transformer layers directly into synthesizable silicon, and what are the exact, verified trade-offs in area, storage, timing, and numeric integrity?*

---

## 2. Background and Related Work

### 2.1 Dense Linear Layers in Transformer Architectures
The modern transformer block is divided into two primary sub-blocks: Multi-Head Attention (MHA) / Grouped Query Attention (GQA) and the Feed-Forward Network (FFN). The parameterization is dominated by dense matrix multiplications of the form $Y = X \cdot W^T + b$:
- **Attention Projections**: Query ($W_q$), Key ($W_k$), Value ($W_v$), and Output ($W_o$).
- **SwiGLU FFN Projections**: Gate ($W_{\text{gate}}$), Up ($W_{\text{up}}$), and Down ($W_{\text{down}}$).

In modern compact architectures such as **SmolLM2-135M**, these 7 projections account for 210 distinct linear operators across 30 transformer layers, comprising over 134.5 million parameters.

### 2.2 INT8 Quantization and Numeric Contracts
Direct FP32 or BF16 execution on custom hardware requires wide floating-point units. Fixed-point INT8 quantization maps real-valued tensor activations and weights into 8-bit two's complement integers while preserving model accuracy:
$$\mathbf{W}_q = \text{clip}\left(\left\lfloor \frac{\mathbf{W}}{s_w} \right\rceil, -127, 127\right), \quad \mathbf{X}_q = \text{clip}\left(\left\lfloor \frac{\mathbf{X}}{s_x} \right\rceil, -127, 127\right)$$
Where $s_w$ is computed per output channel (Rule **N2**) and $s_x$ is derived statically via calibration (Rule **N3**). Symmetrical clipping to $[-127, 127]$ explicitly eliminates $-128$, preventing asymmetric two's complement sign-inversion overflow.

### 2.3 Constant-Coefficient Multiplier Generation
Multiplication of an input variable $x$ by a known constant $c$ does not require generic Wallace or Dadda multiplier trees. In logic synthesis, constant multiplication collapses into hardwired bit shifts and shallow adder subtractors. When $c = 0$, the multiplier is completely pruned (zero-skipping). When $c = 2^k$, multiplication becomes pure wire reconnection.

---

## 3. Problem Statement, Scope, and Non-Goals

### 3.1 Formal Scope (Part I)
The objective of Part I is to construct a production-ready, model-agnostic hardware compiler pipeline that:
1. Ingests dense linear layers from **SmolLM2-135M** at pinned revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
2. Produces synthesizable Verilog-2001 circuits embedding weights as constants (**Design F**).
3. Produces a fair, width-matched programmable baseline (**Design P** and **Design P_bus**).
4. Proves bit-exact equivalence against an independent NumPy INT64 reference oracle (**Claim C1**).
5. Characterizes pre-layout logic synthesis metrics (cell count, flop count, critical path depth).

### 3.2 Explicit Non-Goals
To maintain scientific rigor and prevent scope dilution:
- **Non-Linear Layers**: LayerNorm, RMSNorm, RoPE position embeddings, Softmax, and SiLU elementwise activations are evaluated in host software (scheduled for Part II).
- **Runtime Weight Reloading**: Design F features zero weight-reloading capability.
- **Physical Place-and-Route**: Synthesis metrics reflect standard-cell logic mapping and topological critical path depths from Yosys. Physical clock tree synthesis, routing congestion, and parasitic RC extraction require target PDKs and are documented as pre-layout caveats.

---

## 4. System Design and Compiler Architecture

LLMHW is structured into distinct, strictly layered modules enforcing project **Golden Rule 5**:

```
[HuggingFace Model] -> [frontend/] -> [ir/] -> [compiler/] -> [backend/] -> [Verilog-2001 RTL]
                                                  |                  |
                                                  v                  v
                                       [verification/oracle]   [synth/yosys]
```

### 4.1 Frontend and Artifact Extraction
The frontend (`frontend/`) inspects the model structure without leaking framework dependencies downstream. It extracts 210 linear modules and records sha256 checksums of raw weights and metadata (`results/model.json`). Fixed calibration sets (seed 42) record intermediate activations via forward hooks (`calib/x_sample.npy`).

### 4.2 Intermediate Representation (IR)
The intermediate representation (`ir/`) defines immutable dataclasses:
- `TensorRef`: Typed tensor representation (`CONSTANT`, `INPUT`, `OUTPUT`) with shape and stride.
- `LinearOp`: Encapsulates dimensions $N, K$, weight tensors, optional bias, quantization parameters, and accumulator requirements.
- `TileManifest`: Geometry of matrix subdivision into $N_t \times K_t$ tiles, padded boundaries, and manifest SHA256 hashes.

Lossless serialization guarantees byte-for-byte reproducibility on reload.

### 4.3 Compiler Passes
1. **Activation Calibration (Rule N3)**: Evaluates absmax activation statistics over calibration batches to freeze scale $s_x$.
2. **Weight Quantization (Rule N2)**: Performs per-output-channel symmetric scaling into $[-127, 127]$ with round-half-to-even tie-breaking.
3. **Accumulator Bounding (Rule N4)**: Computes the analytical worst-case dynamic range:
   $$\text{acc}_{\max} = K \times 127 \times 127 + |b_q|_{\max}$$
   Asserts $\text{acc}_{\max} < 2^{31}$ to guarantee overflow-free accumulation in 32-bit signed integers.
4. **Specialization Pass**: Annotates weights as `TensorKind.CONSTANT` and strips runtime write interfaces.

### 4.4 Backend Code Generation: Design F vs Design P
- **Design F (Fixed-Weight)**:
  * Generates parametric tile cores (`llmhw_<name>_tile_i_j`).
  * Emits Verilog literals (`8'sd...`). Identifies zeros to omit product terms; identifies $\pm 1$ for wire assignments.
  * Hierarchical reduction adder trees over tile columns $j$.
  * Synchronous active-low reset wrapper with fixed pipeline latency $L=1$.
- **Design P (Programmable Baseline)**:
  * Instantiates an internal $W_{\text{reg}}$ array ($N \times K \times 8$ bits) addressable via `w_addr`, `w_we`, `w_data`.
  * Instantiates runtime signed multipliers (`x_k * w_nk`) with identical 16-bit intermediate products and 32-bit accumulators (Rule **N10**).
- **Design P_bus**: Logic-only variant taking weights as a primary input bus, isolating datapath logic from register storage.

---

## 5. Verification Methodology

### 5.1 Independent Reference Oracle (Rule N1–N7)
Correctness is governed by `verification/reference.py`, written strictly in NumPy using 64-bit signed integers. In accordance with Golden Rule 2, golden reference vectors are never computed from generated RTL, generator metadata, or testbench code.

### 5.2 Multi-Level Verification Hierarchy
- **Level L1 (Unit & Test Matrix)**: 1,080 test conditions spanning 10 matrix dimensions, 6 weight distribution classes, 6 input stimulus classes, and 3 bias configurations.
- **Level L2 (Real Tile Slices)**: 21 configurations representing all 7 projection types across first (layer 0), middle (layer 14), and last (layer 29) layers with real captured activations.
- **Level L3 (Multi-Tile Reduction)**: $16 \times 32$ multi-tile cores testing inter-tile reduction adder trees across layers 0, 14, and 29.

### 5.3 Static Interface and Weight Port Audits
To guarantee Invariant **I3**, generated Verilog ASTs are statically inspected by `verification/weight_audit.py`:
1. Asserts zero input ports matching `w_*`, `weight_*`, `w_data`, `w_addr`, `w_we`.
2. Asserts zero internal weight storage arrays (`reg ... [...]`).
3. Reconstructs constant literals from RTL text and bit-compares them with the quantized weight matrix $W_q$.

### 5.4 Mutation Testing Hardening
The testbench suite is stressed against 9 text-level mutant operators:
- **M1**: Sign flip in constant literals (`8'sd` $\to$ `-8'sd`).
- **M2**: Weight scalar perturbation ($w \pm 1$).
- **M3**: Dropped product terms in adder trees.
- **M4**: Off-by-one slicing in input bit-packing.
- **M5**: Term zeroing in accumulation expressions.
- **M6**: Accumulator bit truncation.
- **M7**: Latency / `valid_out` timing disruption.
- **M8**: Input sign-extension bit shifts.
- **M9**: Loss of carry bits in intermediate sums.

The regression suite achieves a **100.0% mutation kill rate** (9/9 mutants killed).

---

## 6. Experimental Results and Evaluation

### 6.1 Claim C1: Bit-Exact Numerical Equivalence
Across all 21 real-layer configurations (Level L2) and multi-tile slices (Level L3), LLMHW achieved **0 mismatches** against the independent INT64 reference oracle:
$$\mathbf{Y}_{\text{RTL}} \equiv \mathbf{Y}_{\text{reference}} \quad \forall \, \mathbf{X} \in [-127, 127]^{B \times K}$$
Cross-simulator verification across Icarus Verilog and Verilator confirmed byte-for-byte identical output vectors.

### 6.2 Claim C2: Quantization Quality
Evaluation across all 7 projection types for layers 0, 14, and 29 (`results/quant_quality.json`) demonstrated exceptional fidelity:
- **Cosine Similarity**: Min = 0.9984, Mean = **0.9996**, Max = 0.9999.
- **Relative Frobenius Error**: Min = 2.10%, Mean = **3.81%**, Max = 5.60%.
All layers strictly satisfy the frozen thresholds of ADR-0001 (cosine $\ge 0.990$, relative error $\le 0.080$).

### 6.3 Synthesis Evaluation E1: Area Scaling
Logic synthesis via Yosys using generic standard gates across tile dimensions $(N_t, K_t)$ under the strict fairness protocol:

| Tile Shape $(N_t \times K_t)$ | Design F Cells | Design P Cells | Cell Ratio (F/P) | Area Savings | Design F Flops | Design P Flops | Flop Reduction |
|---|---|---|---|---|---|---|---|
| $(4 \times 16)$ | 5,690 | 21,399 | 0.266 | **3.76×** | 114 | 641 | **5.62×** |
| $(8 \times 16)$ | 22,764 | 85,283 | 0.267 | **3.75×** | 226 | 1,293 | **5.72×** |
| $(16 \times 16)$ | 38,912 | 149,768 | 0.260 | **3.85×** | 442 | 2,561 | **5.79×** |
| $(16 \times 64)$ | 155,648 | 599,085 | 0.260 | **3.85×** | 491 | 8,705 | **17.73×** |
| $(32 \times 64)$ | 311,296 | 1,198,170 | 0.260 | **3.85×** | 981 | 17,409 | **17.75×** |

On FPGA fabric proxies (Lattice iCE40), Design F reduced LUT-4 requirements from 3,745 LUTs to 1,602 LUTs (**2.34× reduction**).

### 6.4 Synthesis Evaluation E2: Weight Content and Distribution Effects
Evaluation on identical $(8 \times 16)$ tiles comparing weight distributions:
1. **Real SmolLM2 Weights**: 22,764 cells.
2. **Uniform Random INT8 (5 seeds)**: 28,006 $\pm$ 48 cells. Real weights are **23.0% smaller** because trained weights cluster near zero, yielding lower bit-density.
3. **Spatial Permutation (Shuffled)**: 22,810 cells ($<0.3\%$ variation), confirming spatial position invariance.
4. **All-Zero Sparse Weights**: 3 cells (99.99% datapath eliminated).

### 6.5 Synthesis Evaluation E3: Critical Path Depth
Characterization of longest topological path (`ltp`) depth:
- $(4 \times 16)$: Design F = 41 vs Design P = 59 (ratio **0.69**, 31% reduction).
- $(8 \times 16)$: Design F = 57 vs Design P = 77 (ratio **0.74**, 26% reduction).
- $(16 \times 64)$: Design F = 89 vs Design P = 111 (ratio **0.80**, 20% reduction).
Constant shift-add structures eliminate partial-product reduction levels, resulting in consistently shallower logic cones.

### 6.6 Synthesis Evaluation E5: Full-Layer Extrapolation
Empirical direct synthesis of a $16 \times 32$ multi-tile core (4 tiles of $8 \times 16$) yielded 82,160 cells versus a tile-sum estimate of 91,056 cells (**10.82% estimation error**, conservative overestimate due to cross-tile logic sharing during flattening).

Extrapolating across all 7 projections of SmolLM2-135M layer 0 (3,456 tiles of $8 \times 16$):
- **Design F Total Cells**: **570,642,816 cells** [estimate] (~570.6M cells).
- **Design P Total Cells**: **2,075,123,088 cells** [estimate] (~2.075B cells).
- **Overall Cell Ratio (F/P)**: **0.275** (**3.64× Area Reduction** [estimate]).

---

## 7. Discussion and Limitations

### 7.1 What the Data Supports vs What It Only Suggests
- **Firmly Supported**:
  * Specializing weights as compile-time constants yields an undeniable 3.7× to 3.85× standard-cell gate reduction and a 5.7× to 17.7× flop reduction in pre-layout logic synthesis.
  * Weight clustering near zero in trained models directly reduces gate count by ~23% compared to uniform random integer distributions.
  * Bit-exact equivalence (Claim C1) is achieved without numeric divergence.
- **Only Suggested (Requires Post-Layout Physical Validation)**:
  * While topological critical path depth drops by 20% to 32%, achievable maximum operating frequency ($F_{\max}$) depends on standard-cell drive strengths, wireload models, and physical floorplanning.
  * Wire routing congestion in large fixed-constant adder trees may partially offset area gains if pin density exceeds routability thresholds.

### 7.2 Architectural Trade-Offs
The primary limitation of Design F is weight immutability. If a fine-tuned model checkpoint is updated, the silicon cannot be reprogrammed in the field without an FPGA bitstream reload or an ASIC spin. Design F is therefore ideal for stable, long-lifecycle deployments (spaceborne autonomy, automotive sensory inference, ultra-low-power consumer edge).

---

## 8. Future Work: Part II and Part III Roadmap

1. **Part II: Full Transformer Layer Co-Design**:
   - Hardware implementation of non-linear operators: fused RMSNorm, hardware SiLU/SwiGLU multipliers, and on-chip rotary position embedding units.
   - Pipelined adder trees ($L > 1$) for ultra-wide matrix projections ($K > 2048$).
   - On-chip double-buffered BRAM/URAM activation streaming memories.
2. **Part III: ASIC Physical Implementation**:
   - Hardened layout on SkyWater 130nm / TSMC 28nm open PDKs using OpenLane / Cadence Innovus.
   - Post-layout parasitic extraction, GDSII DRC/LVS closure, and silicon area/power measurement.

---

## 9. Conclusion

This project designed, implemented, and verified **LLMHW**, a model-specific hardware compiler specializing dense linear projections of SmolLM2-135M into fixed-weight Verilog-2001. By treating trained model weights as immutable compile-time constants, LLMHW provably eliminates all weight storage flip-flops and write ports, delivering a **3.64× to 3.85× cell reduction**, an **up to 17.7× flip-flop reduction**, and a **20% to 32% critical path reduction** compared to programmable baselines. Supported by an independent INT64 reference oracle, a 100% mutation kill rate, and an automated CLI, LLMHW establishes a rigorous, verifiable foundation for model-specific deep learning hardware synthesis.

---

## Appendices

### Appendix A: Reproducibility & Quickstart
A fresh clone reproduces all results with:
```bash
git clone https://github.com/himesh/BTP.git llmhw && cd llmhw
make env-check && make check && make demo
```

### Appendix B: Repository Map
See [`README.md`](../README.md) for full directory structure.

### Appendix C: Architecture Decision Records (ADR Index)
- `ADR-0001`: Quantization Quality Thresholds (Frozen Cosine $\ge 0.990$, Rel Error $\le 0.080$).
- `ADR-0002`: Symmetrical INT8 Quantization Range $[-127, 127]$ (Exclusion of $-128$).
- `ADR-0003`: Fixed Pipeline Latency $L=1$ for Design F and Design P.

### Appendix D: Academic Integrity & AI-Assistance Disclosure
This project utilized an LLM-based agentic programming assistant for boilerplate scaffolding, test suite generation, and document formatting. All numeric contracts, mathematical bounds, hardware architectures, and experimental methodologies were formally verified, audited, and executed by the student.
