# LLMHW System Limitations and Architectural Boundaries

This document formalizes the explicit boundaries, assumptions, and non-goals of the LLMHW hardware compiler for Part I of the project.

---

## 1. Scope Boundaries (Part I)

### 1.1 Linear Layer Focus
- **Included**: All dense projection operations in transformer blocks:
  - Multi-Head Attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
  - Feed-Forward / SwiGLU projections: `gate_proj`, `up_proj`, `down_proj`.
- **Excluded**:
  - Non-linear attention mechanisms: Softmax, Scaled Dot-Product Attention score matrix ($Q K^T$), and attention dropout.
  - Normalization operators: RMSNorm, LayerNorm.
  - Position embeddings: Rotary Positional Embeddings (RoPE).
  - Non-linear activation functions: SiLU, GeLU, ReLU elementwise units.
  *(These operations remain in host software or are scheduled for Part II full-pipeline integration).*

### 1.2 Constant-Weight Specialization
- **Model-Specific Hardware (Design F)**: Weights are hardcoded as compile-time numerical constants (`8'sd...`) directly within standard-cell shift-add logic and multiplier nets.
- **No Runtime Weight Reloading**: Design F features **zero** weight input ports, **zero** write-enable strobes, **zero** address buses, and **zero** weight storage flip-flops or SRAM macros (Invariant I3).
- **Update Model**: Changing model weights requires regenerating the Verilog source code and performing a new FPGA bitstream generation or ASIC fabrication run.

### 1.3 Numeric Range and Quantization Contract
- **INT8 Arithmetic**: Per-channel symmetric INT8 weights ($W_q \in [-127, 127]$, Rule N2) and static per-tensor symmetric INT8 activations ($X_q \in [-127, 127]$, Rule N3). The value $-128$ is explicitly unused to guarantee symmetric dynamic range without two's complement sign-flip overflow.
- **32-Bit Accumulation**: Intermediate row products ($16$-bit signed) accumulate into $32$-bit signed registers (Rule N4, N8). Worst-case dynamic range is strictly bounded: $K \times 127 \times 127 + |b_q|_{\max} < 2^{31}$.
- **Dequantization**: Hardware cores emit raw signed 32-bit accumulators. Rescaling by $s_x \cdot s_w$ and conversion back to floating-point representation is performed by downstream host processing in Part I.

---

## 2. Synthesis and Physical Design Caveats

### 2.1 Pre-Layout Logic Synthesis
- Synthesis evaluation is performed using **Yosys Open SYnthesis Suite** mapping to:
  1. Generic standard gate library (`synth -top <top>; stat`).
  2. Lattice iCE40 FPGA LUT-4 cell proxy (`synth_ice40 -top <top>; stat`).
- **Caveat**: Reported metrics represent pre-layout logic complexity (cell counts, combinational gate equivalents, flip-flop registers, LUT-4 counts, and topological critical path depth `ltp`).
- **Post-P&R Factors**: Interconnect delay, routing congestion, parasitic capacitance, clock tree insertion, voltage drop, and thermal density must be validated with target ASIC PDKs (e.g., SkyWater 130nm, TSMC 28nm) or production FPGA tools (Vivado, Quartus).

### 2.2 Critical Path Proxy
- Timing is characterized via Yosys Longest Topological Path (`ltp`) depth. While topological depth strongly correlates with propagation delay in combinational networks, exact operational clock frequency ($F_{\max}$) depends on specific library cell drive strengths, wireload models, and physical floorplanning.

---

## 3. Extensibility and Roadmap to Part II

The modular architectural separation between `frontend/`, `ir/`, `compiler/`, `backend/`, and `verification/` enables seamless extensions in subsequent project phases:
1. **Pipelined Reduction Trees**: Multi-cycle pipelined adder trees for ultra-wide inner dimensions ($K > 2048$).
2. **On-Chip Activation Buffers**: Ping-pong double-buffered BRAM/URAM activation memories.
3. **Full Transformer Layer Integration**: Integrating fixed-weight matrix multiplication cores with dedicated fused SiLU and RMSNorm hardware accelerators.
