# Context Comparison: Hardwired & Specialized LLM Inference Architectures

> [!NOTE]
> **Task UF.6 / Gate HF Deliverable**  
> All figures presented for our architecture are pre-layout engineering estimates based on synthesized RTL and TSMC 28nm HPC+ cell models, not physical silicon measurements. All vendor claims and independent benchmarks are sourced and cited with explicit architectural caveats.

---

## 1. Executive Comparison Matrix

The table below contrasts our hardwired SmolLM2-135M ROM-based accelerator against published dedicated AI/LLM inference architectures across memory hierarchy, weight mutability, reported throughput, and deployment characteristics.

| Architecture | Model Target | Tech Node | Weight Memory Hierarchy | Weight Mutability | Reported Throughput (tok/s) | Source & Status | Key Architectural Caveats |
|---|---|---|---|---|---|---|---|
| **This Work (SmolLM2-135M ROM Accelerator)** | SmolLM2-135M (INT8) | TSMC 28nm HPC+ | **100% On-Chip Mask ROM** (157.5 MB); 2.97 MB SRAM for KV cache | **Immutable** (hardwired in silicon) | **28.37** (baseline @ 400 MHz, 128 lanes)<br>**63.37** (512 lanes @ 400 MHz) *(estimate)* | Synthesized Verilog (Yosys) + TSMC 28nm standard cell / memory compiler models | Zero external DRAM required; model weights cannot be updated without a new mask set; single-stream decode. |
| **Groq LPU (Language Processing Unit)** | Llama-3-8B / Llama-2-70B | GlobalFoundries 14nm | **100% On-Chip SRAM** (230 MB per chip) | Mutable (loaded into SRAM at boot) | **~500–1,250** (Llama-3-8B across 8–576 chip pod) | Artificial Analysis (2024 independent benchmark) & Groq Whitepapers (2020, 2023) | Requires massive multi-chip interconnect pods (tens to hundreds of chips) to fit model weights into SRAM; high power (~300W per card). |
| **Etched Sohu** | Transformer-only (Llama-3-70B/405B) | TSMC 4nm | **External HBM3e** (up to 144 GB per board) | Mutable (streamed from HBM) | **~500,000** (8-Sohu node on 70B) *(vendor claim)* | Etched Technical Whitepaper (2024; vendor claim, pre-silicon) | Hardwired Transformer dataflow, but weights reside in external high-bandwidth memory; not single-chip autonomous; no independent physical measurements. |
| **Tenstorrent Wormhole / Blackhole** | General LLM / CNN | TSMC 12nm / 6nm | **SRAM + GDDR6 / HBM** (Tensix tile array) | Mutable | **~25–60** (single-device small LLM) | Tenstorrent Microarchitecture Manuals & Developer Benchmarks (2023–2024) | General-purpose spatial grid; packet-switched Network-on-Chip (NoC) introduces dynamic routing latency vs statically scheduled ROM pipeline. |
| **SambaNova SN40L** | Enterprise LLMs | TSMC 5nm | **3-Tier Memory**: On-chip SRAM + On-package HBM + DDR5 | Mutable | **~100–300** (8B models) | SambaNova Dataflow Architecture Brief (2024) | Reconfigurable dataflow architecture requires complex compiler mapping; high thermal dissipation (700W+ per socket). |
| **Apple Neural Engine (A17 Pro / M4)** | Mobile LLMs (3B–7B quantized) | TSMC 3nm | **Unified LPDDR5X System DRAM** (on-package) | Mutable | **~20–35** (3B 4-bit) | MLX Community Benchmarks & Geekbench AI (2024 independent) | Memory-bandwidth limited during autoregressive token decode (~100–150 GB/s bus shared with CPU/GPU); non-dedicated. |

---

## 2. In-Depth Architectural Contrast

### 2.1 Weight Storage: Mask ROM vs SRAM vs HBM
- **Mask ROM (This Work):** By hardwiring all 134.48M INT8 weights and requantization parameters directly into metal/via diffusion programmable ROM, cell area drops to $\sim 0.08\,\mu\text{m}^2/\text{bit}$ in 28nm—approximately **$2\times$ denser than 6T SRAM** and eliminating all DRAM interfaces, PHYs, memory controllers, and refresh circuitry. The entire 135M parameter model fits in $148\,\text{mm}^2$ of standard silicon.
- **SRAM-Only (Groq):** Groq stores weights in high-speed SRAM ($0.15\,\mu\text{m}^2/\text{bit}$ in 14nm). While achieving single-cycle access and determinism, SRAM density limits each chip to 230 MB. To run an 8B model requires a rack of dozens of chips interconnected with custom cables, yielding extreme cost and power.
- **External HBM/DRAM (Etched, Tenstorrent, Apple):** Traditional accelerators stream weights from external memory on every token generation step. Even with 1 TB/s HBM, memory bus energy ($\sim 5\text{–}10\,\text{pJ/bit}$) dwarfs arithmetic computation ($\sim 0.2\,\text{pJ/MAC}$ in INT8). Our architecture performs zero external memory transactions per token.

### 2.2 Control and Scheduling: Static Sequencer vs Dynamic Dispatch
- Modern GPUs and NPUs employ dynamic out-of-order schedulers, register renaming, and hardware warp management.
- Our architecture utilizes a **cycle-exact, compile-time static sequencer** (`rtl/lib/sequencer.v`). Instruction decode is zero-overhead microcode; tile streaming and memory bank addresses are deterministically computed by address generators (`kv_addr_gen`, `embed_gather`). This yields **0% scheduling jitter** and enables exact cycle matching between simulation and silicon.

### 2.3 Arithmetic Specialization vs Flexibility
- **General Accelerators (Apple, Tenstorrent):** Must support arbitrary activation shapes, dynamic padding, convolutions, and arbitrary precision (FP32/FP16/BF16/INT8/INT4).
- **This Work:** Hardware is specialized strictly to SmolLM2-135M dimensions ($d=576$, $d_{\text{mlp}}=1536$, 9 Q heads, 3 KV heads, $d_{\text{head}}=64$). Requantization uses frozen dyadic integer shifts (`requant.v`), non-linear activations use dedicated piecewise linear tables (`swiglu.v`, `rmsnorm.v`, `softmax.v`), and head rotation is hardwired (`rope.v`).

---

## 3. Limits, Caveats, and Honest Assessment

1. **Model Immutability:** The greatest advantage of this architecture is also its greatest limitation. Weights and microcode are etched into ROM. Updating model weights requires a full tape-out mask revision. This design is appropriate for mass-market embedded applications (e.g. voice assistants, edge robotics, consumer appliances) where a model is certified and frozen for millions of units.
2. **Context Length Ceiling:** The KV cache SRAM is sized for $M=256$ tokens ($2.81\,\text{MB}$). Scaling context to $2048$ or $4096$ tokens would increase SRAM from $2.97\,\text{MB}$ to $>24\,\text{MB}$, shifting the die area balance.
3. **Pre-Layout Estimations:** The reported $28.37\text{--}63.37\,\text{tok/s}$ throughput and $147.95\,\text{mm}^2$ die area are analytical projections from synthesized netlists and literature cell constants. Physical silicon will encounter real-world parasitic clock skew, IR drop, and packaging wirebond/bump overheads accounted for by our $1.35\times$ floorplan margin.
