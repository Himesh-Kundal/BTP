# LLMHW: Model-Specific Dense Linear Layer Hardware Compiler

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Verilog: 2001](https://img.shields.io/badge/Verilog-2001-green.svg)](docs/02_architecture_and_contracts.md)
[![Simulation: Icarus%20%2B%20Verilator](https://img.shields.io/badge/Simulation-Icarus%20%2B%20Verilator-purple.svg)](verification/)
[![Synthesis: Yosys](https://img.shields.io/badge/Synthesis-Yosys-orange.svg)](synth/)
[![Tests: 226%20Passed](https://img.shields.io/badge/Tests-226%20Passed-brightgreen.svg)](tests/)

**LLMHW** is an end-to-end model-specific hardware compiler that translates dense linear projections of modern transformer architectures (**SmolLM2-135M**) into synthesizable, constant-weight Verilog-2001 (Design F). 

By specializing trained weights directly as compile-time numerical constants into the datapath arithmetic, **LLMHW** completely eliminates runtime weight input ports, memory write buses, and weight register storage (Invariant I3). The resulting designs achieve a **3.0× to 3.85× standard-cell reduction**, an **up to 17.7× flip-flop reduction**, and a **20% to 32% critical path shortening** compared to programmable baselines under strict fairness protocols.

All generated circuits are validated **bit-exact** against an independent NumPy INT64 reference oracle (**Claim C1**), withstand a **100% mutation kill rate** (Mutants M1–M9), and pass dual-simulator cross-checks across Icarus Verilog and Verilator.

---

## Key Highlights & Core Claims

| Feature / Metric | Design F (Specialized Fixed-Weight) | Design P (Programmable Baseline) | Impact / Evidence |
|---|---|---|---|
| **Weight Ports (I3)** | **0 bits** (Zero weight inputs) | $8 \times N \times K$ bits ($W_{\text{reg}}$ array) | No memory write logic, no bus routing |
| **Logic Cell Count** | **155,648 cells** ($16 \times 64$) | **599,085 cells** ($16 \times 64$) | **3.85× Area Reduction** |
| **Sequential Flops** | **491 DFFs** (pruned dynamic range) | **8,705 DFFs** ($W_{\text{reg}} + \text{acc}$) | **17.7× Storage Elimination** |
| **Topological Depth** | **57 levels** ($8 \times 16$) | **77 levels** ($8 \times 16$) | **26% Shorter Critical Path** |
| **Claim C1 (Equivalence)** | **0 mismatches** across all 7 projections | Reference-equivalent | Proven bit-exact against NumPy INT64 |
| **Mutation Hardening** | **100% killed** (9/9 mutant operators) | Checked on every test | Zero survivability of datapath bugs |
| **Generation Speed** | **0.54 seconds** for full $576 \times 576$ layer | N/A | Highly scalable parametric generator |

---

## System Architecture

LLMHW enforces strict architectural layering (Golden Rule 5): model-specific inspection is confined to `frontend/`, while `ir/`, `compiler/`, `backend/`, and `verification/` are strictly model-agnostic and never import PyTorch or hardcode transformer dimensions.

```
┌────────────────────────────────────────────────────────┐
│                      FRONTEND                          │
│   HuggingFace Transformer (SmolLM2-135M @ 93efa2f)     │
│   • Linear Extractor: 210 projections (7 types)        │
│   • Activation Capture: Forward hooks, fixed seed 42   │
└──────────────────────────┬─────────────────────────────┘
                           │ weight.npy, x_sample.npy
┌──────────────────────────▼─────────────────────────────┐
│                 INTERMEDIATE REPRESENTATION             │
│   Lossless AST: LinearOp, TensorRef, TileManifest      │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│                        COMPILER                        │
│   • Rule N2: Per-channel symmetric INT8 weights [-127,127]
│   • Rule N3: Static per-tensor calibration (absmax)    │
│   • Rule N4: Worst-case 32-bit accumulator bound check │
│   • Specialization: CONSTANT kind, strip write paths   │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
┌────────────────────────┐  ┌────────────────────────┐
│   BACKEND DESIGN F     │  │   BACKEND DESIGN P     │
│  (Fixed-Weight RTL)    │  │ (Programmable Baseline)│
│  • Parametric Tiling   │  │  • W_reg Array Storage │
│  • Sparsity Zero-Skip  │  │  • Runtime Multipliers │
│  • Adder Reduction     │  │  • Weight-load Phase   │
│  • Latency L=1 Wrapper │  │  • Identical Widths    │
└────────────┬───────────┘  └────────────┬───────────┘
             │                           │
             └─────────────┬─────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│                      VERIFICATION                       │
│   • Independent Oracle: NumPy INT64 (N1–N7)            │
│   • Dual Simulator: Icarus Verilog (vvp) & Verilator   │
│   • Static Weight Audit: AST parser verifies literals  │
│   • Mutation Testing: 100% kill rate (M1–M9)           │
│   • Logic Synthesis: Yosys Generic Gates & iCE40 Proxy │
└────────────────────────────────────────────────────────┘
```

---

## Toolchain & Requirements

LLMHW runs on Linux (x86_64 / aarch64) with the following open-source toolchain:
- **Python**: $\ge 3.10$ managed via [`uv`](https://github.com/astral-sh/uv)
- **HDL Simulators**:
  - [Icarus Verilog](http://iverilog.icarus.com/) (`iverilog`, `vvp` $\ge 11.0$)
  - [Verilator](https://www.veripool.org/verilator/) ($\ge 5.0$)
- **Logic Synthesis**:
  - [Yosys](https://yosyshq.net/yosys/) ($\ge 0.33$)

Verify your environment in one command:
```bash
make env-check
```

---

## Quickstart

### 1. Installation
Using `uv`:
```bash
git clone https://github.com/himesh/BTP.git llmhw
cd llmhw
uv sync
```

### 2. Verify Repository Health
Run linting and the entire 226-test suite:
```bash
make check
```

### 3. Flagship Multi-Layer Demonstration
Execute compilation, static weight audit, and bit-exact simulation across all 7 projection types of SmolLM2-135M layer 0:
```bash
make demo
```
*(Runs in under 4 seconds and outputs a formatted verification matrix).*

### 4. Full Verification Regression Matrix
Run the comprehensive multi-dimensional simulation matrix across dimensions, tile geometries, weight distributions, and input classes:
```bash
make regression
```

### 5. Reproduce Synthesis Experiments & Figures
Synthesize tile configurations and regenerate publication-quality figures:
```bash
make synth
make report-figs
```

---

## Unified Command-Line Interface (`llmhw`)

The compiler provides a unified CLI executable `./llmhw` (and `uv run llmhw`):

### Extract Model Weights & Activations
```bash
./llmhw extract --model HuggingFaceTB/SmolLM2-135M --layer 0 --proj q_proj --out artifacts/
```

### Compile to Fixed-Weight Verilog (Design F)
```bash
./llmhw compile --layer 0 --arch q_proj --slice-n 8 --slice-k 16 --design F --out build/rtl/
```

### Compile to Programmable Baseline (Design P)
```bash
./llmhw compile --layer 0 --arch q_proj --slice-n 8 --slice-k 16 --design P --out build/rtl/
```

### Verify RTL Against Independent Reference Oracle
```bash
./llmhw verify --rtl build/rtl/llmhw_l0_q_proj_s8x16.v --layer 0 --arch q_proj --sim iverilog --num-vectors 20
```

### Run Logic Synthesis via Yosys
```bash
./llmhw synth --rtl build/rtl/llmhw_l0_q_proj_s8x16.v --target generic --out results/my_synth.json
```

---

## Synthesis Evaluation Results

Experiments conducted under strict fairness rules (identical bitwidths, latency $L=1$, flattening enabled):

| Experiment | Focus | Key Findings |
|---|---|---|
| **E1 (Area Scaling)** | Tile grid scaling $(4,16)$ to $(32,64)$ | Design F is **3.0× to 3.85× smaller** than Design P; flop count drops by **17.7×**. |
| **E2 (Weight Effects)** | Real vs Uniform INT8 vs Sparse | Real LLM weights are **~23% smaller** than uniform random INT8 due to clustering near 0. Zero-skipping prunes 99.99% of dead terms. |
| **E3 (Critical Path)** | Topological path depth (`ltp`) | Constant shift-add shortens longest paths by **20% to 32%** (ratio 0.68 to 0.74). |
| **E5 (Full-Layer Extrapolation)** | SmolLM2 Layer 0 (3,456 tiles) | Layer 0 is estimated at **570.6M cells** for F vs **2.075B cells** for P (**3.64× reduction**). Direct multi-tile synthesis validates model within 10.8%. |

Generated publication vector plots are available under [`results/figures/`](results/figures/):
- `fig1_area_scaling.svg` / `.png`: Standard cell count vs tile dimensions.
- `fig2_weight_distribution.svg` / `.png`: Impact of trained weight distributions.
- `fig3_critical_path_depth.svg` / `.png`: Critical path depth across tile grids.
- `fig4_full_layer_extrapolation.svg` / `.png`: Multi-tile extrapolation vs direct measurements.

---

## Repository Map

```
llmhw/
├── frontend/       # Pinned model loader, layer extractor, activation capture (PyTorch)
├── ir/             # Model-agnostic IR dataclasses, graph, and tensor references
├── compiler/       # Symmetric INT8 quantization, calibration, and accumulator bounds
├── backend/        # Verilog-2001 codegen, tiling, Design F, and Baseline Design P
├── verification/   # Independent NumPy INT64 oracle, simulation runner, mutation engine
├── synth/          # Automated Yosys runner, fairness auditor, and experiment drivers
├── cli.py          # Unified CLI entry point (`llmhw`)
├── llmhw           # Executable wrapper script
├── Makefile        # Standardized developer targets (env-check, check, demo, synth)
├── scripts/        # Demonstration drivers and figure generation pipelines
├── tests/          # Comprehensive 226-test suite with 100% pass rate
├── docs/           # Specifications, architectural contracts, and walk-throughs
└── results/        # Experimental CSVs, JSON metrics, figures, and gate reports
```

---

## Limitations & Non-Goals

See [`docs/limitations.md`](docs/limitations.md) for full formal scope boundaries.
- **Part I Scope**: Focuses strictly on dense linear projections. Non-linear operations (RMSNorm, Softmax, RoPE, SiLU) are evaluated in host software.
- **Immutable Weights**: Design F weights are hardcoded at compile time. Reprogramming requires bitstream regeneration.
- **Pre-Layout Synthesis**: Metrics reflect standard-cell logic complexity and topological depths. Physical routing, parasitics, and clock tree synthesis depend on target ASIC PDKs.

---

## Academic Integrity & AI-Assistance Disclosure

In accordance with institutional academic integrity policies, this project utilized LLM-based agentic assistance for code scaffolding, automated refactoring, and test generation under direct human architectural specification and review. All mathematical proofs, numeric contracts (Rules N1–N10), architectural invariants (I1–I5), and experimental evaluation protocols were formally audited by the student.
