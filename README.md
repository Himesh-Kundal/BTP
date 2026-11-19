# LLMHW: Model-Specific Transformer Decoder & Linear Hardware Compiler (Part I & Part II)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Verilog: 2001/2005](https://img.shields.io/badge/Verilog-2001%2F2005-green.svg)](docs/part2/02_architecture_and_contracts.md)
[![Simulation: Icarus%20%2B%20Verilator](https://img.shields.io/badge/Simulation-Icarus%20%2B%20Verilator-purple.svg)](verification/)
[![Synthesis: Yosys](https://img.shields.io/badge/Synthesis-Yosys-orange.svg)](synth/)
[![Tests: 168%20Part%20II%20Passed](https://img.shields.io/badge/Tests-100%25%20Passed-brightgreen.svg)](tests/)

**LLMHW** is an end-to-end model-specific hardware compiler that compiles **SmolLM2-135M** ($134.5\text{M}$ parameters, 30 transformer layers, GQA 9:3 attention, SwiGLU MLP, $49,152$ vocabulary) into synthesizable, pure-integer Verilog-2001/2005 hardware:

- **Part I (Spatial Constant-Weight Linear Layer Compiler):** Compiles dense linear projections directly into unrolled shift-add constant-multiplier trees (Design F), eliminating runtime weight ports and achieving a **3.0× to 3.85× standard-cell reduction**, up to **17.7× flip-flop reduction**, and **20% to 32% critical path shortening** over programmable baselines (Design P).
- **Part II (Complete Autoregressive Transformer Decoder Accelerator):** Extends the compiler to the **entire 30-layer SmolLM2-135M decoder** (embedding gather, RMSNorm, Q1.14 RoPE, GQA causal attention with on-chip KV cache, SwiGLU, 24-bit residual stream, tied LM head, and scale-aligned greedy ArgMax) backed by immutable synchronous ROMs and an $R \times C = 8 \times 16$ streaming MAC engine driven by a deterministic microprogram sequencer.

Every generated circuit is verified **100% bit-exact** against an independent pure-integer reference oracle (**Claim C1 / C1′**), achieves a **100% mutation kill rate** across 9 Part I mutants (`M1`–`M9`), 21 Part II unit mutants (`M10`–`M30`), and 7 top-level system mutants (`TM1`–`TM7`), and retains **70.16% Top-1 ArgMax agreement** ($\Delta \text{PPL} = +17.33$ on WikiText-2) with PyTorch FP32 (**Claim C2′**).

---

## Key Highlights & Verified Claims (Part I & Part II)

| Claim / Study | Metric | Measured / Verified Value | Source File |
|---|---|---|---|
| **C1 (Part I Linear Parity)** | Bit-exact linear layer equivalence vs NumPy INT64 | **0 mismatches** across 210 projections & 135 regression configs | `results/regression_summary.json` |
| **Part I Spatial Area** | Design F vs Design P cell count ($16 \times 64$ tile) | **155,648 vs 599,085 cells (3.85× reduction)**; **17.7× fewer DFFs** | `results/E1.csv` |
| **C1′ (Part II Full Decoder Parity)** | 30-layer SmolLM2-135M RTL vs Integer Oracle | **10/10 prompts × 16 tokens 100% token-exact** (285 steps; 2,279,424 trace elements exact) | `results/part2/full_decode.json` |
| **C1′ Long-Context Parity** | Full 30-layer decode at `MAX_CTX = 256` | **256/256 positions token-exact & trace-exact** (4.67B cycles) | `results/part2/ue5_long_context.json` |
| **C2′ (Quantization Quality)** | Pure INT8 W8A8/INT24 vs PyTorch FP32 (WikiText-2) | **Top-1 Agreement = 70.16%** ($\ge 35\%$); **$\Delta \text{PPL} = +17.3282$** ($89.67 \to 107.00 \le 25.0$); **KL = 0.2280** | `results/part2/quality_final.json` |
| **Mutation Hardening** | Fault injection kill rate | **100.0%** (`9/9` Part I, `21/21` Part II units, `7/7` system mutants) | `results/mutation_matrix_v2.json` |
| **EA1–EA3 (Hardware Cost)** | Full accelerator logic vs on-chip memory (`MAX_CTX=256`) | **78,510 generic cells** (`3,661` DFFs); **157.54 MB ROM (98.15%)** + **2.97 MB SRAM (1.85%)** | `results/part2/EA1.csv`, `EA3.json` |
| **EA2 (Cycle Model)** | Analytical cycle model vs Verilator simulation | **13,691,648 vs 14,100,262 cycles/token (-2.90% error)** | `results/part2/EA2.json` |
| **EA4 (Spatial vs ROM)** | Flat spatial vs ROM-sequential ($32 \times 32$ block) | **213,022 vs 21,387 generic cells (9.96×)**; full spatial extrapolates to **27.98B gates** | `results/part2/EA4.json` |
| **EA5 (28nm ASIC Estimate)** | Pre-layout TSMC 28nm HPC+ @ 400 MHz (128 lanes) | **147.95 mm² die area** (Mid estimate), **28.37 tokens/s** ($63.37\text{ tok/s}$ @ 512 lanes) | `results/part2/EA5.json` |

---

## Part II Full-Decoder Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     FRONTEND & OFFLINE SMOOTHQUANT (ADR-2-2)                 │
│  HuggingFace SmolLM2-135M @ 93efa2f097d58c2a74874c7e644dbc9b0cee75a2         │
│  • Per-channel SmoothQuant migration: γ ↔ W_{q,k,v}, γ ↔ W_{gate,up},        │
│    RoPE-pair W_q ↔ W_k, outlier migration (W_{gate}, W_{up}) ↔ W_{down}      │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │ v2 artifacts + calib_scales.json
┌──────────────────────────────────────▼───────────────────────────────────────┐
│                   GRAPH IR v2 & COMPILER PASSES (NO TORCH)                   │
│  • GraphV2 (I1–I7 invariants) → Format & Dyadic (M, s) Assignment            │
│  • Immutable ROM Packing (weights_128b.hex, requant_params.hex, tables*.hex) │
│  • Microprogram Schedule (schedule.json, program.hex) & Memory Planner       │
│  • CompiledArtifactInterpreter (Independent Bit-Exact Software Oracle)       │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────────┐
│                    GENERATED VERILOG ACCELERATOR (top_accel.v)               │
│  ┌───────────────────────┐   ┌────────────────────────────────────────────┐  │
│  │ Microprogram Sequencer├───► Immutable Synchronous ROMs (rom_sync.v)    │  │
│  │     (sequencer.v)     │   │ • Weights (155.25 MB) • Params (2.20 MB)   │  │
│  └───────────┬───────────┘   │ • RoPE / rsqrt / exp2 / SiLU LUTs          │  │
│              │               └────────────────────────────────────────────┘  │
│              ▼                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ Streaming Datapath Units (rtl/lib/)                                    │  │
│  │ • embed_gather.v  • rmsnorm.v (48b acc)  • linear_engine.v (8x16 MACs) │  │
│  │ • rope.v (Q1.14)  • attention.v + softmax.v + kv_cache.v (2.81 MB)     │  │
│  │ • swiglu.v        • residual_add.v (INT24) • argmax.v (49,152 vocab)   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

Detailed architectural specifications and contracts:
- [`docs/part2/01_problem_and_scope.md`](docs/part2/01_problem_and_scope.md)
- [`docs/part2/02_architecture_and_contracts.md`](docs/part2/02_architecture_and_contracts.md)
- [`docs/part2/context_comparison.md`](docs/part2/context_comparison.md)
- [`docs/limitations.md`](docs/limitations.md)
- [`docs/whole_project_report.md`](docs/whole_project_report.md) & [`docs/report_traceability.md`](docs/report_traceability.md)

---

## Quickstart & Turnkey Reproduction

### 1. Environment Check
```bash
make env-check
```

### 2. Fast Test Suites (Part I & Part II)
```bash
make check    # Full test suite (Part I + Part II)
make check2   # Part II lint + 168 unit, RTL, mutation, audit, and CLI tests (~2 min)
```

### 3. Part II Demonstrations (`make demo2-fast` & `make demo2`)
- **Fast Tiny-Model End-to-End RTL & Mutation Demo (~11.5 seconds):**
  ```bash
  make demo2-fast
  ```
  Compiles the 2-layer tiny decoder to Verilog, runs Verilator RTL simulation against the integer interpreter (`115,227` cycles, 100% bit-exact), verifies all 4 structural/ROM audits, and reports the unit (`21/21`) and system (`7/7`) mutation kill matrices.

- **Full 30-Layer SmolLM2-135M Verilator RTL Generation Demo (~106 seconds):**
  ```bash
  make demo2
  ```
  Runs prompt `"The capital of France is Paris. The capital of Germany is"` for 4 new tokens through the 30-layer SmolLM2-135M Verilog accelerator (`top_accel.v`) in Verilator (`221,848,192` cycles), verifying **100% bit-exact token equality** with the integer reference oracle and **4/4 greedy match** (`' Berlin. The capital'`) with PyTorch FP32.

---

## Unified Command-Line Interface (`llmhw` v2)

```bash
# 1. Compile tiny or full SmolLM2-135M decoder to a ROM-backed Verilog bundle
uv run python cli.py compile-model --tiny --out build/tiny_hw
uv run python cli.py compile-model --model HuggingFaceTB/SmolLM2-135M --out build/smollm2_hw

# 2. Run autoregressive token generation (interpreter, rtl, or both)
uv run python cli.py run --bundle build/tiny_hw --tokens "1,10,25" --max-new-tokens 4 --backend both
uv run python cli.py run --bundle build/smollm2_hw --prompt "Once upon a time" --max-new-tokens 6 --backend interpreter

# 3. Verify compiled v2 bundle (ROM immutability + no-oracle audit + Verilator RTL parity)
uv run python cli.py verify --bundle build/tiny_hw --tokens "1,10,25" --max-new-tokens 4 --out results/verify_v2.json

# 4. Synthesize any Part II RTL unit or Part I linear module via Yosys
uv run python cli.py synth --unit rmsnorm --target generic --out results/synth_rmsnorm.json
uv run python cli.py synth --rtl build/rtl/llmhw_l0_q_proj_s8x16.v --target generic
```

---

## Academic Integrity & AI-Assistance Disclosure

In accordance with institutional academic integrity policies, this project utilized LLM-based agentic assistance (Google DeepMind Antigravity) for code scaffolding, refactoring, testbench automation, and report drafting under direct human architectural specification and review. All numeric contracts, hardware interfaces, ADRs, and experimental results were verified against executable scripts and audited by the student. See [`docs/whole_project_report.md`](docs/whole_project_report.md) Appendix D for the complete disclosure.
