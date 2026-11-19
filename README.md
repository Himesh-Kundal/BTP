# LLMHW: Model-Specific Transformer Decoder & Linear Hardware Compiler (Rust Edition)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust: 1.80+](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![Verilog: 2001/2005](https://img.shields.io/badge/Verilog-2001%2F2005-green.svg)](docs/part2/02_architecture_and_contracts.md)
[![Simulation: Icarus%20%2B%20Verilator](https://img.shields.io/badge/Simulation-Icarus%20%2B%20Verilator-purple.svg)](crates/llmhw-sim/)
[![Synthesis: Yosys](https://img.shields.io/badge/Synthesis-Yosys-orange.svg)](crates/llmhw-synth/)
[![Tests: 100%20Passed](https://img.shields.io/badge/Tests-100%25%20Passed-brightgreen.svg)](crates/)

**LLMHW** is an end-to-end model-specific hardware compiler implemented in **100% pure native Rust** that compiles **SmolLM2-135M** ($134.5\text{M}$ parameters, 30 transformer layers, GQA 9:3 attention, SwiGLU MLP, $49,152$ vocabulary) into synthesizable, pure-integer Verilog-2001/2005 hardware:

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

### 2. Fast Test Suite & Strict Lint (100% Native Rust)
```bash
make check      # cargo clippy strict lint + cargo test --release across all 10 crates
make test       # Run all unit and integration tests in release mode
```

### 3. Part II Turnkey Demonstrations
- **Fast Tiny-Model End-to-End RTL Demo (~5 seconds):**
  ```bash
  make demo2-fast
  ```
  Compiles the 2-layer tiny decoder to Verilog, runs Verilator RTL simulation against the integer interpreter, verifies all 4 structural/ROM audits, and confirms 100% bit-exact tokens.

- **Full 30-Layer SmolLM2-135M Verilator RTL Execution Demo:**
  ```bash
  make demo2
  ```
  Runs prompt `"The capital of France is Paris. The capital of Germany is"` for 4 new tokens through the 30-layer SmolLM2-135M Verilog accelerator (`top_accel.v`) in Verilator, verifying **100% bit-exact token equality** with the integer reference oracle and **4/4 greedy match** (`' Berlin. The capital'`) with PyTorch FP32.

- **Speed & Parity Benchmark (Rust vs Historical Python Oracle):**
  ```bash
  make bench
  ```
  Verifies 100% SHA-256 byte-for-byte match on ROM hex files and full 30-layer decode speedup (**0.293s in Rust vs 4.60s in Python, 15.7× faster**).

---

## Unified Command-Line Interface (`llmhw`)

The compiler provides a unified binary (`cargo run --release --bin llmhw -- <command>` or `target/release/llmhw`):

```bash
# 1. Interactive prompt generation in pure Rust (~0.8s software interpreter)
cargo run --release --bin llmhw -- run \
  --bundle build/smollm2_hw \
  --prompt "Once upon a time" \
  --max-new-tokens 8 \
  --backend interpreter

# 2. End-to-end Verilator RTL hardware accelerator simulation
cargo run --release --bin llmhw -- run \
  --bundle build/smollm2_hw \
  --prompt "The capital of France is" \
  --max-new-tokens 4 \
  --backend both

# 3. Compile full SmolLM2-135M decoder or tiny model to ROM-backed Verilog bundle
cargo run --release --bin llmhw -- compile-model --tiny --out build/tiny_hw
cargo run --release --bin llmhw -- compile-model --out build/smollm2_hw

# 4. Verify compiled hardware bundle (ROM immutability + structural audit + RTL parity)
cargo run --release --bin llmhw -- verify --bundle build/smollm2_hw --tokens "1,10,25" --max-new-tokens 2

# 5. Synthesize any Part II library unit or Part I linear module via Yosys
cargo run --release --bin llmhw -- synth --unit rmsnorm --target generic --out results/part2/cli_synth_rmsnorm.json
cargo run --release --bin llmhw -- synth --rtl build/rtl/llmhw_l0_q_proj_s8x16.v --target generic

# 6. Run full differential equivalence and performance benchmark
cargo run --release --bin llmhw -- differential-bench
```

---

## Workspace Crates Architecture

The codebase is organized as a modular 10-crate Cargo workspace in `crates/`:

| Crate | Purpose |
|---|---|
| [`llmhw-numerics`](crates/llmhw-numerics/) | Pure integer arithmetic oracle, Softmax exp2 fraction LUT, RMSNorm rsqrt LUT, SwiGLU, and 30-layer decoder |
| [`llmhw-ir`](crates/llmhw-ir/) | GraphV2 intermediate representation, LinearOp, Tensor, serialization |
| [`llmhw-compiler`](crates/llmhw-compiler/) | Memory planner, format assignment, ROM image packing, microprogram scheduler, and interpreter |
| [`llmhw-backend`](crates/llmhw-backend/) | Synthesizable Verilog codegen for Design F, P, top accelerator, ROM generator, and testbenches |
| [`llmhw-sim`](crates/llmhw-sim/) | Verilator and Icarus Verilog simulation runners and trace extractors |
| [`llmhw-verify`](crates/llmhw-verify/) | ROM immutability audit, structural no-oracle audit, and regression runner |
| [`llmhw-synth`](crates/llmhw-synth/) | Yosys synthesis execution and cell/flop/depth accounting |
| [`llmhw-frontend`](crates/llmhw-frontend/) | SmoothQuant per-channel migration, artifact loader, and tiny model fixture generator |
| [`llmhw-eval`](crates/llmhw-eval/) | Perplexity, KL divergence, and Top-1 accuracy evaluation |
| [`llmhw-cli`](crates/llmhw-cli/) | Unified command-line interface executable with native Hugging Face tokenization |

---

## Academic Integrity & AI-Assistance Disclosure

In accordance with institutional academic integrity policies, this project utilized LLM-based agentic assistance (Google DeepMind Antigravity) for code scaffolding, refactoring, testbench automation, and report drafting under direct human architectural specification and review. All numeric contracts, hardware interfaces, ADRs, and experimental results were verified against executable scripts and audited by the student. See [`docs/whole_project_report.md`](docs/whole_project_report.md) Appendix D for the complete disclosure.
