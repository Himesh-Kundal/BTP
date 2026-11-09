# Part II / 01 — Problem Statement, Scope and Requirements

## 1. Where Part I stopped
Part I delivered a compiler for **dense linear projections**: extract from SmolLM2-135M, IR, INT8 quantization, fixed-weight specialization, tiling, Verilog generation, generated testbenches, bit-exact verification, a programmable-weight baseline, and a Yosys comparison. In that design the non-linear operators and the token loop run in host software; dequantization is outside RTL.

Part II closes the project: the **whole decoder runs in generated hardware**, from token id in to next token id out.

## 2. Problem statement
Extend the compiler so that, given a pretrained decoder-only transformer (SmolLM2-135M), it
1. represents the complete decode step as a graph IR (embedding, RMSNorm, linear, RoPE, GQA attention with KV cache, SwiGLU, residual add, LM head, argmax),
2. assigns integer formats to every tensor and folds all compile-time constants into requantization parameters, tables and ROM images,
3. schedules the model onto a model-specific accelerator architecture whose weights, tables and parameters live in **immutable ROMs**,
4. generates synthesizable Verilog for the accelerator (library units + generated ROMs, schedule and top level) and a generated testbench that runs token generation,
5. verifies, against an independent integer reference, that the RTL produces the **same intermediate tensors and the same tokens bit for bit**, and
6. quantifies the quality of the integer model against fp32 and the area, latency and throughput trade-offs of the architecture, ending in an explicitly labeled projection toward an ASIC.

## 3. Why the full-model design cannot be fully spatial
Part I's fixed-weight design instantiates one constant multiplier per weight. SmolLM2-135M has about 106.2M layer weights plus a 28.3M tied embedding/LM-head table. Even at a conservative few dozen bytes of Verilog per weight term, the layer weights alone are multiple GB of source, beyond what Verilator or Yosys can ingest on a laptop (U0.4 measures this from Part I's real file sizes). Part II therefore uses a **ROM-backed, time-multiplexed engine**: weights are still compile-time constants in immutable ROM, but a parameterized MAC array is reused across layers. This is also the closest open analogue of a mask-ROM model-specific chip. Part I's spatial generator is retained for a block-level comparison (EA4).

## 4. Research questions
Hypotheses may be false; negative or mixed results are reported as measured.
- **RQ4 (end-to-end correctness):** Can a compiler-generated accelerator with immutable weights reproduce an integer reference model's token stream exactly, at full scale? *Expected: yes.*
- **RQ5 (numerics):** How much quality (perplexity, KL, top-1 agreement, greedy match length) does an integer-only, hardware-friendly SmolLM2-135M retain relative to fp32, and which operators and precisions dominate the loss?
  - H4: naive static INT8 everywhere loses measurable quality; a small number of targeted choices (residual width, a few higher-precision activations, SmoothQuant-style folding into constants) recover most of it at low hardware cost.
- **RQ6 (architecture):** What is the cycles/token, area and memory split of the ROM-based engine as lane count (R×C) varies, and how does it compare with the spatial design at block level?
  - H5: weight-ROM bits dominate silicon area at modest lane counts; compute logic and non-linear units are a small fraction.
  - H6: linear-engine cycles dominate cycles/token; non-linear units add a small share (measured, not assumed).

## 5. Scope tiers
**MUST (project fails without these)**
- U0: audit and reproduction of Part I; retrospective; model facts v2.
- Numerics-first: NumPy integer decoder (bit-exact spec), frozen formats, quality measurements, golden traces.
- Full-decoder graph IR, format assignment, constant folding, ROM packing, schedule, memory plan, and a **compiled-artifact interpreter** that reproduces the integer decoder from compiler outputs.
- RTL unit library with per-unit bit-exact tests (exhaustive where the domain allows) and mutation tests.
- Tiny random-weight transformer end to end in RTL: token-exact across many prompts, trace-equal.
- **Full SmolLM2-135M in RTL simulation: token-exact greedy decode** for a battery of prompts (Verilator).
- Quality report (integer vs fp32) and architecture/synthesis study with fairness protocol.
- CLI, one-command demo, whole-project report (Part I + II), slides, viva readiness.

**SHOULD**
- Cross-simulator equality (Icarus vs Verilator) on the tiny model.
- Long-context test up to `MAX_CTX`.
- Spatial vs ROM-sequential block comparison (EA4).
- Liberty-based area for units and engine; ASIC projection with sensitivity analysis.
- Weight/precision sensitivity table for the report.

**STRETCH (only after gate HE; never at the expense of MUST/SHOULD)**
- V1: layer-pipelined multi-token throughput model.
- V2: zero-weight skipping in the engine.
- V3: weight bit-width study (W6/W4) for ROM size vs quality.
- V4: top-k/temperature sampling with a deterministic LFSR.

## 6. Fallback ladder (decided by ADR before the end of W9, not after)
- **F0 (headline):** full 30-layer SmolLM2-135M token-exact in RTL simulation.
- **F1:** first L layers of the real model (L ≥ 4) plus Python for the rest, token-exact for the hybrid, with the RTL/Python boundary at a layer output; full-model numbers come from the Python integer model proven bit-exact on the RTL-covered portion.
- **F2:** tiny random model only. Not acceptable as the headline; indicates a failed project phase and a supervisor discussion.

## 7. Non-goals
Tokenizer or detokenizer in hardware; training or fine-tuning; LoRA; sampling beyond greedy (except stretch V4); second model; prefill-specific datapath; batching or multiple concurrent sequences; FPGA bitstreams or board bring-up; place-and-route, tape-out, power signoff; transistor-level modeling; speculative decoding.

## 8. Requirements
**Functional**
| ID | Requirement |
|---|---|
| FR2-1 | Graph IR with ops {Embed, RMSNorm, Linear, RoPE, Attention(GQA, causal, KV), SiLU, EltMul, Add, ArgMax}; tensor kinds CONSTANT / DYNAMIC / STATE; per-edge integer format; JSON-serializable and lossless. |
| FR2-2 | Format assignment from calibration, with constant folding of γ, 1/√d, and per-channel scales into requantization parameters. |
| FR2-3 | ROM packing with an address map; every ROM image has a sha256 in the manifest. |
| FR2-4 | Static schedule (microprogram) with a layer loop, a memory plan, and a cycle estimate. |
| FR2-5 | Compiled-artifact interpreter that executes the schedule on ROM images and equals the numerics decoder bit-exactly. |
| FR2-6 | RTL units: requant, RMSNorm, RoPE, softmax, SiLU/SwiGLU, residual add, embedding gather, argmax, KV cache, linear engine (R×C), sequencer. |
| FR2-7 | Generated top level with token-step interface; generated testbench with trace dump. |
| FR2-8 | Verilator flow for the full model; Icarus flow for the tiny model. |
| FR2-9 | Yosys flow for units, engine and memory accounting. |
| FR2-10 | CLI covering compile-model, run, verify, synth. |

**Verification**
| ID | Requirement |
|---|---|
| VR2-1 | RTL output tokens equal the integer reference tokens exactly on every test prompt. |
| VR2-2 | RTL intermediate tensors equal the golden trace at every traced op (tiny: all; full: sampled layers/tokens plus full on layer 0). |
| VR2-3 | Every unit passes bit-exact tests; units with domains ≤ 2²⁰ are tested exhaustively. |
| VR2-4 | All mutants in the Part II mutation suite are killed. |
| VR2-5 | Weight audit v2: ROM hex images equal the quantized tensors; ROM modules have no write path; generated RTL contains no activations, tokens or expected values. |
| VR2-6 | Determinism: generating twice is byte-identical. |
| VR2-7 | Quality (integer vs fp32) measured on pinned sets: perplexity, KL, top-1 agreement, greedy match length. Thresholds set at gate HA from measurements, then frozen. |

**Non-functional**
| ID | Requirement |
|---|---|
| NFR2-1 | Fresh clone → `make demo2-fast` (minutes) and `make demo2` (documented runtime) reproduce headline results. |
| NFR2-2 | Laptop/workstation CPU only; budgets measured at U0.4 and frozen. |
| NFR2-3 | Pinned versions for Python deps, model revision, EDA tools; recorded in `results/env.json`. |

## 9. Headline success criteria
1. Full-model RTL simulation generates tokens identical to the integer reference for the prompt battery, with 100% mutation kill rate and clean audits.
2. A quality table (integer vs fp32) with a defensible account of where loss comes from.
3. An architecture study (cycles, area, memory split) with a documented fairness protocol and a clearly labeled ASIC projection, including what it does **not** show.
4. A whole-project report and demo reproducible from a fresh clone.
5. The student can explain and defend every component.

## 10. Glossary additions
**Decode step:** one forward pass for one token at the current position. **Prefill:** processing prompt tokens, done here as repeated decode steps. **Residual stream:** the running hidden vector `h` per token. **Requant:** integer rescale `sat(rshift_rnd(acc·M, s))`. **Trace:** per-token, per-op dump of intermediate tensors. **Dyadic scale:** `M·2⁻ˢ`. **R×C:** engine lanes (R output rows × C input elements per cycle).
