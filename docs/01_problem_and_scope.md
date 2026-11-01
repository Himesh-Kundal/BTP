# 01 — Problem Statement, Scope and Requirements

## 1. Context
Pretrained LLM weights do not change during inference; only activations do. Conventional accelerators still treat weights as runtime operands fetched through a memory hierarchy and multiplied by general-purpose multipliers. A model-specific design can instead bake the known weights into the datapath as constants (inspired by model-specific silicon approaches such as Taalas), letting synthesis specialize the arithmetic per coefficient.

Doing this by hand for each layer does not scale across shapes and weight values. A compiler should do it.

## 2. Problem statement
Design and implement a compiler framework that
1. extracts dense linear layers from a pretrained decoder-only transformer (SmolLM2-135M),
2. represents them in a framework-independent, hardware-independent IR,
3. quantizes them to INT8 with INT32 accumulation,
4. specializes them so the quantized weights become immutable compile-time constants,
5. maps them onto a configurable tiled hardware structure,
6. emits synthesizable Verilog with **no runtime weight interface**, plus automatically generated testbenches,
7. verifies the generated RTL **bit-exactly** against an independent integer reference, and
8. quantifies, against a programmable-weight baseline generated from the same IR, what fixed-weight specialization changes in synthesized hardware cost.

## 3. Research questions and hypotheses
Hypotheses may be false. A negative or mixed result is a valid result and is reported as measured.

- **RQ1 (correctness):** Can fixed-weight RTL for real LLM layers be generated automatically and shown bit-exact against an integer reference? *Expected: yes.*
- **RQ2 (cost):** How does logic area of fixed-weight hardware compare with an equivalent programmable-weight design at the same tile size?
  - H1: fixed-weight logic is smaller than the programmable design's multiplier array.
  - H2: fixed-weight area depends on the weight values (zeros vanish, small magnitudes are cheap); real weights are cheaper than uniform-random INT8 weights.
  - H3: critical-path proxy is comparable or shorter.
- **RQ3 (quality):** What accuracy does INT8 per-channel weight / per-tensor activation quantization retain per layer type? (A property of the quantizer, reported separately from RTL correctness.)

## 4. Scope tiers
**MUST (project fails without these)**
- Walking skeleton: toy linear → IR → Verilog → generated testbench → simulation PASS → mutation FAIL.
- Frontend extraction of the 7 projection types from SmolLM2-135M.
- IR, INT8 quantization with calibration, accumulator bound analysis.
- Fixed-weight specialization, tiling (with padding), Verilog backend, testbench and vector generation.
- Programmable-weight baseline generator and the F-vs-P comparison via Yosys.
- Bit-exact verification across the test matrix, real-layer slices, mutation tests, weight audit.
- CLI, reproducible demo, report.

**SHOULD**
- Verilator lint and cross-simulator equality check.
- Weight-distribution experiment (real vs random vs shuffled).
- One directly synthesized larger point that validates extrapolation.
- Coverage of a first, middle and last layer.

**STRETCH (only after Gate G6 passes; never at the expense of MUST/SHOULD)**
- S1: Python full-model simulator using the bit-exact INT8 linear reference inside the HF forward pass; report perplexity and token agreement.
- S2: coefficient-ROM / time-multiplexed backend mode.
- S3: constant-multiplier arithmetic optimization (CSD shift-add pass) evaluated against leaving `*` to synthesis.

## 5. Non-goals (Part I)
Full transformer RTL; attention, softmax, RoPE, RMSNorm, embedding or KV-cache hardware; requantization/dequantization hardware; monolithic synthesis of all 135M parameters; FPGA bitstreams; ASIC fabrication; place-and-route; power modeling; a second model; training or fine-tuning; custom hardware simulators.

## 6. Requirements
**Functional**
| ID | Requirement |
|---|---|
| FR-1 | Extract all 7 projection types (q, k, v, o, gate, up, down) from any decoder layer, with weights, shapes, dtype, bias presence, module path. |
| FR-2 | IR is hardware-independent, framework-independent, JSON-serializable, lossless on round-trip. |
| FR-3 | INT8 symmetric quantization, INT32 accumulation, bias support, scale metadata (contract in `docs/02` §2). |
| FR-4 | Fixed-weight specialization: weights become constants; generated modules expose no weight inputs. |
| FR-5 | Configurable tile `(Nt, Kt)`; non-multiple shapes padded correctly; tile manifest emitted. |
| FR-6 | Verilog generation is synthesizable, deterministic, lint-clean. |
| FR-7 | Testbench and vector files generated automatically; testbench reports `CHECKED` and `MISMATCH` counts. |
| FR-8 | Programmable-weight baseline generated from the same IR and tile manifest. |
| FR-9 | Simulation pipeline: Icarus (primary), Verilator (lint + larger runs), machine-readable report. |
| FR-10 | Yosys synthesis flow producing JSON/CSV metrics. |
| FR-11 | CLI covering extract → compile → verify → synth. |

**Verification**
| ID | Requirement |
|---|---|
| VR-1 | RTL output equals the integer reference exactly on every test vector. |
| VR-2 | Integer reference vs float reference error is measured and reported (not asserted bit-exact). |
| VR-3 | Every seeded fault in the mutation suite is detected. |
| VR-4 | Weight audit: constants parsed back from RTL equal the quantized weight tensor. |
| VR-5 | Determinism: generating twice gives byte-identical output. |

**Non-functional**
| ID | Requirement |
|---|---|
| NFR-1 | Fresh clone → `make demo` reproduces the headline result. |
| NFR-2 | CPU-only laptop; no GPU. Time budgets are measured at T4.5 and then frozen. |
| NFR-3 | Pinned versions: Python deps, model revision, EDA tools (recorded in `results/env.json`). |

## 7. Headline success criteria
1. Generated RTL for real SmolLM2 layer tiles passes bit-exact tests across all 7 projection types, with 100% mutation kill rate and a clean weight audit.
2. The F-vs-P synthesis comparison exists with a documented fairness protocol and honest caveats.
3. Anyone can clone the repo and reproduce 1 and 2 from one command.
4. The student can explain and defend every component (see `docs/05_workflow.md` §8).

## 8. Glossary
**F** — fixed-weight design (weights are constants). **P** — programmable-weight baseline. **Tile** — `Nt × Kt` block of the weight matrix. **Core** — combinational dot-product module. **Wrapper** — registered-I/O module around cores. **Golden** — frozen expected-output files. **Mutation** — deliberately injected RTL fault used to test the verification. **Gate** — phase exit review.

## 9. Long-term direction (context only, not Part I work)
Part II: add embedding, RMSNorm, RoPE, attention, softmax, SwiGLU, residuals, KV-cache to compile a full transformer block. Part III: technology-aware synthesis, mask-ROM or custom memory, physical design, eventual ASIC.
