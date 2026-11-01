# 04 — Plan and Task Backlog

Only tasks in this file may be worked on. Each task becomes `tasks/T<id>.md` (template in `docs/05` §6) when it starts. "Done when" is executable wherever possible.

## 1. Timeline (14 weeks; set the start date in STATE.md)
| Phase | Weeks | Goal | Gate |
|---|---|---|---|
| P0 Bootstrap | 1 | Environment, skeleton, oracle, harness state | G0 |
| P1 Walking skeleton | 1–2 | Toy linear end to end, with mutation check | G1 |
| P2 Frontend and real weights | 2–3 | Extract 7 projection types, capture activations | G2 |
| P3 IR, quantization, calibration | 3–5 | Quantized fixed-weight IR with quality report | G3 |
| P4 Specialization, tiling, backend F | 5–7 | Real tiles to Verilog, padding, composition | G4 |
| P5 Programmable baseline | 7–8 | P generator, F == P == reference | G5 |
| P6 Verification hardening | 8–10 | Matrix, real layers, mutation, Verilator | G6 |
| P7 Synthesis evaluation | 10–11 | Yosys experiments E1–E5 | G7 |
| P8 Integration, CLI, report | 12–13 | `make demo`, README, report | G8 |
| Buffer | 14 | Fixes and rehearsal. **No new features after the end of W12.** | — |

Stretch S1–S3 may start only after G6 passes, in parallel with P7/P8, and never consume buffer time.

**Cut list if behind (cut in this order):** S3 → S2 → S1 → second and third layer depth → extrapolation beyond directly measured points → Verilator large runs. **Never cut:** walking skeleton, bit-exact verification, mutation suite, weight audit, F-vs-P comparison, report.

## 2. Tasks
Tags: `rtl` and `numeric` and `gate` tasks require a verifier pass. Estimates are in agent-hours of focused work and are only for the 2× escalation rule.

### P0 — Bootstrap
| ID | Task | Deps | Done when |
|---|---|---|---|
| T0.1 | Pinned environment (`pyproject`/lock), `scripts/check_env.sh` printing versions of iverilog, vvp, verilator, yosys, python, torch, transformers; writes `results/env.json` | — | `make env-check` exits 0 and env.json exists |
| T0.2 | Repo skeleton from `docs/02` §9; Makefile with targets in AGENTS.md; pytest + ruff config; `.gitignore` | T0.1 | `make check` green on empty skeleton |
| T0.3 | `verification/reference.py`: int linear (int64), quantize/dequantize per N2–N6; hand-computed tests incl. the 3,7,−2 example | T0.2 | `pytest tests/test_reference.py` green; ≥ 8 hand-verified cases |
| T0.4 | `tests/test_layering.py` (forbidden imports, SmolLM string scan) | T0.2 | test green and fails when a forbidden import is injected |
| T0.5 | Initialize `state/STATE.md`, `LOG.md`, `BLOCKERS.md`; copy docs in | T0.2 | files exist; STATE has start date and supervisor info |

### P1 — Walking skeleton (toy shapes, hand-built IR)
| ID | Task | Deps | Done when |
|---|---|---|---|
| T1.1 | Minimal `LinearOp` dataclass + validation | T0.2 | I1, I2 tests green |
| T1.2 | Codegen v0: combinational fixed-weight core for small N,K (direct constants, signed conventions) | T1.1 | golden-file test on a 2×4 and the 1×3 example; `iverilog -Wall` clean |
| T1.3 | Vector writer (`x.hex`, `y_exp.hex`) from the reference; testbench generator (readmemh, packing, summary line) | T0.3, T1.2 | TB prints `TB_RESULT CHECKED=n MISMATCH=0` |
| T1.4 | Simulator runner (`verification/simulator.py`): compile, run, parse, enforce the CHECKED rule, emit `report.json` | T1.3 | `make sim` passes toy; corrupted `y_exp.hex` is reported FAIL (negative control) |
| T1.5 | `verification/mutate.py` with mutants M1 (sign flip), M2 (weight +1), M3 (drop term) | T1.4 | all three killed on toy |
| T1.6 | `examples/compile_linear.py --toy` and `make demo-toy` | T1.4 | single command prints PASS and the mutation kill matrix |
| T1.G | **Gate G1** | all P1 | gate report `results/gates/G1.md` + verifier PASS |

### P2 — Frontend and real weights
| ID | Task | Deps | Done when |
|---|---|---|---|
| T2.1 | `model_loader.py`: load SmolLM2-135M at a pinned revision, fp32, CPU; record revision and file hashes | T0.1 | loader test (skippable offline) + hashes in `results/model.json` |
| T2.2 | `architecture_parser.py` + `linear_extractor.py`: enumerate linear modules, classify into the 7 types, capture shape, dtype, bias presence, module path | T2.1 | table printed by `examples/inspect_smollm.py` |
| T2.3 | Write `docs/model_facts.md` from the real inspection; reconcile with the expected table in `docs/02` §1 | T2.2 | doc exists; any discrepancy raised as an ADR |
| T2.4 | Dump artifacts (`weight.npy`, `meta.json`) with sha256 | T2.2 | files + hashes; reload test equals original tensor bit-for-bit |
| T2.5 | Forward-hook activation capture on a fixed calibration set (fixed prompts and seed, recorded) | T2.2 | `calib/x_sample.npy` reproducible (same hash on rerun) |
| T2.6 | Frontend validation: `X @ W.T` in torch equals the module output | T2.4, T2.5 | max abs diff ≤ 1e-5 on captured activations |
| T2.G | **Gate G2** | all P2 | gate report + verifier PASS |

### P3 — IR, quantization, calibration
| ID | Task | Deps | Done when |
|---|---|---|---|
| T3.1 | IR complete (`tensor.py`, `operators.py`, `linear.py`, `graph.py`), JSON + `.npy` serialization, loader from artifacts | T1.1, T2.4 | I1–I5 tests green, round-trip lossless |
| T3.2 | `compiler/quantization.py`: weight quantization per N2 | T3.1, T0.3 | property tests: `|W − s_w·W_q| ≤ s_w/2`; range ±127; zero-row case |
| T3.3 | Activation calibration per N3 (absmax and percentile) | T3.1, T2.5 | scales recorded in IR; deterministic |
| T3.4 | Bias handling per N5 | T3.2 | synthetic-bias tests vs reference |
| T3.5 | Accumulator bound pass per N4 | T3.2 | rejects an artificial overflow case; records `acc_bits_min` |
| T3.6 | Quality report (C2): 7 types × layers {first, middle, last}; writes `results/quant_quality.json` and freezes thresholds | T3.2–T3.5 | report exists; thresholds recorded as an ADR |
| T3.7 | Pipeline driver `compiler/pipeline.py` (IR → quantized fixed-weight IR) | T3.2–T3.5 | integration test on a real layer slice |
| T3.G | **Gate G3** | all P3 | gate report + verifier PASS (`numeric`) |

### P4 — Specialization, tiling, backend F
| ID | Task | Deps | Done when |
|---|---|---|---|
| T4.1 | `constant_weights.py`: mark weights CONSTANT, strip any weight-input notion | T3.7 | I3 test |
| T4.2 | `tiling.py` + tile manifest; padding | T3.7 | property tests from `docs/02` §4 incl. shapes with N or K not multiples of the tile |
| T4.3 | `weight_encoder.py`: int8 → Verilog literals (`8'sd`, negatives), sparsity-aware term omission | T4.1 | unit tests incl. −127, 0, ±1, 127 |
| T4.4 | `verilog_codegen.py`: tile core for arbitrary `Nt×Kt` | T4.2, T4.3 | golden-file test; lint clean |
| T4.5 | `module_generator.py`: tile instantiation, adder trees, bias, wrapper (latency L), top level; measure generation time and file size for a full 576×576 layer and freeze budgets | T4.4 | real 8×16 tile passes sim; full layer generates; budgets recorded |
| T4.6 | Interface conformance test (ports, widths, signedness, no weight port) | T4.5 | test green and fails on an injected weight port |
| T4.7 | Determinism test (generate twice, byte-compare) | T4.5 | green |
| T4.G | **Gate G4** | all P4 | gate report + verifier PASS (`rtl`) |

### P5 — Programmable baseline
| ID | Task | Deps | Done when |
|---|---|---|---|
| T5.1 | P generator (`W_reg` array with load port, generic multipliers) and `P_bus` variant, from the same IR and manifest | T4.5 | lint clean; same widths asserted by test (N10) |
| T5.2 | Testbench support for the P weight-load phase (`w.hex`) | T5.1 | P passes toy and a real tile |
| T5.3 | Differential test: `Y_F == Y_P == Y_ref` on the standard vector set | T5.2 | green across ≥ 5 shapes |
| T5.G | **Gate G5** | all P5 | gate report + verifier PASS (`rtl`) |

### P6 — Verification hardening
| ID | Task | Deps | Done when |
|---|---|---|---|
| T6.1 | Test-matrix generator (dims × tile configs × weight classes × input classes, see `docs/06` §A) | T5.3 | matrix runs in `make regression`; time budget recorded |
| T6.2 | Real-layer tests: slices of all 7 types from first, middle, last layer, with real captured activations quantized by the IR scales | T6.1, T2.5 | all PASS |
| T6.3 | Verilator lint `-Wall` clean; Verilator build/run for larger slices; cross-simulator equality check | T6.1 | outputs identical on both simulators |
| T6.4 | `verification/weight_audit.py`: parse RTL constants, compare with `W_q` | T4.5 | audit passes on all generated designs; fails on an injected change |
| T6.5 | Extend mutation suite to M1–M9 (`docs/06` §A) | T1.5, T6.1 | kill matrix 100%; survivors are bugs in tests, not accepted |
| T6.6 | Regression runner and `results/regression.json` | T6.1–T6.5 | `make regression` green, deterministic |
| T6.7 | Waveform walkthrough: VCD for one passing and one deliberately failing case; `docs/walkthroughs/waveforms.md` | T6.6 | doc exists with annotated screenshots by the student |
| T6.G | **Gate G6** | all P6 | gate report + verifier PASS (`rtl`, `numeric`) |

### P7 — Synthesis evaluation
| ID | Task | Deps | Done when |
|---|---|---|---|
| T7.1 | Yosys flow scripts for F and P (generic gates; ice40 LUT proxy; liberty if O-3 provides it); parse `stat` and `ltp` to JSON | T5.1 | deterministic JSON for one tile |
| T7.2 | Fairness protocol checklist implemented in the script (same flags, widths, flatten, reporting views) | T7.1 | test asserts both designs use the same script and widths |
| T7.3 | Experiments E1 (area vs tile size), E3 (depth proxy), with ≥ 5 weight seeds where random | T7.2 | `results/E1.csv`, `E3.csv` |
| T7.4 | Experiment E2: real vs uniform-random vs magnitude-matched vs shuffled-real weights | T7.2 | `results/E2.csv` |
| T7.5 | E5: extrapolation model, validated against one directly synthesized larger point | T7.3 | `results/E5.json` with error of the estimate |
| T7.6 | Plots and tables generated by script from `results/` | T7.3–T7.5 | `make report-figs` reproduces figures |
| T7.G | **Gate G7** | all P7 | gate report + verifier PASS |

### P8 — Integration, CLI, report
| ID | Task | Deps | Done when |
|---|---|---|---|
| T8.1 | CLI: `llmhw extract / compile / verify / synth` with `--tile --arch --layer --out` | T6.6, T7.1 | CLI tests; `--help` documented |
| T8.2 | Multi-layer demo: all 7 projection types of layer 0 compiled and verified | T8.1 | `make demo` passes |
| T8.3 | Varying-shape demo: ≥ 3 distinct (N,K) pairs including a full-layer RTL generation | T8.1 | listed in `results/demo.json` |
| T8.4 | README, architecture doc, limitations; reproduce-from-fresh-clone test | T8.2 | fresh clone in a clean directory passes `make demo` |
| T8.5 | Report draft with every number traced to `results/` | T7.6, T8.3 | traceability table complete (`docs/07` §7) |
| T8.6 | Presentation skeleton and viva prep notes | T8.5 | student has answered all explain-it questions |
| T8.G | **Gate G8** | all P8 | gate report + verifier PASS |

### Stretch (only after G6)
| ID | Task | Done when |
|---|---|---|
| S1 | Python full-model INT8 simulator: replace the 7 linear types in all layers with the bit-exact integer reference; report perplexity on a small fixed eval set and next-token agreement vs fp32 | `results/S1.json`; reference proven bit-exact to RTL on slices (G6 evidence) |
| S2 | Coefficient-ROM time-multiplexed mode and comparison | passes the same verification suite |
| S3 | CSD shift-add arithmetic pass vs `*` left to synthesis | area comparison in `results/S3.csv` |
