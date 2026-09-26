# Project Execution Log

2026-11-01 19:43 | T0.1 | Pinned environment with uv and check_env.sh | make env-check passed, results/env.json created | 19b7527
2026-11-01 19:49 | T0.2 | Repo skeleton, Makefile, and test configuration | make check passed on empty skeleton, 1 test green | 7ed2e74
2026-11-01 19:51 | T0.3 | Independent integer reference oracle and tests | 9 hand-calculated tests green, N1-N7 contracts verified | 5805dea
2026-11-01 19:52 | T0.4 | Architectural layering and forbidden import test | Golden Rule 5 enforced, AST checker passes and catches negative controls | 26b7450
2026-11-01 19:53 | T0.5 | Initialize state tracking, STATE.md, and LOG.md | STATE, LOG, BLOCKERS initialized | da904ef
2026-11-01 19:54 | T0.G | Gate G0 review and verification pass | All G0 checklist items verified green, report at results/gates/G0.md | 6597cd7
2026-11-01 19:56 | T1.1 | Minimal LinearOp IR dataclass and invariant tests | Invariants I1, I2, I5 validated | 035f4c7
2026-11-01 19:59 | T1.2 | Combinational fixed-weight core codegen | Verilog-2001 core, no weight ports, iverilog -Wall clean | 1f40932
2026-11-01 20:01 | T1.3 | Vector writer and testbench generator | x.hex, y_exp.hex, self-checking TB with TB_RESULT | 4527748
2026-11-01 20:03 | T1.4 | Simulator runner, negative control, and make sim | CHECKED rule enforced, negative control confirmed FAIL | ec1ce12
2026-11-01 20:04 | T1.5 | Mutation testing engine (M1-M3) | 100% kill rate (9/9 killed) across sign-flip, +1, term drop | bbd252f
2026-11-01 20:06 | T1.6 | Toy end-to-end compiler demo and make demo-toy | Single command make demo-toy prints PASS and kill matrix | e028f10
2026-11-01 20:08 | T1.G | Gate G1 review and walking skeleton verification | All G1 checklist items green, walkthrough in docs/walkthroughs/P1.md | 81167d8
2026-11-01 20:10 | T2.1 | Pinned SmolLM2-135M model loader | Loads pinned revision fp32 on CPU, generates results/model.json | 775f407
2026-11-01 20:12 | T2.2 | Linear layer extractor | Extracts all 210 projections across 30 layers | a82dddb
2026-11-01 20:15 | T2.3 | Model facts reconciliation | Profiled model matches docs/02 §1 exactly, docs/model_facts.md created | 8b069b4
2026-11-01 20:17 | T2.4 | Weight artifact dumper | Dumps weight.npy and meta.json with SHA256 hashes, reload test bit-identical | 491708b
2026-11-01 20:20 | T2.5 | Forward-hook activation capture | Deterministic activation capture with fixed seed 42, byte-identical on rerun | 7609642
2026-11-01 20:25 | T2.6 | Frontend mathematical validation | Proves X @ W.T matches module output <= 1e-5 across all 7 projection types | 105980d
2026-11-01 20:28 | T2.G | Gate G2 review and verification pass | All G2 checklist items verified green, report at results/gates/G2.md | 24743bc
2026-11-01 20:35 | T3.1 | Complete IR dataclasses, serialization, and artifact loading | TensorRef, LinearOp, Graph, lossless JSON + .npy round-trip, I1-I5 verified | 87bfd42
2026-11-01 20:38 | T3.2 | Symmetric INT8 weight quantization per rule N2 | Property tests pass, range [-127, 127], zero-row scale 1.0, error bound verified | 2f20b12
2026-11-01 20:40 | T3.3 | Activation calibration and quantization per rule N3 | Absmax & percentile calibration, deterministic scale and quantized activations | 1cb6000
2026-11-01 20:43 | T3.4 | Synthetic bias quantization per rule N5 | Scaling sb = sx * sw, 32-bit clipping, bit-exact against reference oracle | a6fb082
2026-11-01 20:45 | T3.5 | Accumulator bound pass per rule N4 | K*128*127 + |bq|max < 2^31 enforced, acc_bits_min annotated on IR | 3ebee29
2026-11-01 20:48 | T3.6 | Quantization quality evaluation (Claim C2) and thresholds | 21 configs evaluated, results/quant_quality.json, ADR-0001 frozen | 4c7679a
2026-11-01 20:50 | T3.7 | End-to-end compiler pipeline driver | Sequential passes, tested on real SmolLM2 layers, pipeline.py created | 07cdb3e
2026-11-01 20:52 | T3.G | Gate G3 review and verification pass | All G3 checklist items green, Numeric checklist audited, G3.md report | d3968a5
2026-11-01 20:58 | T4.1 | specialize constant weights and port stripping | tests/test_constant_weights.py PASS (6/6), make check PASS (101/101), Invariant I3 enforced | 578a176
2026-11-01 21:02 | T4.2 | matrix tiling, zero-padding, manifest generation | tests/test_tiling.py PASS (6/6), make check PASS (107/107), property tests green | 03166c9
2026-11-01 21:06 | T4.3 | weight literal encoder and sparsity optimization | tests/test_weight_encoder.py PASS (7/7), make check PASS (114/114), Verilog-2001 signed conventions | d2db0ba
2026-11-01 21:12 | T4.4 | parametric tile core codegen | tests/test_verilog_codegen.py PASS (3/3), make check PASS (117/117), iverilog and verilator lint clean | 8745851
2026-11-01 21:20 | T4.5 | full module generator and budgets | tests/test_module_generator.py PASS (3/3), make check PASS (120/120), real 8x16 sim PASS, 576x576 budget frozen (0.17s, 7.33MB) | c252d12
