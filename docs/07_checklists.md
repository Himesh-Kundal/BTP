# 07 — Checklists

Use these literally. Each item should be answerable with a command, a file, or a link.

## 1. Task Definition of Done (every task)
- [ ] Every acceptance bullet in the task file is checked with a command that was actually run
- [ ] Tests were written first and mapped to acceptance bullets
- [ ] `make check` green; ruff clean; no new warnings from iverilog/verilator on touched RTL
- [ ] No test weakened, skipped, or deleted; no frozen file touched (or an ADR exists)
- [ ] No hardcoded expected values in generators; no model-specific constants in `ir/ compiler/ backend/ verification/`
- [ ] Generated outputs are deterministic (rerun diff is empty)
- [ ] Docs touched if behaviour or interface changed
- [ ] Task file Evidence filled with commands and summarized outputs
- [ ] LOG line appended; STATE updated (current, next 3, blockers, handoff)
- [ ] Commit message `T<id>: ...`; branch merged only after green
- [ ] If tagged `rtl`, `numeric` or `gate`: verifier verdict is PASS

## 2. Session start / end
**Start:** [ ] `git status` clean or understood · [ ] `make check` green · [ ] STATE read · [ ] task file read · [ ] spec sections read
**End:** [ ] work committed · [ ] STATE handoff note written (done / next / gotchas) · [ ] open blockers logged · [ ] no uncommitted generated files that should be tracked

## 3. RTL review (generated Verilog)
- [ ] Verilog-2001 subset only; no SystemVerilog constructs
- [ ] All data signals signed; constants carry `s` (`8'sd3`, `-8'sd2`); no unsigned operand in any arithmetic expression
- [ ] Product wires have explicit width (16); accumulation has explicit sign extension to the IR accumulator width
- [ ] Packing order matches `docs/02` §6 (element 0 in the LSBs)
- [ ] Padding lanes are zero; padded outputs not exported
- [ ] Zero-weight terms omitted by design; ±1 handled; nothing else hand-optimized (unless S3)
- [ ] F has no weight port, no `w_*` signals, no writable weight storage
- [ ] Latency equals the manifest value; `valid_out` aligned with `y_flat`
- [ ] No latches, no `initial`, no `$display`, no `#` delays, no tri-states
- [ ] Reset only on valid pipeline; data path not reset
- [ ] File header has the manifest hash, no timestamps or absolute paths
- [ ] Lint: iverilog `-Wall`, verilator `-Wall`, Yosys `check -assert` all clean
- [ ] Module and file names match the convention

## 4. Numeric / quantization review
- [ ] Rounding is `rint` (half-to-even), computed in float64
- [ ] Weights clipped to [−127, 127]; zero-row case handled (`s_w = 1.0`)
- [ ] Per-channel weight scales have shape `[N]`; activation scale is a scalar from the recorded calibration
- [ ] Calibration set, seed, statistic recorded in IR
- [ ] Bias quantization uses `s_x·s_w[n]`; synthetic-bias test exists
- [ ] Accumulator bound uses `K·128·127 + |b_q|max` and rejects ≥ 2³¹
- [ ] Reference uses int64; RTL uses the IR accumulator width
- [ ] C1 asserted exactly; C2 reported with all four metrics; thresholds frozen via ADR
- [ ] Dequantization appears nowhere in RTL

## 5. Gate checklists

### G0 — Bootstrap
- [ ] `make env-check` green; versions in `results/env.json`
- [ ] Makefile targets exist and run on the empty skeleton
- [ ] Reference tests green (≥ 8 hand-verified cases)
- [ ] Layering test exists and fails when a forbidden import is injected
- [ ] STATE/LOG/BLOCKERS initialized; start date and supervisor info recorded
### G1 — Walking skeleton
- [ ] `make demo-toy` prints PASS from a single command
- [ ] Negative control FAILs; M1–M3 killed
- [ ] Generated toy RTL has no weight port; lint clean; deterministic
- [ ] Verifier PASS; student answered P1 explain-it questions
### G2 — Frontend
- [ ] 7 projection types extracted for ≥ 1 layer; table printed
- [ ] `docs/model_facts.md` written; discrepancies resolved by ADR
- [ ] Reload of dumped weights is bit-identical; sha256 recorded
- [ ] Activation capture reproducible (same hash on rerun)
- [ ] torch `X @ W.T` matches module output within 1e-5
- [ ] Layering test green
### G3 — IR and quantization
- [ ] IR round-trip lossless; invariants I1–I5 tested
- [ ] Quantization property tests green; bias and bound passes tested
- [ ] C2 report for 7 types × {first, middle, last} written; thresholds frozen by ADR
- [ ] Numeric checklist (§4) completed by the verifier
### G4 — Backend F
- [ ] Tiling property tests green, including non-multiple shapes
- [ ] Real 8×16 tile and a non-multiple synthetic shape pass in Icarus
- [ ] Interface test and weight audit pass; both fail on injected faults
- [ ] Determinism test green; generation time and file size budgets recorded for a full 576×576 layer
- [ ] RTL checklist (§3) completed by the verifier
### G5 — Baseline P
- [ ] P and P_bus lint clean; widths identical to F (tested)
- [ ] P passes toy and real tile with the weight-load phase
- [ ] `Y_F == Y_P == Y_ref` across ≥ 5 shapes
### G6 — Verification hardening
- [ ] Full L1 matrix green; L2/L3 pass for all 7 types × first/middle/last
- [ ] Mutation kill rate 100% (M1–M9), kill matrix saved
- [ ] Verilator lint clean; Icarus and Verilator outputs identical
- [ ] `make regression` deterministic with recorded runtime
- [ ] Waveform walkthrough completed by the student
- [ ] Stretch items may start only now
### G7 — Evaluation
- [ ] Fairness protocol (`docs/06` B2) checked item by item
- [ ] E1, E2, E3 results with ≥ 5 seeds where random; E5 validated by one directly synthesized larger point
- [ ] Figures regenerate from `make report-figs`
- [ ] Synthesis summary lists what the data supports vs only suggests, with caveats
- [ ] Verifier re-ran at least one data point from a clean checkout
### G8 — Release
- [ ] Fresh clone in a clean directory: `make demo` passes
- [ ] CLI documented; README complete; limitations stated
- [ ] Report draft complete; traceability table has no unsourced number
- [ ] AI-assistance disclosure included per institute policy
- [ ] Student answered all explain-it questions; viva rehearsal done

## 6. Demo readiness
- [ ] One command shows: extract → compile → generate RTL → simulate PASS → mutation kill matrix → synthesis summary
- [ ] A pre-run output is saved in case a live run fails
- [ ] You can point to the weight constants in the generated RTL and to the audit result
- [ ] You can open a waveform and explain one transaction
- [ ] You can state C1, C2, the fairness protocol, and the caveats without notes

## 7. Report outline and traceability
**Outline:** Abstract · 1 Introduction and motivation · 2 Background (transformer linear layers, INT8 quantization, constant-coefficient arithmetic, model-specific hardware) · 3 Problem statement, scope, non-goals · 4 System design (frontend, IR, quantization, specialization, tiling, backend, baseline) · 5 Verification methodology (oracle, levels, mutation, audits) · 6 Results (C1, C2, F-vs-P) · 7 Discussion and limitations · 8 Future work (Part II and III) · 9 Conclusion · Appendices: reproducibility, repo map, ADR list, AI-assistance disclosure.

**Traceability table (maintain in `docs/report_traceability.md`):**
| Report item | Value | Source file | Producing command | Commit |
Every number in the report appears once in this table. If it is not here, it does not go in the report.

**Report checks:** [ ] claims match what the data supports (B5) · [ ] extrapolations labeled as estimates · [ ] negative or mixed findings stated · [ ] limitations include: Part I covers linear layers only, no dequantization hardware, no place-and-route, proxy metrics only · [ ] figures reproducible

## 8. Viva preparation
- [ ] I can explain every pipeline stage and its file handoff
- [ ] I can explain the numeric contract and why each choice was made
- [ ] I can explain why the verification can be trusted (independent oracle, mutation, audits)
- [ ] I can explain the F-vs-P fairness protocol and what the results do not show
- [ ] I can say what Part II would add and what it would change in the IR and backend
- [ ] I can run the demo and inspect a waveform unaided
