# Part II / 03 — Decisions, Change Control and Risks

## 1. Frozen decisions
| ID | Decision | Why |
|---|---|---|
| D2-01 | Part II ends the whole project. ASIC physical design, tape-out and FPGA bring-up are out of scope; the ASIC appears only as a labeled projection (EA5) and as future work. | Keeps the scope finishable in one semester. |
| D2-02 | Full-model architecture is a **ROM-backed, time-multiplexed engine** (R×C lanes). Part I's spatial generator is kept for block-level comparison. **Needs your confirmation (O2-1).** | Fully spatial full-model RTL is multiple GB of source (§01 §3). This reverses Part I's choice to leave a ROM-based design out; Part I compared only spatial vs programmable. |
| D2-03 | **Numerics first.** Integer semantics for every op are defined and measured in Python (`numerics/`) and frozen at gate HA before any RTL for that op. | The biggest risk is quality, and RTL cannot fix it; finding it early is cheap. |
| D2-04 | One oracle. `numerics/` defines semantics; the compiler's packing and folding are checked by the compiled-artifact interpreter (UB.7) before RTL exists. | Separates "wrong spec/packing" from "wrong RTL". |
| D2-05 | Greedy decoding only; ties go to the lowest token id. | Determinism and exact comparison. |
| D2-06 | Tokenizer and detokenizer stay on the host. The accelerator's interface is token ids. | Out of scope; not a hardware problem. |
| D2-07 | Weights, tables, requant parameters and the program are immutable ROM modules; only KV cache and activation buffers are writable RAM. | Preserves the fixed-weight thesis at full scale. |
| D2-08 | No floating point in RTL or the integer reference path. | Hardware realism and exact comparison. |
| D2-09 | Prefill is repeated decode steps. No special prefill datapath. | Scope control. |
| D2-10 | Verification ladder V0–V5, trace equality, mutation testing, structural audits carry over from Part I and are extended (see `06`). | Part I's methodology worked; reuse it. |
| D2-11 | Frozen-file rules carry over; after HA, the numeric spec, golden traces and quality thresholds are frozen too. | Prevents "tune the oracle until it passes". |
| D2-12 | RTL equals the integer reference exactly (C1′). The gap between the integer model and fp32 is a measured quality property (C2′), never a pass/fail criterion for RTL. | Keeps the two claims separate and honest. |
| D2-13 | Fallback ladder F0/F1/F2 is invoked only by an ADR before the end of W9. | Prevents late scope collapse. |
| D2-14 | Synthesis fairness protocol (Part I `docs/06` B2) applies, plus: ROM/SRAM bits are reported separately from logic and never mixed into gate counts; every projection lists its assumptions. | Avoids misleading area claims. |
| D2-15 | Large ROM images load from hex files; no giant generated `case` constants. | Tool scalability (Verilator build time and memory). |

## 2. Open decisions (human to resolve; defaults apply until answered)
| ID | Question | Default |
|---|---|---|
| O2-1 | Confirm D2-02 (ROM-backed engine for the full model). The alternative is a spatial-only Part II limited to a few layers of the real model. | D2-02 |
| O2-2 | Quality acceptance thresholds for the integer model (perplexity increase, top-1 agreement). | Set at HA from measurements, then frozen. Supervisor input welcome before HA. |
| O2-3 | Does the supervisor expect any FPGA or ASIC-flow content beyond Yosys synthesis and the projection? | No. |
| O2-4 | Area technology for EA1/EA5: generic gates only, or an open liberty (e.g. sky130). | Generic + liberty if a file is provided. |
| O2-5 | Evaluation corpus. | WikiText-2 test, fixed windows, plus a small hand-written prompt battery. |
| O2-6 | `MAX_CTX` for demos and tests. | 256 (set at U0.4/UB.6 from memory and sim-time budgets). |
| O2-7 | One combined whole-project report, or a Part II report that references Part I? | Combined, with Part I chapters revised. |
| O2-8 | Harness capabilities (subagents, long-running jobs, RAM/disk). | Single agent, separate sessions per role, background jobs via `make long-*`. |
| O2-9 | Part II start date; whether Phase UA starts early (winter break head start). | UA may start before semester; RTL work starts with the semester. |

## 3. Change control (ADR)
Any change to `02`, a frozen numeric choice, a frozen test, a threshold, scope or schedule needs an entry, approved by the human:
```
### ADR-2-<n>: <title>          Date: <YYYY-MM-DD>   Status: proposed | accepted | rejected
Context:        what forced the question, with links to results/
Options:        A, B, (C), each with measured trade-offs
Decision:       chosen option
Consequences:   what changes (docs, code, tests, results to regenerate)
Approved by:    <human>
```

## 4. Risk register
| ID | Risk | L / I | Mitigation | Early signal |
|---|---|---|---|---|
| RP1 | **Quality collapse** of the integer model (static INT8 activations, small 135M model). | H / H | Numerics-first, per-op ablation, residual width, mixed precision on a few layers, SmoothQuant-style folding into constants, percentile calibration; report honestly. | Perplexity jumps or top-1 agreement < expectation at UA.5. |
| RP2 | Residual-stream outliers ("massive activations") break narrow formats. | M / H | Wide residual container, sweep widths, per-layer scale; check ranges per layer in UA. | Saturation counts > 0 in traces. |
| RP3 | Non-linear approximations (softmax, RMSNorm) drift. | M / M | LUT sizing sweeps, exhaustive unit tests, per-op ablation. | One op dominates the quality loss. |
| RP4 | Verilator full-model decode too slow. | M / H | Measure early (U0.4); choose lanes accordingly; short sequences; overnight runs; trace sampling. | Cycles/s × tokens exceeds the budget. |
| RP5 | Verilator build time/memory with large ROM arrays. | M / M | Hex-file ROMs; split ROM modules; avoid `case` blobs; measure at UE.2. | Build > budget or OOM. |
| RP6 | Rounding/saturation mismatch between Python and RTL. | H / H | Single numerics spec, shared `numerics.vh`, exhaustive cross-tests on small widths. | Off-by-one mismatches on odd values or negatives. |
| RP7 | Scope explosion (many new units). | H / M | Scope tiers, reuse Part I, units small and independent, cut list, freeze at end of W12. | Tasks outside `04`. |
| RP8 | KV cache / GQA / position bugs. | M / H | Tiny model with many positions, mutants (causal off, kv-head map, pos off-by-one), trace diff. | Divergence only at longer contexts. |
| RP9 | Schedule/sequencer complexity. | M / M | Layer-loop microprogram; compiled-artifact interpreter as sequencing oracle. | Interpreter and RTL disagree on op order. |
| RP10 | Debugging a divergence deep in 30 layers. | H / M | First-divergence trace tool; bisect on tiny model; layer-0 block tests. | Hours spent per divergence. |
| RP11 | Argmax over per-channel-scaled logits compares incomparable numbers. | M / H | N2-12 spec; unit test with adversarial scales. | Token mismatches only for rare tokens. |
| RP12 | Timeline: Part II is bigger than Part I. | H / H | Numerics head start, cut list, fallback ladder, buffer weeks, freeze W12. | Gate slips > 1 week. |
| RP13 | Student cannot explain a much larger system. | M / H | Per-gate walkthroughs, explain-it questions, numerics decision log. | Skipping gate reviews. |
| RP14 | Part I defects or overstated claims found in the audit. | M / M | U0.1 audit; fix or document; claims in the final report re-derived from `results/`. | Audit finds gaps. |
| RP15 | Text from RTL differs from fp32 HF text and confuses demos. | H / L | Report agreement metrics; never promise identical text; show first-divergence position. | Early token divergence. |
| RP16 | ASIC projection read as a prediction. | M / H | Assumptions table, ranges, sensitivity, "estimate" labels, caveats paragraph. | Single-number claims. |
| RP17 | Agent gaming (golden-trace copying, oracle edits, weak tests). | M / H | Rules 2, 4, 9; structural audit; mutation suite; negative controls; verifier role. | Green tests with surviving mutants. |
