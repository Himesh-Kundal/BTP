# 03 — Decisions, Change Control and Risks

## 1. Frozen decisions
| ID | Decision | Why |
|---|---|---|
| D-01 | INT8 symmetric; weights per-output-channel, activations per-tensor static; range ±127; round-half-even. | Per-channel costs nothing in RTL (scales applied after accumulation) and improves accuracy; static scales keep hardware simple. |
| D-02 | INT32 accumulator at the interface; compiler proves a bound and records the minimal width. | Plan requires INT32; worst case for K=1536 needs ~26 bits, so overflow is impossible by construction. |
| D-03 | RTL outputs the raw INT32 accumulator; dequantization lives in Python. | Keeps C1 exact and separate from C2 quality. |
| D-04 | Generated RTL is a Verilog-2001 subset; testbenches are plain Verilog. | Widest tool compatibility; fewer signedness surprises. |
| D-05 | Icarus is the primary simulator; Verilator for lint and larger runs; **Yosys is required** (the F-vs-P comparison is a core result, not an optional extra). | The plan lists Yosys as optional, but without it the "advantage" claim cannot be evaluated. |
| D-06 | Core comparison is F (spatial fixed) vs P (programmable baseline). ROM/time-multiplexed mode is stretch. | Keeps scope small while still answering RQ2. |
| D-07 | Single model: SmolLM2-135M, pinned revision. The compiler is shape-generic and tested on synthetic shapes. | Scope control without hardcoding. |
| D-08 | Frontend is separate from the compiler; handoff via files. | Plan requirement; enforced by the layering test. |
| D-09 | Seven projection types are covered, including `gate_proj`. | The plan listed up/down only; SwiGLU's MLP needs gate. |
| D-10 | Default tile 8×16; the evaluation grid varies it. | Plan's starting configuration. |
| D-11 | Expected outputs come from Python-written hex files, never from the testbench. | Independence of the oracle. |
| D-12 | Tests, golden files and tolerances are frozen once merged. | Prevents "fix the test" failure mode in agent work. |
| D-13 | Phase order changes from the plan: walking skeleton first, then deepen; verification is built from week 1, not weeks 11–12. | Late verification is where agent-built projects fail. |
| D-14 | Fair-comparison protocol in `docs/06` §B is mandatory for every synthesis claim. | Avoids accidental unfairness (accumulator widths, I/O registers, weight storage accounting). |

## 2. Open decisions (human to resolve; defaults apply until then)
| ID | Question | Default if unanswered |
|---|---|---|
| O-1 | Confirm F-vs-P as the core comparison and time-multiplexed ROM as stretch (the original plan's §7.5 describes a reused compute unit with a coefficient store). | As in D-06. |
| O-2 | Supervisor's report format, page limit, and deadlines for mid-term and final. | None assumed; add to STATE.md. |
| O-3 | Area technology for Yosys: generic gates only, FPGA LUT proxy (ice40/xilinx), or an open standard-cell liberty (e.g. sky130). | Generic gates + ice40 LUT proxy; liberty only if a file is available. |
| O-4 | Is stretch S1 (full-model Python simulator) wanted? | Not started until G6 passes. |
| O-5 | Which agent harness runs this (shell/subagent/permission capabilities)? | Single-agent with separate sessions per role. |
| O-6 | Institute policy on AI assistance and disclosure. | Disclose in the report appendix. |

## 3. Change control (ADR)
Any change to `docs/02`, a frozen test, a tolerance, or scope needs an entry here, approved by the human:

```
### ADR-<n>: <title>            Date: <YYYY-MM-DD>   Status: proposed | accepted | rejected
Context: what forced the question
Options: A, B, (C) with trade-offs
Decision: chosen option
Consequences: what must change (docs, code, tests, results to regenerate)
Approved by: <human>
```

## 4. Risk register
| ID | Risk | Likelihood / impact | Mitigation | Early signal |
|---|---|---|---|---|
| R1 | Agent games the checks (weak tests, self-computed expectations, fabricated results). | M / H | Independent oracle, frozen tests, mutation suite, testbench negative control, verifier role, evidence rule. | Tests green but mutation survivors; numbers without a `results/` source. |
| R2 | Signedness and width bugs in generated Verilog. | H / H | Conventions in `docs/02` §6, one-hot and boundary vectors, mutation M7/M6, cross-simulator check. | Mismatches only on negative inputs. |
| R3 | Simulation time or file size explodes for full layers. | H / M | Verification levels L0–L4 (full layers only in Verilator, optional), slice testing, size budgets measured at T4.5. | Compile time > a few minutes. |
| R4 | Yosys time/memory blow-up on large tiles. | M / M | Grid capped, timeouts, extrapolation clearly labeled, one validated larger point. | Synthesis run > timeout. |
| R5 | Per-tensor INT8 activations are poor in some layers (outliers). | M / M | Report honestly (C2), percentile-calibration option, do not tune to hide it. Not an RTL problem. | SQNR very low for down_proj or similar. |
| R6 | Unfair F-vs-P comparison produces a misleading headline. | M / H | Fairness protocol, two accounting views, multiple seeds, shuffled-weight control, caveats paragraph. | Result "too good" or sensitive to tool flags. |
| R7 | Scope creep (stretch items, extra operators). | H / M | Scope tiers, parking lot, freeze at end of W12. | Tasks not in `docs/04`. |
| R8 | Context loss between agent sessions. | H / M | `STATE.md`, LOG, handoff notes, short AGENTS.md, task files as the unit of work. | Agent re-deriving past decisions. |
| R9 | Tool-version drift (iverilog vs Verilator semantics). | M / M | Pin versions, record in `results/env.json`, cross-simulator equality test. | Different outputs on the two simulators. |
| R10 | Student cannot explain the work at the viva. | M / H | Walkthrough docs per phase and explain-it questions (`docs/05` §8). | Student skipping gate reviews. |
| R11 | Model download or HF API changes. | L / M | Pin revision, cache weights, record hashes. | Hash mismatch. |
| R12 | Result is negative (fixed is not smaller). | M / L | Hypotheses framed as questions; negative results are reported and analyzed. | None; plan for it. |
