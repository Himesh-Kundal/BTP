# 06 — Verification Strategy and Evaluation Protocol

# Part A — Verification

## A1. What must be proven
- **C1:** generated RTL output equals the integer reference exactly (bit-exact).
- **Fixed-weight property:** generated F has no weight port or write path, and its constants are exactly `W_q`.
- **The verification itself works:** deliberately broken designs fail.
- **C2** (quantization quality) is measured, not asserted exact; see `docs/02` N9.

## A2. Oracle and independence
`verification/reference.py` is numpy int64 and is the only source of expected values. It is reviewed by hand against small cases (T0.3) and never imports from `backend/`. Expected values reach the testbench only through `y_exp.hex` files. Testbenches never compute results. Golden files in `verification/golden/` are frozen.

## A3. Verification levels
| Level | What | Simulator | Frequency |
|---|---|---|---|
| L0 | Toy: 1×3 example, 2×4 | Icarus | every `make check` |
| L1 | Synthetic shapes and weights: N ∈ {1,3,4,8,9}, K ∈ {1,5,16,17,33}, tiles ∈ {(1,1),(4,16),(8,16),(8,8)} | Icarus | `make check` (subset), `make regression` (all) |
| L2 | Real SmolLM2 tile (8×16 or 16×64) with captured activations | Icarus | regression |
| L3 | Real slice with full K (e.g. 64×576, 64×1536) for all 7 types, first/middle/last layer | Icarus or Verilator | regression |
| L4 | Full layer (optional) | Verilator only | on demand, before G8 |

Budgets for L1–L4 are measured at T4.5/T6.1 and frozen.

## A4. Input classes (each applies at every level)
1. Uniform random int8 in [−128, 127].
2. Uniform random within the quantizer range [−127, 127].
3. All −128; all 127; all 0.
4. **One-hot** per lane (value 1 at lane i): output equals column i of W, so every weight is checked directly and lane swaps are caught.
5. Sign-aligned worst case: `x_k = 127·sign(w_k)` and `−128·sign(w_k)` to approach accumulator extremes.
6. Alternating signs; sparse random.
7. Real captured activations quantized with the IR's `act_scale`.

## A5. Weight classes
Uniform random [−127,127] · extremes (±127) · all zero · single nonzero · sparse · near-Gaussian · real layer slices. Include one case with a bias vector and one with `b_q` near ±2²⁰ to confirm bias handling.

## A6. Mutation suite (`verification/mutate.py`)
Text-level faults applied to generated RTL; the standard test set for that design must FAIL for each.
| ID | Mutant | Typically caught by |
|---|---|---|
| M1 | Flip sign of one weight constant | any random vector |
| M2 | Change one weight by +1 | one-hot vectors |
| M3 | Drop one product term | one-hot / random |
| M4 | Swap two input lanes | one-hot |
| M5 | Reverse packing order of outputs | any |
| M6 | Truncate accumulator to 16 bits | worst-case vectors |
| M7 | Make one multiplication unsigned | negative inputs |
| M8 | Off-by-one in padding (last lane misrouted) | non-multiple shapes |
| M9 | Add one cycle of latency | timing check |
Requirement: kill rate 100%. A surviving mutant means the tests are weaker than assumed; fix the tests, never remove the mutant.

## A7. Structural audits
- **Interface test:** parse generated F: ports ⊆ {clk, rst_n, valid_in, x_flat, valid_out, y_flat}; no `w_*` ports; no `reg` arrays written by a weight path. Must fail when a weight port is injected.
- **Weight audit** (`verification/weight_audit.py`): parse constants from RTL and compare with `weights_q.npy` by value and by sha256 in the manifest.
- **Determinism:** generate twice; byte-identical.
- **Lint:** `iverilog -g2005 -Wall`, `verilator --lint-only -Wall`, Yosys `hierarchy -check; proc; check -assert` all clean.

## A8. Testbench rules
- Reads vectors with `$readmemh`, drives the DUT, compares to `y_exp`.
- Prints exactly one summary line: `TB_RESULT CHECKED=<n> MISMATCH=<m> TIMING_ERR=<t>`.
- Runner passes only if `CHECKED == expected`, `MISMATCH == 0`, `TIMING_ERR == 0`.
- **Negative control:** a corrupted `y_exp.hex` must produce FAIL. Runs in `make check`.
- A `+dump` plusarg writes a VCD.

## A9. Cross-checks
- `Y_F == Y_P == Y_ref` on identical vectors.
- Icarus and Verilator outputs identical.
- Integer reference vs float reference: C2 metrics per layer type, depth, and calibration option.

## A10. Failure triage (for the agent)
1. Reproduce with the smallest shape (N=1, K=1) and a one-hot vector.
2. Compare packing: check `x_flat`/`y_flat` element order first.
3. Check signedness: any unsigned literal or implicit extension.
4. Check padding lanes.
5. Dump VCD and compare against the reference for that vector.
6. Only after the cause is understood, fix the generator, never the expected file.

# Part B — Synthesis Evaluation (F vs P)

## B1. Questions
RQ2 and hypotheses H1–H3 in `docs/01` §3.

## B2. Fairness protocol (mandatory, D-14)
1. Same Yosys script, same flags, same top-level flatten for F and P.
2. Same widths: x 8, w 8, product 16, accumulator width as in the IR (N10).
3. Same wrapper (registered I/O). F reported with the identical I/O registers as P.
4. P is reported under two accountings: **(a) logic-only** (`P_bus`: weights as primary inputs, no storage) and **(b) with weight register storage**. Both are shown; the conclusion states which one it relies on and why. Note that real accelerators keep weights in SRAM that is not counted in logic area.
5. Flops are always reported separately from combinational cells.
6. Random-weight experiments use ≥ 5 seeds; report mean ± sd.
7. Tool versions and flags are recorded in `results/env.json` and each result file.
8. Every claim carries the caveat: pre-layout logic-synthesis proxy; no placement, routing, memory macros, or power.

## B3. Metrics
- Cell counts by type from `stat` (generic gates), flops separate.
- LUT count (ice40 or xilinx flow) as an FPGA-style proxy.
- Gate-equivalent area if a liberty file is provided (O-3).
- Longest topological path (`ltp`) as a depth proxy, with the caveat that it is not STA.
- Wall time of synthesis, for the budget.

## B4. Experiments
| ID | Question | Design |
|---|---|---|
| E1 | How does area scale with tile size? | F, P, P_bus over a tile grid such as (4,16), (8,16), (16,16), (16,64), (32,64); real weights from several layer types; capped by timeout |
| E2 | Does weight content matter for F? | Same tile with: real weights; uniform-random INT8; random with matched magnitude histogram; shuffled real weights (same distribution, no structure); all-zero; one-hot |
| E3 | Is the depth proxy comparable? | `ltp` for E1 designs |
| E4 | Quantization quality | C2 metrics per type and depth (from T3.6); not a synthesis experiment, reported next to E1–E3 for context |
| E5 | Can we extrapolate to a full layer? | Linear model per-tile cost × tile count (and a sparsity-adjusted variant); validate against one directly synthesized larger point; report the error of the estimate; label every extrapolated number "estimate" |

## B5. Interpreting results
- If F is smaller: report by how much, for which tile sizes and weight classes, and what drives it (zero weights, low bit-weight, shared adders).
- If F is not smaller or the result is sensitive to flags: report that as the finding and analyze why. Do not tune flags to obtain a preferred outcome; if flags are changed, rerun both designs and record why.
- Never extrapolate beyond the validated range without saying so.
- Keep two lists in the report: what the data supports and what it only suggests.

## B6. Outputs
`results/E1.csv … E5.json`, plots produced by `make report-figs`, a one-page `results/synthesis_summary.md`, each with producing command, commit hash, tool versions.
