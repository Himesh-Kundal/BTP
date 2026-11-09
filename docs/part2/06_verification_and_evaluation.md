# Part II / 06 — Verification Strategy and Evaluation Protocol

# Part A — Verification

## A1. What must be proven
- **C1′ (exactness):** RTL equals the integer reference for every traced tensor and every output token.
- **Immutability:** every weight, table and parameter lives in a ROM with no write path, and the ROM contents equal the compiler's tensors.
- **The verification works:** deliberately broken designs fail, and a corrupted golden fails.
- **Separation of claims:** C2′ (integer vs fp32 quality) is measured and reported; it is never a RTL pass/fail criterion.

## A2. Oracles and their independence
| Oracle | Defines | Checked by |
|---|---|---|
| `numerics/` integer decoder | semantics of every op and of the whole decode step | hand-computed unit cases; scorer == stepper; exhaustive small-width tests |
| compiled-artifact interpreter (UB.7) | that compiler packing/folding/scheduling preserve semantics | equals `numerics/` bit-exactly |
| `eval/fp32/` NumPy fp32 decoder | the real-valued reference for quality | equals HF within 1e-3 logits |
| golden traces/manifests | expected tensors for RTL | frozen after HA; regenerated only by ADR |

Expected values reach testbenches only through files produced by these oracles. Testbenches never compute results.

## A3. Verification ladder
| Level | What | Simulator | Frequency |
|---|---|---|---|
| V0 | Single unit vs `numerics/` (exhaustive where domain ≤ 2²⁰; random + boundary + adversarial elsewhere) | Icarus | every `make check2` (subset), regression (all) |
| V1 | Engine, attention datapath, unit compositions (block level), Part I linear vectors reshaped | Icarus | regression |
| V2 | Tiny random model: token-exact and trace-exact over ≥ 100 prompts incl. contexts up to `MAX_CTX` | Icarus + Verilator | regression |
| V3 | SmolLM2 layer 0 full block: trace equality at several positions | Verilator | `make long-*` |
| V4 | Full 30-layer decode: ≥ 10 prompts × ≥ 16 new tokens token-exact; sampled-layer trace equality | Verilator | `make long-*` before HE and HG |
| V5 | Quality (integer vs fp32) on pinned sets; scorer vs stepper agreement; interpreter vs oracle | Python | `make numerics-eval` |

Budgets for V2–V4 are measured at U0.4/UE.2 and frozen. Prompt battery: pinned list in `eval/prompts.json` covering short/long prompts, repeated tokens, rare tokens, and prompts that reach `MAX_CTX`.

## A4. Unit test classes (V0)
For every unit: (1) hand-computed cases; (2) boundary values (min/max of every port, saturation edges, zero, one, negative odd values for rounding); (3) random with fixed seeds; (4) adversarial structure (RMSNorm: all zeros, one huge, equal elements, tiny values; softmax: one position, equal scores, huge spread, `pos = MAX_CTX−1`; argmax: ties, scale differences that flip order; RoPE: all positions; engine: non-multiples of R and C, padding lanes); (5) exhaustive over the full input domain whenever ≤ 2²⁰ points (SiLU LUT, small-width `rshift_rnd`/`requant`, exp2 fraction table, reciprocal/rsqrt mantissa tables).

## A5. Trace equality
Golden traces are compared by `verification/trace_diff.py` (first diverging token/layer/op/tensor/element). Tiny model: every tensor of every token. SmolLM2: layer 0 for all tokens; sampled layers and tokens elsewhere (seeded, listed in the manifest); every output token always.

## A6. Mutation suite v2 (`verification/mutate2.py`)
Text-level faults applied to generated or library RTL; the standard test set for that level must FAIL for each.
| ID | Mutant | ID | Mutant |
|---|---|---|---|
| M10 | requant rounds half-down | M21 | SiLU LUT sign error |
| M11 | saturate to −128 / no saturation | M22 | residual add skipped in one layer / wrong order |
| M12 | RoPE pairs adjacent elements | M23 | argmax ties to highest id |
| M13 | RoPE table index off by one | M24 | argmax compares raw per-row accumulators |
| M14 | K stored before RoPE | M25 | ROM word swap / wrong base for one layer |
| M15 | GQA map `i % 3` instead of `i // 3` | M26 | KV write at `pos+1` or overwriting |
| M16 | causal mask off (attends `pos+1`) | M27 | embedding row index off by one |
| M17 | softmax without max subtraction | M28 | engine accumulator truncated |
| M18 | softmax LUT index off by one | M29 | engine padding lane nonzero |
| M19 | rsqrt LUT index wrong | M30 | extra latency / handshake break |
| M20 | RMSNorm skips γ | | |
Requirement: 100% kill. A survivor means the tests are weaker than assumed; fix the tests, never drop the mutant. Part I mutants M1–M9 still run on the linear engine.

## A7. Structural audits
- **Weight audit v2:** ROM hex images == quantized tensors (value and sha256); address map consistent with packing; table hex == `numerics/` builders.
- **ROM interface test:** ROM modules have only `clk`, `addr`, `data`; no write enable anywhere reaching them. Fails on an injected write port.
- **No-expected-values audit:** generated RTL/hex contain nothing derived from tokens, activations or expected outputs.
- **Determinism:** generate twice, byte-identical.
- **Lint:** `iverilog -g2005 -Wall`, `verilator --lint-only -Wall`, Yosys `hierarchy -check; proc; check -assert`.

## A8. Testbench rules
Drives the step interface, dumps trace taps (hierarchical), prints one `TB2_RESULT` line (see `02` §8). The runner passes only if counts equal expectations and all mismatch counters are zero. **Negative control:** a corrupted golden must FAIL (runs in `make check2` on the tiny model).

## A9. Cross-checks
scorer == stepper (≥ 50 sequences) · interpreter == `numerics/` · Icarus == Verilator (tiny model, full trace) · engine reproduces Part I linear vectors · RTL cycle counts vs the analytic cycle model (error reported).

## A10. Failure triage
Follow the divergence protocol in `AGENTS.md`; record in the bug notebook; add a regression test and, if general, a mutant.

# Part B — Evaluation

## B1. Numerics experiments (UA)
Evaluation fixed first: WikiText-2 test windows (count and length pinned) + prompt battery; all hashed.
**Metrics:** perplexity (teacher-forced, same code path as the decoder), mean KL(fp32‖int) per token, top-1 agreement, greedy match length from fp32 (mean/median over prompts). Report deltas with 95% bootstrap confidence intervals over windows.
| ID | Experiment | Variable |
|---|---|---|
| EN1 | Per-op ablation | swap one op at a time between fp32 and integer to attribute loss |
| EN2 | Residual width | {16, 20, 24, 32} bits, per-layer vs global scale; saturation counts |
| EN3 | Linear-input precision and calibration | INT8 vs wider at selected layer types; absmax / percentile / MSE statistics |
| EN4 | SmoothQuant-style migration folded into constants | α grid; folded into γ and next weights (zero runtime cost) |
| EN5 | RMSNorm internals | rsqrt LUT bits, Newton iterations, accumulator width |
| EN6 | Softmax internals | exp fraction LUT bits, probability bits, reciprocal LUT bits |
| EN7 | RoPE table width | Q-format of cos/sin |
| EN8 | (Stretch V3) weight bits | W8 vs W6/W4 |
Each run records: config hash, metrics, cost proxies (ROM bits, widest multiplier, LUT bytes, buffer widths), runtime, commit. Selection (UA.7): Pareto of quality vs cost proxies; thresholds fixed by ADR from the data; chosen config frozen.

## B2. Architecture experiments (UF)
Fairness rules from Part I `docs/06` B2 apply (same script and flags, identical widths and I/O boundaries, flops separate, ≥ 5 seeds where random, tool versions recorded). Additional rules: ROM/SRAM bits are reported separately from logic; every projection carries its assumptions.
| ID | Question | Design |
|---|---|---|
| EA1 | Cost of each unit | Yosys per unit and engine: cells by type, flops, LUTs (FPGA proxy), gate-equivalents if a liberty is available, `ltp` depth proxy |
| EA2 | Area and cycles vs lanes | Engine grid (R, C); analytic cycles/token = Σ over matrices of `ceil(N/R)·ceil(K/C)` + non-linear unit cycles + fill/drain; validated against simulated cycles; report model error |
| EA3 | Where the memory is | ROM bits (weights, tables, requant params, program), KV cache at `MAX_CTX`, buffers |
| EA4 | Spatial vs ROM-sequential | One block (e.g. an MLP slice) at a synthesizable size, both architectures from the same IR: area, cycles, throughput; extrapolation labeled |
| EA5 | ASIC projection | Total area ≈ ROM bits × a_rom + SRAM bits × a_sram + logic gates × a_gate, times an overhead factor; tokens/s = f_clk / cycles per token. Inputs are ranges (low / mid / high) with sources; show sensitivity (which assumption moves the answer most) |

## B3. Interpreting and reporting
- State which claims are **measured** (simulation, synthesis of generated RTL) and which are **estimated** (everything in EA5, any extrapolation).
- Published hardwired-LLM figures are context only: distinguish vendor-reported from independently measured, and never compare across different models, precisions, context lengths or batch conditions without saying so.
- If the ROM-based engine is slower or larger than expected, or the integer model loses more quality than hoped, report it with the analysis. Do not tune flags or thresholds after seeing a disappointing result; if something changes, rerun everything affected and log why.
- Keep two lists in the report: what the data supports, and what it only suggests.
- Mandatory caveat on all synthesis results: pre-layout logic-synthesis proxy; no placement/routing, memory macros, clock tree, or power.

## B4. Outputs
`results/part2/` — `EN*.csv`, `EA*.csv/json`, `quality_final.json`, `full_decode.json`, `part1_audit.md`, `budgets.json`, plots via `make report-figs2`, a one-page `results/part2/summary.md` listing each headline number with producing command, commit and tool versions.
