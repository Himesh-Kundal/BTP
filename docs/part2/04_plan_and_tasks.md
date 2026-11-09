# Part II / 04 — Plan and Task Backlog

Only tasks in this file may be worked on. Each becomes `tasks/U<id>.md` (template in Part I `docs/05` §6) when it starts. IDs: tasks `U<phase>.<n>`, gates `H<x>`. "Done when" is executable wherever possible. Tags `rtl`, `numeric`, `gate` require a verifier pass.

## 1. Timeline (14 weeks; set the start date in STATE.md)
| Phase | Weeks | Goal | Gate |
|---|---|---|---|
| U0 Audit and bootstrap | 1 | Reproduce Part I, retrospective, model facts, budgets, tiny model | H0 |
| UA Numerics first | 1–3 | Integer decoder, quality sweeps, frozen formats, golden traces. **May start before the semester.** | HA |
| UB Compiler extension | 3–5 | Graph IR, formats, ROM packing, schedule, memory plan, artifact interpreter | HB |
| UC RTL unit library | 4–7 | Every unit bit-exact, mutation-tested | HC |
| UD Tiny model end to end | 6–8 | Token-exact tiny transformer in Icarus and Verilator | HD |
| UE Full-model scale-up | 8–10 | Layer 0, then 30 layers, token-exact | HE |
| UF Architecture and synthesis study | 10–11 | Units, engine grid, memory split, block comparison, ASIC projection | HF |
| UG Integration, report, viva | 12–13 | CLI, demo2, whole-project report, slides | HG |
| Buffer | 14 | Fixes and rehearsal. **No new features after the end of W12.** | — |

Phases overlap by design: UC starts when the numerics for the first units are frozen (per-op freezing inside HA is allowed via ADR), and UB design work runs alongside UA. The fallback decision (ADR-2-F, §3) is due before the end of W9.

## 2. Tasks

### U0 — Audit and bootstrap
| ID | Task | Deps | Done when |
|---|---|---|---|
| U0.1 | Reproduce Part I from a fresh clone: `make check` (record test count), `make demo`, tool versions; audit Part I deliverables against Part I `docs/07` G0–G8; tag `part1-final`; archive Part I STATE as `state/archive/part1_STATE.md` | — | `results/part2/part1_audit.md` lists every deliverable with evidence or gap; fresh-clone run passes |
| U0.2 | Retrospective: from Part I LOG/BLOCKERS, list what cost the most time and what checks caught real bugs; propose process trims | U0.1 | ADR-2-0 proposed; human decision recorded |
| U0.3 | Model facts v2 from the real config and weights: vocab, tie, rope theta, max positions, eps, activation, head layout, shapes; reconcile with `02` §2 | U0.1 | `docs/model_facts.md` updated; discrepancies raised as ADRs |
| U0.4 | Part II skeleton and Makefile targets; env check (Verilator ≥ 5, `--binary --timing`); **measure budgets**: Verilator cycles/s for a ~1000-MAC design, build time/RAM for a large ROM array, RTL size per weight from Part I outputs (extrapolate to the full model), disk per run | U0.1 | `results/part2/budgets.json`; `make check2` green; D2-02 feasibility number confirmed |
| U0.5 | Tiny model: HF-style random-weight transformer (d=64, L=2, 4 heads / 2 KV, FFN 128, vocab 256, ctx 32, tied embeddings, same op set) generated deterministically and flowing through the Part I frontend interfaces | U0.3 | tiny model artifacts + fp32 forward available; layering test green |
| U0.G | **Gate H0** | all U0 | gate report + verifier PASS |

### UA — Numerics first (Python only)
| ID | Task | Deps | Done when |
|---|---|---|---|
| UA.1 | `eval/fp32/` NumPy fp32 decoder implemented from op definitions (RMSNorm, rotate-half RoPE, GQA, SwiGLU, tied head); validated against HF | U0.3 | logits max-abs diff ≤ 1e-3 and argmax identical on ≥ 20 prompts; tiny model too |
| UA.2 | Evaluation harness: fixed WikiText-2 windows + prompt battery; metrics perplexity, KL(fp32‖int), top-1 agreement, greedy match length; dataset hashes pinned | UA.1 | `make numerics-eval` reproducible (same hashes); runtime recorded |
| UA.3 | `numerics/ops.py`: `rshift_rnd`, `sat`, `requant`, LUT builders (exp2 fraction, rsqrt mantissa, reciprocal, SiLU), with edge-case tests | U0.4 | tests incl. negatives, ties at half, saturation, zero shift; exhaustive on small widths |
| UA.4 | Integer decoder v0 (KV cache, greedy, all ops) built from `numerics/ops.py` and the Part I linear reference; step function and a vectorized teacher-forcing scorer | UA.3 | scorer equals step-by-step decoder bit-exactly on ≥ 50 sequences |
| UA.5 | Baseline quality: defaults from `02` §3; **per-op ablation** (swap each op between fp32 and integer, one at a time) to attribute loss; range/saturation statistics per layer | UA.2, UA.4 | `results/part2/ua5_ablation.csv`; saturation counts reported |
| UA.6 | Sweeps EN2–EN7 (`06` §B1) with cost proxies; every run logged, none deleted | UA.5 | `results/part2/numerics_sweep.csv` complete; plots script |
| UA.7 | Choose the config: Pareto table, ADR with frozen quality thresholds and the final format table; update `02` §3 from (D) to frozen | UA.6 | ADR-2-1 accepted by the human; `numerics/config_frozen.json` hashed |
| UA.8 | Golden traces: tiny model all tensors/tokens; SmolLM2 layer 0 all tokens plus sampled layers/tokens; `trace_diff.py`; MANIFEST with hashes | UA.7 | traces regenerate with identical hashes; trace_diff tested on injected faults |
| UA.9 | Hardware cost proxies per config: ROM bits (weights, tables, requant params), widest multiplier, accumulator and buffer widths, LUT bytes | UA.7 | `results/part2/cost_proxies.json` |
| UA.G | **Gate HA** | all UA | gate report + verifier PASS (`numeric`) |

### UB — Compiler extension
| ID | Task | Deps | Done when |
|---|---|---|---|
| UB.1 | Graph IR v2 (ops, tensor kinds, formats), JSON/hex serialization, invariants I1–I7 | U0.5, UA.7 | invariant tests; round-trip lossless |
| UB.2 | Frontend v2: extract the full structure (embedding, norms, projections, tied head, RoPE params, eps) into artifacts with sha256 | U0.3 | reload tests; tiny and SmolLM2 artifacts |
| UB.3 | Format-assignment pass: apply frozen config and calibration to every edge; fold γ, 1/√d, per-channel scales into `(M, s)` | UB.1, UA.7 | per-edge format table; range checks against calibration traces |
| UB.4 | ROM packing pass: tile-ordered INT8 weight images (reuse Part I tiling and padding), requant-parameter images, table images; address map | UB.3 | images + `address_map.json`; sha256s; padding tests |
| UB.5 | Schedule/microprogram generator with layer loop and per-layer ROM base table; cycle estimate per op | UB.4 | program ROM image; estimate within the unit-level model |
| UB.6 | Memory planner: buffer sizing by liveness, KV layout, `MAX_CTX` selection, total RAM report | UB.5 | memory report for tiny and SmolLM2 |
| UB.7 | **Compiled-artifact interpreter**: executes schedule on packed ROM images and formats; must equal the `numerics/` decoder | UB.4–UB.6 | bit-exact on tiny (all tokens, all traces) and on SmolLM2 prompts (tokens + layer-0 trace) |
| UB.G | **Gate HB** | all UB | gate report + verifier PASS (`numeric`) |

### UC — RTL unit library (each: bit-exact tests, mutants, lint)
| ID | Task | Deps | Done when |
|---|---|---|---|
| UC.1 | `rtl/lib/numerics.vh` (`rshift_rnd`, `sat`, requant) and `requant` unit | UA.7 | exhaustive cross-test against `numerics/ops.py` on small widths; random/boundary at full width |
| UC.2 | ROM generator and ROM module template; **weight audit v2**; ROM interface test (no write path) | UB.4 | audit passes; fails on an injected write port or altered hex |
| UC.3 | `rmsnorm` unit | UC.1, UA.8 | bit-exact on random + adversarial (zeros, tiny, huge, equal elements); mutants |
| UC.4 | `rope` unit + table ROMs | UC.1, UB.4 | bit-exact for all positions `< MAX_CTX` and boundary values |
| UC.5 | `softmax` unit (max, exp2, sum, reciprocal, normalize) | UC.1 | bit-exact incl. single-position context, equal scores, extreme spreads, `pos = MAX_CTX−1` |
| UC.6 | `swiglu` unit (SiLU LUT + product + requant) | UC.1 | **exhaustive** over the SiLU input domain × representative `up` values |
| UC.7 | `residual_add` and `embed_gather` units | UC.1 | bit-exact; saturation cases |
| UC.8 | `argmax` unit with the per-row scale handling of N2-12 | UC.1 | adversarial scale tests; tie-break lowest id |
| UC.9 | `kv_cache` RAM and address generator (layer, kv head, pos) | UB.6 | read/write ordering tests; GQA mapping test |
| UC.10 | `linear_engine` R×C with ROM streaming and requant output | UC.1, UC.2 | passes Part I vectors reshaped to the engine and new non-multiple shapes; ROM order matches packing |
| UC.11 | `attention` datapath (score loop, softmax, weighted V, GQA) | UC.5, UC.9, UC.10 | bit-exact for several positions and head groups |
| UC.12 | `sequencer` + program ROM execution | UB.5 | executes the tiny program; op order equals the interpreter |
| UC.13 | Mutation suite v2 per unit (`06` §A6) | UC.3–UC.12 | kill matrix 100% |
| UC.G | **Gate HC** | all UC | gate report + verifier PASS (`rtl`, `numeric`) |

### UD — Tiny model end to end
| ID | Task | Deps | Done when |
|---|---|---|---|
| UD.1 | Top-level generator (composition, ROM instances, step interface) | UC.12, UB.7 | tiny top lints clean in iverilog, Verilator, Yosys `check` |
| UD.2 | Testbench generator: feed prompt tokens, generate N tokens, dump traces, summary line | UD.1 | `TB2_RESULT` produced; negative control (corrupt golden) FAILs |
| UD.3 | Token-exact and trace-exact on ≥ 100 random prompts incl. sequences up to `MAX_CTX` | UD.2 | all PASS; first-divergence tool exercised on an injected fault |
| UD.4 | Cross-simulator equality: Icarus vs Verilator traces identical | UD.3 | `diff` empty |
| UD.5 | Top-level mutation suite (causal off, kv-head map, pos off-by-one, rope table shift, residual scale, tie-break, ROM word swap) | UD.3 | 100% killed |
| UD.6 | Determinism and structural audits (no expected values in RTL, ROM-only weights) | UD.1 | tests green |
| UD.G | **Gate HD** | all UD | gate report + verifier PASS (`rtl`) |

### UE — Full-model scale-up
| ID | Task | Deps | Done when |
|---|---|---|---|
| UE.1 | Generate full SmolLM2 ROM images, program and top; measure generation time/size; fix scalability issues | HD | `results/part2/ue1_generation.json` within the U0.4 budgets |
| UE.2 | Verilator build of the full top; record build time, RAM, sim speed (cycles/s) | UE.1 | build succeeds; numbers recorded |
| UE.3 | Layer-0 block: trace equality for the first positions and several later positions | UE.2 | all traced tensors equal golden |
| UE.4 | **Full decode:** ≥ 10 prompts × ≥ 16 new tokens token-exact; sampled-layer trace equality; cycles/token measured | UE.3 | `results/part2/full_decode.json`; PASS |
| UE.5 | Long-context test up to `MAX_CTX` | UE.4 | token- and trace-exact at length |
| UE.6 | Host demo: prompt text → tokens → RTL sim → text; compare with fp32 HF greedy text; report first-divergence statistics | UE.4 | demo script; statistics table |
| UE.7 | Final quality numbers from the Python integer model on the full eval set; tie to RTL via UE.4 equality | UE.4, UA.2 | `results/part2/quality_final.json` |
| UE.G | **Gate HE** | all UE | gate report + verifier PASS (`rtl`, `numeric`) |

### UF — Architecture and synthesis study
| ID | Task | Deps | Done when |
|---|---|---|---|
| UF.1 | Yosys flow for each unit and the engine (generic gates; liberty if O2-4 provides one); flops separate; deterministic JSON | HC | `results/part2/EA1.csv` |
| UF.2 | Engine grid (R, C) ∈ a documented set: area vs lanes, depth proxy, analytic cycles/token, validated against simulated cycles | UF.1, UE.4 | `EA2.csv`; model error reported |
| UF.3 | Memory accounting: ROM bits (weights, tables, params, program), KV cache at `MAX_CTX`, buffers | UB.6 | `EA3.json` |
| UF.4 | Spatial (Part I) vs ROM-sequential for one block at a synthesizable size: area, cycles, throughput | UF.1 | `EA4.csv`; limits of extrapolation stated |
| UF.5 | ASIC projection: assumptions table with sources, low/mid/high ranges, sensitivity analysis; tokens/s and area vs lanes | UF.2, UF.3 | `EA5.json`; every number labeled "estimate" |
| UF.6 | Context section: published hardwired-LLM claims (vendor-reported vs independently measured) placed beside your numbers with caveats | UF.5 | text block with sources |
| UF.G | **Gate HF** | all UF | gate report + verifier PASS |

### UG — Integration, report, viva
| ID | Task | Deps | Done when |
|---|---|---|---|
| UG.1 | CLI v2: `compile-model`, `run`, `verify`, `synth` | HE | CLI tests; `--help` documented |
| UG.2 | `make demo2-fast` (tiny, minutes) and `make demo2` (full model, short generation) | UG.1 | both pass from a fresh clone; runtimes documented |
| UG.3 | README, architecture doc, limitations, reproduce-from-fresh-clone test | UG.2 | clean-directory run passes |
| UG.4 | Whole-project report (Part I + II) with revised Part I claims (e.g. share of per-token arithmetic in the 7 projections: ≈79% with LM head, >90% only for prefill), traceability table | HF, UE.7 | every number has a source row |
| UG.5 | Slides, speaker notes, viva Q&A; AI-assistance disclosure per institute policy; supervisor certificate page **prepared and signed by humans** | UG.4 | student answered all explain-it questions |
| UG.G | **Gate HG** | all UG | gate report + verifier PASS |

### Stretch (only after HE)
| ID | Task | Done when |
|---|---|---|
| V1 | Layer-pipelined multi-token throughput model | `results/part2/V1.json`, labeled estimate |
| V2 | Zero-weight skipping in the engine | passes full ladder; cycle/area delta measured |
| V3 | Weight bit-width study (W6/W4): ROM bits vs quality | table in `results/part2/V3.csv` |
| V4 | Top-k/temperature with deterministic LFSR | exact vs reference |

## 3. Cut list and fallback
**Cut in this order if behind:** V1–V4 → UF.6 → UF.4 → UE.5 → reduce evaluation-set size (keep pinned and reported) → UF.5 sensitivity detail.
**Never cut:** numerics-first (HA), the compiled-artifact interpreter, tiny-model token-exact RTL, weight audit v2, mutation suites, quality report, whole-project report.
**Fallback (ADR-2-F before the end of W9):** if full-model token-exact is unrealistic (sim speed, build size, or schedule), choose F1 (first L ≥ 4 real layers in RTL, boundary at a layer output, remainder in the Python integer model). Document what is proven and what is inferred.
