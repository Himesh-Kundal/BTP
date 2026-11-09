# Part II / 07 — Checklists

Use literally. Each item must be answerable with a command, a file or a link.

## 1. Task Definition of Done (every task)
- [ ] Each acceptance bullet verified by a command that was actually run
- [ ] Tests written first, mapped to acceptance bullets
- [ ] `make check2` green; ruff clean; no new warnings from iverilog/verilator on touched RTL
- [ ] No test, tolerance, golden, frozen config or oracle changed without an ADR
- [ ] No oracle values in generated RTL; no model-specific constants in `ir/ compiler/ backend/ numerics/ verification/`
- [ ] Generated outputs deterministic (rerun diff empty)
- [ ] Docs updated if behaviour, format or interface changed
- [ ] Task file Evidence filled; LOG line appended; STATE updated (current, next 3, blockers, handoff, budgets)
- [ ] Commit `U<id>: …`; merged only after green
- [ ] Tagged `rtl`/`numeric`/`gate`: verifier PASS
- [ ] Long-job tasks: logs, timeouts and resumability in place; results written with config hash and commit

## 2. Session start / end
**Start:** `git status` understood · `make check2` green · STATE and task file read · spec sections read · budgets in STATE reviewed
**End:** work committed · handoff note (done / next / gotchas) · blockers logged · background jobs recorded in STATE with expected finish and success criteria

## 3. RTL review (units and system)
- [ ] Verilog-2001/2005 subset; shared `numerics.vh` used for `rshift_rnd`, `sat`, `requant`; no ad-hoc rounding
- [ ] Signed discipline: no unsigned operand in signed arithmetic; explicit widths and sign extension everywhere
- [ ] Widths match the frozen format table; no silent truncation (check every assignment between different widths)
- [ ] Rounding is half-up toward +∞ per N2-2; INT8 saturation is symmetric ±127 unless the spec says otherwise
- [ ] Every ROM: only `clk/addr/data`, loaded from hex, no write path; no weight, table or parameter stored in any other module
- [ ] RAMs only for KV cache and activation buffers; sized by the memory plan; reset behaviour documented
- [ ] Unit protocol respected (`start/busy/done`, one read and one write port); no combinational loops
- [ ] GQA mapping `i // 3`; causal range `[0, pos]`; K stored after RoPE; V stored raw
- [ ] Softmax subtracts the max; handles `pos = 0` and `pos = MAX_CTX−1`
- [ ] Argmax: lowest-id tie-break; per-row scale applied before comparison (or per-tensor head, per frozen spec)
- [ ] Trace taps listed in the manifest; no `$display`, `#`, `initial` in RTL except ROM `$readmemh`
- [ ] Lints clean: iverilog `-Wall`, verilator `-Wall`, Yosys `check -assert`
- [ ] Names, headers, manifest hash per conventions; no timestamps or absolute paths

## 4. Numerics review
- [ ] Every op's integer algorithm, widths, rounding and saturation are written in the spec and implemented once in `numerics/ops.py`
- [ ] Constants (`M, s`, LUTs, tables, `INV_D`) derived offline in float64 with explicit rounding; hashes recorded
- [ ] Range statistics per layer reported; saturation counts are zero or explained
- [ ] Calibration data, statistic and seeds recorded; evaluation windows hashed and disjoint from calibration where required
- [ ] Per-op ablation done before sweeps; sweeps logged completely (no deleted runs)
- [ ] The scorer equals the stepper bit-exactly; the interpreter equals the oracle bit-exactly
- [ ] Quality thresholds set by ADR from measurements, then frozen
- [ ] Argmax comparison domain is consistent across vocabulary entries
- [ ] No float in the integer path; no hidden `np.float` operations (tests assert dtype integrity)

## 5. Gate checklists

### H0 — Audit and bootstrap
- [ ] Fresh-clone Part I reproduction: `make check`, `make demo` pass; test count and versions recorded
- [ ] `part1_audit.md` lists each Part I deliverable with evidence or gap; Part I claims re-derived from `results/` (e.g. arithmetic share of the 7 projections: ≈79% per decoded token with LM head; >90% only for prefill)
- [ ] `part1-final` tag exists; Part I STATE archived
- [ ] Retrospective ADR decided; process trims applied
- [ ] `docs/model_facts.md` v2 complete; discrepancies resolved
- [ ] Budgets measured: Verilator speed, build time/RAM, RTL bytes per weight, disk; `budgets.json` written
- [ ] Tiny model generated and flowing; `make check2` green; layering test green
### HA — Numerics frozen
- [ ] fp32 NumPy reference matches HF (≤ 1e-3 logits, identical argmax on ≥ 20 prompts)
- [ ] `numerics/ops.py` tests incl. exhaustive small-width; scorer == stepper
- [ ] Per-op ablation and full sweep logs present; Pareto table and ADR-2-1 approved by the human
- [ ] Final format table and thresholds frozen (`config_frozen.json` hash)
- [ ] Golden traces generated and hashed; `trace_diff.py` tested on injected faults
- [ ] Cost proxies written; numerics checklist (§4) completed by the verifier
- [ ] Student answered HA explain-it questions
### HB — Compiler
- [ ] Graph IR invariants I1–I7 tested; round-trip lossless
- [ ] Format assignment covers every edge; calibration ranges respected
- [ ] ROM packing: images, address map, sha256; padding tests green
- [ ] Schedule/microprogram and memory plan generated for tiny and SmolLM2
- [ ] Interpreter == `numerics/` decoder bit-exactly (tiny all traces; SmolLM2 tokens + layer 0)
### HC — RTL units
- [ ] Every unit passes V0 (exhaustive where ≤ 2²⁰ points) and its mutants
- [ ] Weight audit v2 and ROM interface test pass and fail on injected faults
- [ ] Engine reproduces Part I vectors and new non-multiple shapes
- [ ] Sequencer op order equals the interpreter
- [ ] Mutation suite v2 on units: 100% killed; RTL checklist (§3) completed by the verifier
### HD — Tiny model end to end
- [ ] ≥ 100 prompts token-exact and trace-exact incl. contexts to `MAX_CTX`
- [ ] Icarus and Verilator traces identical
- [ ] Top-level mutation suite 100% killed; negative controls FAIL as required
- [ ] Determinism and structural audits pass; lints clean
### HE — Full model
- [ ] Layer-0 trace equality at multiple positions
- [ ] Full 30-layer decode: ≥ 10 prompts × ≥ 16 tokens token-exact; sampled-layer trace equality
- [ ] Long-context run at `MAX_CTX` exact
- [ ] Cycles/token and sim speed recorded; text demo works with divergence statistics vs fp32 greedy
- [ ] Final quality numbers produced by script and linked to the RTL equality evidence
- [ ] Fallback ladder: F0 achieved, or ADR-2-F records F1 with exactly what is proven vs inferred
### HF — Evaluation
- [ ] EA1–EA3 complete with fairness rules checked item by item
- [ ] Analytic cycle model validated against simulation (error reported)
- [ ] EA4 comparison done or cut by ADR; limits of extrapolation stated
- [ ] EA5 assumptions table with sources, ranges, sensitivity; every output labeled "estimate"
- [ ] Published-claims context separates vendor-reported from independent measurements
- [ ] Verifier re-ran at least one data point from a clean checkout
### HG — Release
- [ ] Fresh clone: `make demo2-fast` and `make demo2` pass; runtimes documented
- [ ] CLI documented; README and limitations complete
- [ ] Whole-project report complete; traceability has no unsourced number
- [ ] Certificate/endorsement pages prepared and **signed by humans**; AI-assistance disclosure included
- [ ] Student answered all explain-it questions; viva rehearsal done

## 6. Demo readiness
- [ ] `make demo2-fast`: tiny model, prompt → tokens in RTL sim → PASS + mutation kill matrix
- [ ] `make demo2`: SmolLM2 prompt → text in RTL sim; shows tokens equal to the integer reference and the first divergence from fp32 text
- [ ] A pre-run output is saved in case a live run is slow
- [ ] You can open a ROM hex image and point to a weight, a requant parameter, a table
- [ ] You can open a trace diff and explain how the first divergence is found
- [ ] You can state C1′, C2′, the fairness rules, and the ASIC projection's caveats without notes

## 7. Whole-project report
**Outline:** Abstract · 1 Introduction and motivation · 2 Background (transformer decoder, INT8 and integer-only inference, constant-coefficient arithmetic, model-specific hardware) · 3 Problem statement, scope, non-goals · 4 Part I: linear-layer compiler (revised) · 5 Part II: numerics (formats, non-linear operators, quality) · 6 Part II: compiler extension (IR, ROM packing, schedule) · 7 Part II: architecture (engine, units, KV cache, sequencer) · 8 Verification methodology (oracles, ladder, mutation, audits) · 9 Results (C1′; quality C2′; architecture study; ASIC projection with labeled estimates) · 10 Discussion and limitations · 11 Future work (physical design, spatial/ROM hybrids, more models, longer context, prefill hardware) · 12 Conclusion · Appendices: reproducibility, repo map, ADR list, AI-assistance disclosure.

**Traceability table** (`docs/report_traceability.md`): `Report item | Value | Source file | Producing command | Commit`. Every number appears once. If it is not here, it does not go in the report.

**Report checks:** claims match what the data supports · estimates labeled · negative or mixed findings stated · Part I claims re-derived (no stale percentages) · limitations include: greedy decoding only, host tokenizer, short `MAX_CTX`, simulation not silicon, proxy metrics, projection assumptions · figures reproducible.

## 8. Viva preparation
- [ ] I can explain the full decode step op by op, with the formats on every edge
- [ ] I can explain why each non-linear op is implemented the way it is and how its error was measured
- [ ] I can explain why exactness (C1′) and quality (C2′) are separate claims and what links them
- [ ] I can explain the oracle structure and why the verification can be trusted (interpreter, traces, mutants, audits)
- [ ] I can explain why the full model uses a ROM-based engine and what the spatial design from Part I contributes
- [ ] I can explain the ASIC projection: its inputs, its ranges, what dominates, and what it does not show
- [ ] I can run a demo, open a trace diff, and inspect a ROM image unaided
- [ ] I can state what a real chip would still need (place-and-route, memory macros, clocking, power, test) and what Part III would be
