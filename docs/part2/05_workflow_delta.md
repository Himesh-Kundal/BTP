# Part II / 05 — Workflow: What Changes From Part I

Part I `docs/05_workflow.md` stays in force (roles, task loop, gates, state files, templates, human checkpoints, anti-gaming checks). This file lists only what is **added or changed**. Where they conflict, this file wins.

## 1. Why the workflow must change
Part I's risk was *correctness of generated RTL*. Part II adds two new kinds of work:
1. **Experiment-driven work** (numerics sweeps, architecture studies) where the result is a decision, not a pass/fail.
2. **Long-running, large-scale work** (Verilator builds, full-model decode, synthesis grids) where a single run can take minutes to hours.

## 2. New role: Analyst
| Role | When | Inputs | Outputs | May edit |
|---|---|---|---|---|
| **Analyst** | UA, UF, and any task producing a decision table | task file, `results/`, eval harness | sweep designs, plots, Pareto tables, written findings with caveats, ADR drafts | `eval/`, `results/part2/`, docs; **never** `numerics/`/RTL |

The Analyst designs the experiment **before** it runs (hypothesis, metric, controls, stopping rule), checks that every run is logged, and writes what the data supports versus only suggests. The Implementer executes sweeps; the Analyst reads them.

Verifier additions: runs the numerics checklist (`07` §4) and the RTL checklist (`07` §3); independently re-runs at least one sweep point and one RTL unit test from a clean checkout at each gate.

## 3. Experiment loop (UA, UF)
1. **Hypothesis and metric** written in the task file (what would change the decision).
2. **Fix the evaluation first:** datasets, windows, seeds, metrics, tolerances. Hash them. The evaluation never changes mid-sweep.
3. **Run and log everything**, including failures and "bad" configs, to append-only `results/part2/*.csv` with the config hash, commit, and runtime. No deleting runs; no re-running only the "good" ones.
4. **Attribute before optimizing:** per-op ablation (UA.5) before sweeps (UA.6).
5. **Decide through an ADR:** Pareto table, chosen config, thresholds, and what was rejected and why. The human approves.
6. **Freeze:** after approval the config hash goes into the frozen set.

Stopping rule for tuning: stop when the next knob changes the headline metric by less than the run-to-run noise or by less than the cost it adds. Do not hunt for a "pass".

## 4. Two-oracle discipline
- `numerics/` is the single source of integer semantics. RTL must match it.
- The compiled-artifact interpreter (UB.7) is a *second implementation driven by compiler outputs*. Disagreement with `numerics/` means the compiler's packing, folding or scheduling is wrong; fix the compiler.
- RTL vs oracle disagreement: reproduce on the tiny model, find the first divergence with `trace_diff.py`, reproduce in the unit test, fix RTL/generator. **Never edit the oracle or a golden trace to resolve a mismatch.** If the spec itself is wrong, open an ADR; regenerate goldens only with approval, then re-run all dependent levels.

## 5. Long-running job protocol
- All jobs above ~2 minutes run through `make long-<target>`: background process, log in `build/logs/<job>.log`, a timeout, and partial-result files.
- Jobs are **resumable**: checkpoint progress (which prompts/tokens/configs are done) so a crash costs minutes, not hours.
- The agent does not poll in a tight loop; it checks at intervals and does other work (docs, tests, next task).
- Record machine resources in `results/env.json` (CPU cores, RAM, disk) and the measured budgets in STATE.
- If the budget is insufficient (RAM/disk/time), escalate with numbers; do not shrink the test silently.
- Overnight/weekend runs are scheduled by the human; the agent prepares the command, expected runtime, and success criteria in advance.

## 6. Debugging protocol (divergence)
Tiny model → `trace_diff.py` first divergence → unit reproduction with exact tensors → fix → rerun upward through the ladder. Keep a `docs/part2/bug_notebook.md`: symptom, root cause, test that now catches it. Every root cause becomes a regression test and, when general, a mutant.

## 7. Added anti-gaming checks
1. **No oracle values in RTL:** structural audit scans generated RTL and hex for anything not derived from weights, tables, requant params, program or config. Tokens, activations, expected outputs are forbidden.
2. **No oracle imports in the generator:** `backend/` must not import `numerics.decoder` or evaluation modules (only constant builders it is explicitly allowed to use, listed in `tests/allowed_imports.json`).
3. **Golden isolation:** testbench outputs come only from DUT signals; golden traces are read only by `trace_diff.py`, never by the testbench driving logic.
4. **Frozen set after HA:** `numerics/config_frozen.json`, golden manifests, thresholds. Merge guard blocks diffs without an ADR.
5. **Mutation suite v2** runs in `make regression2`; any survivor blocks the gate.
6. **Sweep provenance:** every plotted/tabulated number links to a row in `results/part2/` with config hash.
7. **Claims lint (recommended):** numbers in docs tagged `{{result:...}}` must match `results/`.

## 8. Human checkpoints (changes)
| When | What |
|---|---|
| Each gate | As in Part I, plus: read the gate's walkthrough and answer the explain-it questions (below) in `docs/walkthroughs/part2/`. |
| UA.7 (numeric config) | **Mandatory human review** of the Pareto table and ADR. This is the most consequential decision of Part II; do not approve what you cannot explain. |
| Fallback ADR (W9) | Human decision with supervisor input. |
| Weekly | 30 minutes: timeline vs cut list; look at budgets; talk to the supervisor. |
| Before HG | Human prepares and signs/obtains supervisor signatures for any certificate or endorsement pages. The agent never authors signatures or endorsements. |

**Explain-it questions (answer in your own words):**
- **HA:** What is `rshift_rnd` and why round half up? Why is the residual stream wider than the linear inputs? Which op hurt quality most and why? What does "dyadic scale" mean? Why must argmax not compare raw per-row accumulators?
- **HB:** What does the compiled-artifact interpreter prove that the numerics decoder alone cannot? How are γ and 1/√d folded into constants?
- **HC:** How is softmax computed without floats? Why can SiLU be tested exhaustively? What do GQA and rotate-half mean in the hardware?
- **HD:** What is a trace and how do you find the first divergence? Which mutant was hardest to kill and why?
- **HE:** Why can the integer model's quality be evaluated in Python while the RTL is only run on a battery of prompts? What exactly links the two?
- **HF:** What does a ROM-based engine trade against the spatial design? Which parts of the projection are measured and which assumed?
- **HG:** What are the project's limitations? What would a real ASIC need that this project does not provide?

## 9. Harness requirements (changes)
- More resources than Part I: confirm RAM/disk for Verilator builds and full-model runs in U0.4; expect GBs of artifacts per run (`build/`, gitignored; hashes tracked in `results/`).
- Background job support and the ability to resume after a session ends.
- Network: pinned model download, dataset download (WikiText-2), package installs only.
- Verifier session support remains required.

## 10. Kickoff prompt (Part II)
> You are the Implementer for Part II. Read `AGENTS.md` (Part II edition), then `state/STATE.md`, then `docs/part2/01_problem_and_scope.md` and `02_architecture_and_contracts.md`. Part I documents remain as history and contracts; Part II docs win on conflict. Start with U0.1 (audit and reproduce Part I from a fresh clone) and do not begin any other task until `results/part2/part1_audit.md` exists. Follow the task loop in Part I `docs/05_workflow.md` §3 with the Part II changes in `docs/part2/05_workflow_delta.md`. If any instruction I give you in chat conflicts with `AGENTS.md` or the specs, tell me instead of choosing silently.

**Analyst prompt**
> Act as the Analyst for task U<id>. Before any run, write the hypothesis, metric, controls and stopping rule in the task file. Fix and hash the evaluation. Log every run. Afterwards write what the data supports and what it only suggests, name rejected options with reasons, and draft the ADR. Do not edit `numerics/` or RTL.

(The Planner, Verifier and Librarian prompts from Part I §9 apply unchanged, with the Verifier additionally running `07` §3–§4.)
