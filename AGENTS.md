# AGENTS.md (Part II edition) — read at the start of EVERY session

This replaces the Part I AGENTS.md. Part I docs (`docs/01`–`docs/07`) stay in the repo as history and as the source of contracts that Part II extends. Where a Part II doc (`docs/part2/`) disagrees with a Part I doc, **Part II wins**. Part II task IDs start with `U`, gates with `H`, ADRs are `ADR-2-<n>`, so nothing collides with Part I's `T`/`G`.

## Mission
Close out the whole project. Extend the Part I compiler from single linear layers to the **complete SmolLM2-135M decoder** (embedding, RMSNorm, RoPE, GQA attention with KV cache, SwiGLU, residuals, LM head, greedy argmax) and generate a model-specific accelerator in Verilog whose weights live in immutable ROMs. Prove the generated RTL generates **the same tokens, bit for bit,** as an independent integer reference. Measure the integer model's quality against fp32, and study area, latency and throughput of the architecture, ending in a clearly labeled ASIC projection.

Statement and scope: `docs/part2/01_problem_and_scope.md` · Contracts: `docs/part2/02_architecture_and_contracts.md`

## Read order (read what the task needs, not everything)
1. `state/STATE.md` → 2. your task file in `tasks/` → 3. the doc sections it cites.

## Golden rules (Part I rules 1–10 still apply; restated and extended)
1. **Evidence over claims.** "Done" means a command ran and its output shows it. Every number in docs or report comes from a `results/` file produced by a script at a recorded commit.
2. **One oracle.** `numerics/` defines the meaning of every operation. Expected values come only from it (or from the fp32 reference harness for quality). Never compute expectations from RTL, from the code under test, or inside a testbench.
3. **Numerics first.** No RTL for an operation before its integer semantics are frozen (gate HA). If RTL and `numerics/` disagree, the frozen spec decides; if the spec is ambiguous, raise an ADR. **Never change the oracle to match the RTL.**
4. **Never weaken a test to get green.** Frozen: `verification/golden/**`, golden traces, existing assertions, tolerances, quality thresholds after HA, mutation kill requirements. Changes need an ADR and human approval.
5. **The spec is law.** Code that disagrees with `docs/part2/02` is wrong until an ADR says otherwise.
6. **Layering.** `ir/`, `compiler/`, `backend/`, `numerics/`, `verification/` never import torch/transformers. Only `frontend/`, `eval/fp32/` and `examples/` may. Enforced by `tests/test_layering.py`.
7. **Immutable weights.** Every weight, table and requantization parameter lives in a ROM module with no write port, initialized from compiler-generated files. Only dynamic state (KV cache, activation buffers) is writable memory. Enforced by the ROM interface test and the weight audit v2.
8. **No floating point** in RTL or in the integer reference path. Floats appear only in the fp32 reference and in offline compile-time computation of constants.
9. **Compiler outputs, not oracle outputs.** The RTL generator may embed weights, tables and requant parameters, never tokens, activations or expected values. Enforced by a structural audit.
10. **Small, green, reversible.** One task per branch; `main` passes `make check` (fast subset); long jobs run via `make long-*` targets and never block a merge unless the task requires them.
11. **Determinism.** Fixed seeds, pinned model revision and tools, no timestamps in generated files; same input ⇒ byte-identical output.
12. **Scope discipline.** Only tasks in `docs/part2/04_plan_and_tasks.md`. New ideas go in the Parking lot in STATE.
13. **Ask on numerics and interfaces.** A wrong guess there corrupts everything downstream.

## Commands
Part I: `make env-check · check · sim · regression · synth · demo`. Part II adds (created in U0.4):
`make check2` (fast: lint + unit + numerics + tiny-model quick sim) · `make numerics` · `make sim-tiny` · `make sim-full` (long) · `make synth2` · `make demo2-fast` (tiny model) · `make demo2` (full model, short generation) · `make long-<target>` (background jobs with logs)

## Autonomy zones
- **GREEN:** code, added tests, scripts, docs, experiments within the plan, new results.
- **YELLOW (propose in task file, wait for the human):** any change to formats, rounding, scales, tables, LUT sizes after HA; interfaces or file/trace formats; new dependency or tool; editing frozen items; scope or schedule changes; choosing the final numeric config (UA.7); invoking the fallback ladder.
- **RED (never):** fabricating or hand-editing results; marking done with failing checks; deleting `results/` or history; force-push; weakening/skipping tests silently; hardcoding expected outputs; copying golden traces into testbench outputs; network beyond pinned model/data downloads and package installs.

## Escalate (entry in `state/BLOCKERS.md`, stop that task) when
- Spec ambiguity or contradiction affecting correctness.
- 3 distinct fix hypotheses fail (log each).
- A required tool or resource is missing or insufficient (RAM, disk, Verilator version). Do not substitute a weaker check silently.
- Passing would require changing a test, tolerance or oracle.
- A task exceeds 2× its estimate.
- Quality results contradict the plan (e.g. perplexity blow-up): stop tuning, write up, ask.

## Debugging divergence (RTL ≠ oracle)
Reproduce on the tiny model → run `verification/trace_diff.py` to find the **first** diverging op/tensor → reproduce that op in its unit test with the exact inputs → fix the generator/RTL (never the expected file) → re-run the full ladder from that level upward.

## Long jobs
Run in the background with `make long-*`, log to `build/logs/`, set timeouts, write partial results, and resume rather than restart. Do not poll in a tight loop; check at sensible intervals and do other work.

## Definition of Done / session protocol
`docs/part2/07_checklists.md` §1–2. Short form: acceptance commands pass → tests first → `make check2` green → docs/STATE/LOG updated → commit `U<id>: <summary>` → verifier pass if tagged `rtl`, `numeric`, `gate`. Start: `git status`, `make check2`, read STATE and task. End: update task, STATE, LOG, commit, write handoff note.

## Repo map (additions to Part I)
`numerics/` (integer op library and decoder, the oracle) · `eval/` (fp32 reference, datasets, metrics) · `rtl/lib/` (hand-written parameterized units) · `sim/` (testbench generators, Verilator harness) · `results/part2/` · `tasks/` · `docs/part2/`
