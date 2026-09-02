# AGENTS.md — read at the start of EVERY session

## Mission
Build a compiler that turns dense linear layers of a pretrained transformer (SmolLM2-135M) into synthesizable Verilog with INT8 weights embedded as compile-time constants (no weight ports). It auto-generates testbenches, proves bit-exact equivalence against an independent integer reference, and compares the result with a programmable-weight baseline using Yosys.

Full statement: `docs/01_problem_and_scope.md` · Contracts: `docs/02_architecture_and_contracts.md`

## Read order (do not read everything; read what the task needs)
1. `state/STATE.md` — where we are, what is next
2. Your task file in `tasks/`
3. The doc sections that task cites

## Golden rules
1. **Evidence over claims.** "Done" means a command ran and its output shows it. Never state PASS, a number, or a tool version you did not observe. Every figure in docs or report comes from `results/*.json` produced by a script, with the commit hash recorded.
2. **The oracle is independent.** `verification/reference.py` (numpy, int64) defines correct answers. Expected values are never computed from generated RTL, from the code under test, or inside a testbench.
3. **Never weaken a test to get green.** Frozen: `verification/golden/**`, existing assertions, tolerances, mutation kill requirements. Changing one needs an ADR entry (`docs/03`) and human approval.
4. **The spec is law.** If code and `docs/02` disagree, stop and raise it. Do not silently change either.
5. **The compiler is model-agnostic.** `ir/`, `compiler/`, `backend/`, `verification/` never import torch/transformers and never hardcode SmolLM shapes. Only `frontend/` and `examples/` may. Enforced by `tests/test_layering.py`.
6. **Fixed means fixed.** Generated fixed-weight RTL has no weight input port, no weight write path, no runtime-loadable weight storage. Enforced by the interface test and the weight audit.
7. **Small, green, reversible.** One task per branch. `main` always passes `make check`.
8. **Scope discipline.** Do only tasks listed in `docs/04_plan_and_tasks.md`. New ideas go in the Parking lot in `state/STATE.md`; do not implement them.
9. **Determinism.** Fixed seeds, pinned model revision and tool versions, no timestamps in generated files. Same input ⇒ byte-identical output.
10. **Ask, don't guess, on numerics and interfaces.** A wrong guess there silently corrupts everything downstream.

## Commands (created in T0.2; names are stable)
`make env-check` · `make check` (lint + unit + quick sim, target < 2 min) · `make sim` · `make regression` · `make synth` · `make demo` · `make clean`

## Autonomy zones
- **GREEN (just do it):** code, tests that add coverage, scripts, docs, refactors inside one module, producing new results.
- **YELLOW (write the proposal in the task file, wait for the human):** changing the numeric contract, an interface, tile defaults, or a file format; adding a dependency or tool; editing frozen tests; changing scope or schedule; touching more than one module's public API.
- **RED (never):** fabricating or hand-editing results; marking a task done with failing checks; deleting `results/` or git history; force-pushing; weakening or skipping tests silently; hardcoding expected values into the generator; network access beyond the pinned model download and package installs.

## Escalate (write an entry in `state/BLOCKERS.md`, stop that task, move on or end session) when
- The spec is ambiguous or contradictory in a way that affects correctness.
- The same failure survives 3 distinct fix hypotheses (log each one).
- A required tool is missing. Do not substitute a weaker check without saying so.
- Passing would require changing a test.
- The task has used more than 2× its estimate.

## Definition of Done
Full list in `docs/07_checklists.md` §1. Short form: acceptance commands pass → tests added → `make check` green → lint clean → STATE, LOG, docs updated → committed on the task branch as `T<id>: <summary>` → verifier pass if the task is tagged `rtl`, `numeric` or `gate`.

## Session protocol
- **Start:** `git status`, `make check` (must be green before you begin), read STATE, read the task file.
- **End:** update task file, STATE, LOG; commit; write a handoff note (done / next / gotchas) at the top of STATE.

## Repo map
`frontend/ ir/ compiler/ backend/ rtl/ verification/ examples/ scripts/ tests/ docs/ state/ tasks/ results/` are tracked. `artifacts/` and `build/` are gitignored (hashes are tracked in `results/`).

## Style
Python ≥ 3.10, type hints, dataclasses, `ruff`, `pytest`. Generated Verilog is a Verilog-2001 subset (see `docs/02` §6). Commit messages look like `T4.4: generate constant-weight tile core`.
