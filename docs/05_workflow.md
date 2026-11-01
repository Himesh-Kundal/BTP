# 05 — Agent Workflow

## 1. Principles
1. **The task file is the unit of work.** Sessions are disposable; task files, STATE, LOG and git are the memory.
2. **Build the thin end-to-end slice first**, then deepen it. Verification exists from day one.
3. **The builder never grades itself.** An independent verifier session checks correctness-critical work.
4. **Evidence or it did not happen.** Claims link to commands, outputs, `results/` files or commits.
5. **Humans own decisions and understanding.** The agent owns execution.

## 2. Roles
If the harness supports subagents, use them. Otherwise run each role as a separate session with its own prompt (§9).

| Role | When | Inputs | Outputs | May edit |
|---|---|---|---|---|
| **Planner** | start of each phase | `docs/04`, STATE | `tasks/T*.md` with plans, estimates, flagged yellow items | tasks/, STATE |
| **Implementer** | per task | AGENTS.md, task file, cited spec sections | code, tests, evidence | everything except frozen files |
| **Verifier** | tasks tagged `rtl`, `numeric`, `gate` | spec, diff, test outputs only (**no implementer reasoning**) | verdict PASS/FAIL with findings; extra tests | tests/ and `results/gates/` only, never the implementation |
| **Librarian** | end of each phase | everything | docs updated, results index, walkthrough draft, STATE | docs/, state/, results index |

The verifier's job is to **try to break the work**: run the checklists, craft adversarial vectors, run mutants, look for hardcoded expectations, and confirm that numbers in docs match `results/`.

## 3. Task loop (Implementer)
1. **Orient.** Read AGENTS.md, STATE, the task file, and cited spec sections. `git switch -c t<id>-<slug>`. Run `make check`; it must be green before starting. If not, fix or escalate first.
2. **Plan.** Fill the task file's Plan section: files to touch, tests to write first, risks, estimate. If the plan contains a YELLOW item, stop and ask.
3. **Tests first.** Write failing tests that map one-to-one to the acceptance bullets. Commit them as `T<id>: add failing tests`.
4. **Implement** the minimum that makes them pass. Keep generated code and generator code separate. No stubs left silently; mark any with `TODO(T<id>)` and list them in Notes.
5. **Verify.** Run acceptance commands, `make check`, lint. Run the relevant checklist from `docs/07`. Paste a summary of outputs (not whole logs) into Evidence.
6. **Review.** If tagged, hand off to the verifier session. Fix findings. Re-verify.
7. **Record.** Update task Status, Evidence, Notes; append a LOG line; refresh STATE (current task, next 3, blockers, handoff note). Commit as `T<id>: <summary>`; merge to `main` only if `make check` is green.
8. **Stop conditions.** Any escalate trigger in AGENTS.md. Write the BLOCKERS entry, leave the branch intact, update STATE, end or switch tasks.

## 4. Gates
Each phase ends with a gate task `T<phase>.G`:
1. Librarian compiles `results/gates/G<n>.md`: checklist from `docs/07` §5 with an evidence link per item, known issues, deviations, ADRs raised.
2. Verifier session independently re-runs the gate commands from a clean checkout and returns PASS/FAIL.
3. Human review (§7). Only then is the gate tagged `git tag gate-G<n>` and the next phase opened. **No next-phase work starts before the tag**, except planning.

## 5. State files (all under `state/`)
**STATE.md** — one page, always current: start date, supervisor/deadlines, current phase and task, last green commit, handoff note, next 3 tasks, open blockers, open yellow items awaiting the human, tool versions, parking lot. Template is in `state/STATE.md`.

**LOG.md** — append-only, one line per event:
`2026-11-04 15:12 | T1.4 | implemented simulator runner | make sim PASS, negative control FAIL as expected | a1b2c3d`

**BLOCKERS.md** — entries:
```
## B-<n> (T<id>) — <title>        Status: open|resolved
Symptom: ...
Hypotheses tried: 1) ... → result  2) ... → result  3) ...
Need from human: decision / info / permission
Options: A ... B ...
```

## 6. Templates
**Task file `tasks/T<id>.md`**
```
# T<id> — <title>
Status: todo | in-progress | blocked | review | done
Phase/Gate: P<n> / G<n>     Tags: [rtl|numeric|gate|yellow]     Estimate: <h>
Depends on: T..           Spec refs: docs/02 §.., docs/06 §..
## Goal
## Acceptance (executable)
- [ ] `<command>` → expected result
## Plan (before coding)
Files / tests first / risks
## Evidence (after)
Commands run, summarized outputs, commit hash
## Verifier review
Verdict, findings, vectors tried
## Notes / decisions / TODOs
```

**Gate report `results/gates/G<n>.md`:** checklist table (item | evidence link | status), deviations, ADRs, known issues, verifier verdict, human sign-off line.

## 7. Human checkpoints (the student's job)
| When | What | Time budget |
|---|---|---|
| Daily or per session | skim LOG and STATE; answer BLOCKERS and yellow items | 10 min |
| Each gate | read the gate report; run the demo command yourself; read the phase walkthrough; answer explain-it questions in your own words (`docs/walkthroughs/P<n>.md`) | ≤ 1 h |
| Weekly | 30-minute review of progress vs timeline; update the cut list status; talk to the supervisor | 30 min |
| Before accepting any ADR | read the options; decide | as needed |

Rule: **if you cannot explain a gate's work, the gate is not passed.**

## 8. Understanding and viva preparation
At every gate the Librarian drafts `docs/walkthroughs/P<n>.md` (what was built, why, how to run it, how it could fail). The student then answers these in their own words:

- **P1:** Why is a testbench that checks zero vectors dangerous? What does a mutation test prove that passing tests cannot?
- **P2:** Why does `X @ W.T` reproduce the Linear layer? What would change with biases?
- **P3:** Why per-channel weight scales but a per-tensor activation scale? Why does K=1536 never overflow INT32? What do C1 and C2 each claim?
- **P4:** What does "fixed-weight" mean in the generated Verilog, and how do we prove no weight port exists? How does padding keep results exact? Why can signed/unsigned mixing silently break a MAC?
- **P5:** What makes the F-vs-P comparison fair? What is counted in P's area that is not in F's?
- **P6:** Walk through one waveform. Which mutants are hardest to catch, and why?
- **P7:** What does Yosys `stat` tell you and what can it not? Why is extrapolation only an estimate? What would falsify H1?
- **P8:** What are the limitations, and what would Part II need?

## 9. Prompts
**Kickoff (once)**
> You are the Implementer on the project described in `AGENTS.md`. Read `AGENTS.md`, then `state/STATE.md`. Start with T0.1. Follow the task loop in `docs/05_workflow.md` §3 exactly. Do not read or build anything outside the current task's cited sections. If anything in AGENTS.md or the specs conflicts with an instruction I give you in chat, tell me instead of silently choosing.

**Planner (per phase)**
> Act as the Planner for phase P<n>. Using `docs/04` and STATE, create `tasks/T<id>.md` for every task in the phase with a concrete plan, acceptance commands copied from `docs/04`, tags, and estimates. Flag every YELLOW item. Do not write code.

**Verifier (per tagged task or gate)**
> Act as the Verifier for T<id>. You have the spec, the diff, and the test output. You have not seen the implementer's reasoning. Do not edit implementation files. Try to break the work: run the relevant checklist from `docs/07`, write adversarial tests, run mutants, look for hardcoded expectations and tests that cannot fail, and confirm every number claimed matches `results/`. Return PASS or FAIL with evidence.

**Librarian (end of phase)**
> Act as the Librarian for phase P<n>. Compile the gate report from evidence, update docs and STATE, regenerate the results index, and draft `docs/walkthroughs/P<n>.md`. Do not change code or results.

## 10. Harness requirements
- Shell, filesystem and git in a persistent working directory; permission to run `make`, `pytest`, `iverilog`, `vvp`, `verilator`, `yosys`.
- Tools preinstalled or installable (iverilog, verilator ≥ 5, yosys). GTKWave is for the human: the agent produces VCDs.
- Long jobs (synthesis grids) run with timeouts and write partial results; the agent must not poll in tight loops.
- Network limited to: model download (pinned revision) and package installs.
- No secrets in the repo or environment.
- Subagent or multi-session support for the Verifier role; if absent, the human starts a fresh session with the Verifier prompt.
- Keep AGENTS.md ≤ ~150 lines and STATE ≤ 1 page. Long content lives in `docs/` and is read on demand.
- Commit often on task branches; a hook or CI step runs `make check` before merge to `main` if available.

## 11. Anti-gaming checks the harness should run automatically
1. `tests/test_layering.py` — forbidden imports and hardcoded model names.
2. Frozen-file guard: `git diff` of `verification/golden/**` and assertion lines in merged tests is non-empty ⇒ merge blocked pending an ADR.
3. Results provenance: every file in `results/` has the producing command and commit in its header or sidecar.
4. Mutation suite run in `make regression`.
5. Testbench negative control in `make check`.
6. Docs numeric lint: numbers tagged `{{result:...}}` in docs must match `results/` (optional but recommended).
