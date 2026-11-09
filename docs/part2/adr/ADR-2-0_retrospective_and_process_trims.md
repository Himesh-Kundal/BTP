# ADR-2-0: Part I Retrospective and Process Trims

**Status:** Accepted  
**Date:** 2026-11-09  
**Authors:** Himesh Kundal & Antigravity  
**Context:** Transition from BTP Part I (Linear Layer Hardware Compiler) to Part II (Complete Autoregressive Decoder).

---

## 1. Context & Retrospective Findings

During Part I, the team built a complete compiler from Hugging Face PyTorch linear projections to constant-weight Verilog (Design F), reaching all 8 gates (G0–G8) and achieving 230 passing tests. An analysis of the Part I engineering logs, blockers, and bug notebooks reveals key lessons:

### What Worked Well (Keep & Reinforce):
1. **Mutation Testing as the Verification Gold Standard (M1–M9):**
   Mandating a 100% kill rate on text-level fault injections caught edge-case signed arithmetic bugs, off-by-one sign-extension errors, and missing accumulator bits before they escaped to integration.
2. **Strict Architectural Layering (Golden Rule 5 / Rule 6):**
   Confining PyTorch and Hugging Face imports exclusively to `frontend/` kept the compiler (`ir/`, `compiler/`, `backend/`, `verification/`) fast, portable, and completely model-agnostic.
3. **Dual-Simulator Cross-Checks (Icarus Verilog + Verilator):**
   Simulating identical test vectors on both an event-driven simulator (Icarus) and a cycle-compiled simulator (Verilator) caught simulator-specific edge cases in signed port widths and `$signed()` evaluation.

### What Cost the Most Time / Friction Points (Trim & Adapt):
1. **Spatial Flat-Verilog Scaling Wall:**
   Generating unrolled Verilog for large matrices ($576 \times 576$ produced 30 MB / 548,223 lines of RTL) took ~0.54s to emit but severely stressed downstream toolchains (Icarus compilation took >10s, Yosys memory exceeded several GBs). For the full 134.5M parameters of SmolLM2, a purely spatial implementation would require ~20B standard cells and gigabytes of source text.
2. **Synchronous Long-Running CLI Runs:**
   Running long synthesis grids or full regressions synchronously blocked interactive agent turns and risked timeouts.
3. **Tool Version Sensitivities:**
   Minor differences in Verilator CLI flags (`--binary`, `--timing`) between versions required defensive scripting.

---

## 2. Process Decisions & Adjustments for Part II

1. **Shift to ROM-Backed Time-Multiplexed Engine (Decision D2-02):**
   Instead of generating monolithic spatial arrays with billions of gates, Part II uses a parameterized $R \times C$ MAC engine reading weights from immutable `$readmemh` ROMs. This keeps generated RTL small (< 5k lines of module code) while storing weights as structured hex files.
2. **Two-Oracle Discipline:**
   - Single semantic oracle: `numerics/` (pure integer Python reference).
   - Second independent check: `UB.7` compiled-artifact interpreter, which executes the compiled microprogram on packed ROM hex images to guarantee compiler correctness prior to RTL simulation.
3. **Layered Verification Targets:**
   - `make check2`: Ultra-fast inner-loop test suite (< 30 seconds: lints, unit math, tiny model sim).
   - `make sim-tiny`: Full token-exact simulation of tiny model.
   - `make long-<target>`: Background execution protocol with `build/logs/<job>.log`, timeouts, and partial result serialization for jobs > 2 minutes.
4. **Script Trimming:**
   Deprecated single-use Part I demonstration wrappers (e.g. `scripts/serve_docs.py` has been removed; `scripts/run_toy_sim.py` remains as legacy check; new flows use unified `./llmhw`).

---

## 3. Consequences

- **Positive:** Workstation RAM and disk usage remain bounded (< 4 GB RAM, < 2 GB disk); verification iteration cycles drop from minutes to seconds; clear separation between compiler scheduling bugs and RTL bugs.
- **Negative:** Requires developing a microprogrammed sequencer and address generator that did not exist in Part I's pure spatial datapath.
