# Task UD.6 — Determinism and Structural Audits

## Objective
Enforce strict architectural contracts across the generated hardware accelerator and compilation pipeline:
1. **Rule 7 (Immutable Weights):** All weights, lookup tables, and requantization parameters reside in dedicated ROM modules with zero write ports. Dynamic writable storage is strictly limited to authorized activation buffers (`buf_res`, `buf_act`, `buf_q`, `buf_k`, `buf_v`, `buf_attn_out`, `buf_gate`, `buf_up`, `buf_swiglu`, `buf_logits`) and KV cache.
2. **Rule 9 (Compiler Outputs, Not Oracle Outputs):** Generated hardware modules and packed hex images contain no test oracle values, hardcoded tokens, activations, or expected output sequences.
3. **Rule 11 (Determinism):** Compilation, code generation, and simulation outputs are 100% deterministic (byte-identical across repeat invocations under pinned inputs).
4. **Lint Cleanliness:** Zero errors or severe warnings across Icarus Verilog (`iverilog -Wall -g2005`), Verilator (`verilator --lint-only -Wall`), and Yosys (`hierarchy -check; proc; check -assert`).

## Implementation Details
1. **Audit Engine (`verification/structural_audit.py`):**
   - `audit_no_expected_values_in_rtl`: Scans cleaned RTL for forbidden oracle symbols, expected memory files, and hardcoded token output assignments.
   - `audit_top_accel_memory_structure`: Inspects memory declarations and `rom_sync` instantiations in `top_accel.v`. Verifies zero write ports on ROM instances and flags any unauthorized writable arrays.
   - `audit_generator_determinism`: Verifies byte-identical output (identical SHA-256) on repeat RTL generator invocations.
   - `run_multi_tool_lint`: Executes Icarus Verilog, Verilator, and Yosys lint and design checks on the complete accelerator hierarchy.
2. **Verification Suite (`tests/test_audits_v2.py`):**
   - Tests clean top accelerator conformance for all structural, memory, and oracle rules.
   - Includes rigorous negative controls:
     - Injected `exp_tokens` array -> Rejected.
     - Injected golden trace `$readmemh` -> Rejected.
     - Injected unauthorized writable array `secret_weights` -> Rejected.
     - Injected write port `.we(1'b0)` into `rom_weights_inst` -> Rejected.
   - Verifies simulation determinism across repeat runs (identical output tokens, identical cycles, and byte-identical hierarchical trace logs).
   - Verifies lint cleanliness across all three EDA tools.

## Verification & Acceptance
- Ran `pytest tests/test_audits_v2.py -v`: 9/9 tests PASSED in 14.01s.
- Ran `pytest tests/test_layering.py -v`: 3/3 tests PASSED (pure Python, zero torch/transformers in `verification/`).

## Status
Status: complete
Commit: UD.6: determinism and structural audits
