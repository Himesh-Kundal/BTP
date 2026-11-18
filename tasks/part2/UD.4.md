# Task UD.4 — Cross-Simulator Equality

## Objective
Demonstrate cross-simulator trace and token equality between Icarus Verilog (`iverilog` / `vvp`) and Verilator (`--binary --timing`). Verify that standard POSIX `diff -r` between the two simulator trace directory trees is completely empty across all decoded tokens, layers, and operations.

## Implementation Details
1. **Simulation Runner Parity (`sim/tb_generator.py`):**
   - Configured uniform stimulus, timing, and sampling hooks for both Icarus Verilog and Verilator.
   - Suppressed non-critical Verilator-specific timing/linter warnings (`-Wno-BLKSEQ -Wno-WIDTHTRUNC -Wno-INITIALDLY -Wno-DECLFILENAME -Wno-CASEINCOMPLETE -Wno-SYNCASYNCNET -Wno-UNDRIVEN`).
2. **Cross-Simulator Test Suite (`tests/test_cross_simulator.py`):**
   - Ran identical prompt sequences through both Icarus and Verilator top-level simulations.
   - Verified that emitted output tokens match identically (`res_iv.tokens == res_ver.tokens`).
   - Verified that `verification/trace_diff.py` reports `is_identical=True` and `first_divergence=None`.
   - Executed system `diff -r traces_iverilog traces_verilator`, verifying returncode 0 and empty stdout.

## Verification & Acceptance
- Ran `pytest tests/test_cross_simulator.py -v`:
  - `test_cross_simulator_trace_and_token_equality`: PASSED in 10.07s.
- Ran `make check2`: 145/145 tests passing.

## Status
Status: complete
Commit: UD.4: cross-simulator trace equality between Icarus and Verilator
