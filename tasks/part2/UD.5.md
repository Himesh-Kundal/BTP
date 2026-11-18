# Task UD.5 — Top-Level Mutation Suite

## Objective
Implement a comprehensive top-level mutation testing suite covering the full accelerator datapath and memory subsystem. Ensure 100% kill rate across 7 critical fault models:
1. `TM1: causal off` — Softmax causal mask normalized past current position (`pos + 2`)
2. `TM2: kv-head map` — GQA KV-head mapping uses modulo instead of floor division
3. `TM3: pos off-by-one` — Position counter / sequencer initialized to 1 instead of 0
4. `TM4: rope table shift` — RoPE sin/cos table lookup address offset by +1
5. `TM5: residual scale` — Residual addition subtracted instead of added
6. `TM6: tie-break` — Argmax tie-break selects highest index instead of lowest
7. `TM7: ROM word swap` — Weight ROM word 0 and word 1 swapped in `weights_128b.hex`

## Implementation Details
1. **Mutation Harness (`tests/test_top_mutation.py`):**
   - Implemented programmatically generated mutated Verilog files for each fault model.
   - Tested mutations against golden integer reference outputs.
   - For `TM1`-`TM5` and `TM7`, detected divergence via token mismatches or intermediate trace verification (`trace_diff`).
   - For `TM6` (tie-break), generated an adversarial weight scenario where top logits tie, confirming the golden model picks index 0 while the mutant with `>=` picks index 254 (100% killed).
2. **Result Matrix (`results/top_mutation_matrix.json`):**
   - Generated structured JSON matrix logging mutant ID, name, description, status (`KILLED`), and specific divergence/kill reason.
   - Total mutants: 7. Killed mutants: 7. Kill rate: 100.0%.

## Verification & Acceptance
- Ran `pytest tests/test_top_mutation.py -v`:
  - 7/7 mutants killed.
  - Test passed in 51.85s.
- `results/top_mutation_matrix.json` verified.

## Status
Status: complete
Commit: UD.5: top-level mutation suite with 100% kill rate
