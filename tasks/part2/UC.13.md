# UC.13 — Mutation Suite v2 per Unit
Status: done
Phase/Gate: UC / HC     Tags: [verification, mutation]     Estimate: 4h
Depends on: UC.3–UC.12   Spec refs: docs/part2/06 §A6, docs/part2/04 §UC.13

## Goal
Implement `verification/mutate2.py` and test suite `tests/test_mutation_suite_v2.py`:
1. Injects 21 text-level synthetic hardware faults (Mutants M10 through M30) across all RTL units implemented in UC.1–UC.12.
2. Runs the unit verification battery against each mutant.
3. Enforces 100% kill rate (all 21 mutants caught and killed by the test suite).
4. Saves the complete kill matrix to `results/mutation_matrix_v2.json`.

## Mutant Definitions (docs/part2/06 §A6)
- **M10**: `numerics.vh` / `requant.v`: Requant rounds half-down instead of half-up.
- **M11**: `numerics.vh` / `requant.v`: Asymmetric saturation to -128 instead of symmetric -127.
- **M12**: `rope.v`: Pairs adjacent elements `(2i, 2i+1)` instead of split halves `(i, i+d/2)`.
- **M13**: `rope.v`: Table address off by one (`dim + 1`).
- **M14**: `attention.v`: Key stored before RoPE transformation.
- **M15**: `kv_addr_gen.v` / `attention.v`: GQA head mapping uses modulo `i % 3` instead of floor divide `i // 3`.
- **M16**: `attention.v`: Causal mask off (attends to future position `pos + 1`).
- **M17**: `softmax.v`: Softmax computes base-2 exp without maximum score subtraction.
- **M18**: `softmax.v`: Softmax exp2 LUT address off by one (`diff + 1`).
- **M19**: `rmsnorm.v`: Rsqrt LUT address wrong (off by one or unshifted).
- **M20**: `rmsnorm.v`: RMSNorm skips gamma scaling (multiplies by 1 instead of gamma).
- **M21**: `swiglu.v`: SiLU LUT sign inversion.
- **M22**: `residual_add.v`: Residual addition bypassed / skipped.
- **M23**: `argmax.v`: Argmax ties break to highest index instead of lowest.
- **M24**: `argmax.v`: Argmax compares raw unscaled values instead of dyadic scaled.
- **M25**: `sequencer.v`: ROM base address offset corrupted / swapped.
- **M26**: `kv_cache.v`: KV cache writes to future position `pos + 1`.
- **M27**: `embed_gather.v`: Embedding row gather index off by one (`tok_id + 1`).
- **M28**: `linear_engine.v`: Linear engine accumulator truncated to 16 bits.
- **M29**: `linear_engine.v`: Linear engine padding lane injects non-zero value.
- **M30**: `sequencer.v`: Handshake break / premature done assertion.

## Acceptance (executable)
- [x] `verification/mutate2.py` implemented covering all 21 mutant fault models M10–M30.
- [x] 100% kill rate (21/21 mutants killed) verified by test suite.
- [x] Kill matrix saved to `results/mutation_matrix_v2.json`.
- [x] `tests/test_mutation_suite_v2.py` passes and is added to `make check2`.

## Evidence
- `verification/mutate2.py` and `tests/test_mutation_suite_v2.py` execute full campaign across M10 through M30.
- All 21 mutants caught and killed:
  - Total Mutants: 21
  - Killed: 21
  - Survived: 0
  - Kill Rate: 100.0%
- Kill matrix serialized to `results/mutation_matrix_v2.json`.
- `make check2` passes 136/136 tests green.
