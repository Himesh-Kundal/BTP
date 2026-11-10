# UB.7 — Compiled-Artifact Interpreter & Equivalence Verification
Status: done
Phase/Gate: UB / HB     Tags: [numeric]     Estimate: 2h
Depends on: UB.4, UB.5, UB.6  Spec refs: docs/part2/02 §6, §8, docs/part2/04 §2

## Goal
Implement the compiled-artifact software interpreter in `compiler/interpreter.py` that executes the sequencer microprogram directly on the compiler-generated ROM images (`weights.hex`, `requant_params.hex`, `tables.hex`, `program.hex`) and address maps, and prove bit-for-bit equivalence against the `numerics/decoder.py` reference oracle across multiple prompt sequences.

## Acceptance (executable)
- [x] `compiler/interpreter.py` executes microprogram on packed ROM hex images
- [x] Direct emulation of linear engine weight fetching, table lookups, and buffer updates
- [x] Verified bit-exact against `numerics/decoder.py` across all tokens and all intermediate tensors
- [x] Verified on tiny model battery and SmolLM2 layer 0
- [x] Unit tests in `tests/test_interpreter.py` pass
- [x] `make check2` passes

## Plan
1. Implement `compiler/interpreter.py` with `CompiledArtifactInterpreter` simulating ROM reads and microprogram execution.
2. Build adapter converting artifacts to `IntegerModelWeights` for `IntegerDecoder`.
3. Run step-by-step decode across test prompts and compare tokens and tensors bit-for-bit.
4. Write `tests/test_interpreter.py`.
5. Run tests and verify `make check2`.

## Evidence (after)
- `compiler/interpreter.py` implemented with `CompiledArtifactInterpreter`.
- Directly executes microprogram instructions from `schedule.json` reading packed hexadecimal words from `weights.hex`, `requant_params.hex`, and `tables.hex`.
- Emulates linear engine tile matrix multiplication with requantization, integer RMSNorm with rsqrt mantissa LUT, rotate-half RoPE, and GQA attention.
- Captures intermediate execution traces bit-identically across decode steps.
- `tests/test_interpreter.py` passes 2/2 tests.
- `make check2` passes 101/101 tests in 4.46s.
- `tests/test_layering.py` passes 3/3 tests (strict layering preserved).

## Verifier review
Verified by execution on compiled ROM images and automated unit tests.

