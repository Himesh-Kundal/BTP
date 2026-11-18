# Task UD.3 — Multi-Prompt Trace & Token Exactness

## Objective
Verify token-exactness and bit-exact intermediate trace equality across $\ge 100$ random prompts, including sequences up to `MAX_CTX` ($32$ positions for the tiny model). Validate the first-divergence diagnosis tool (`verification/trace_diff.py`) on an intentionally injected fault to prove precise failure localization.

## Implementation Details
1. **Multi-Prompt Testbench Generator & Runner (`sim/tb_generator.py`):**
   - Implemented `generate_multi_prompt_tb_verilog`: packs flat stimulus buffers and offset tables in Verilog, looping across prompts back-to-back with clean `seq_reset` on step 0 of each sequence.
   - Implemented `run_multi_prompt_simulation`: compiles and runs multi-prompt testbenches with Verilator or Icarus Verilog, parses per-sequence step outputs, verifies summary line `TB2_RESULT`, and compares dumped hierarchical traces.
2. **End-to-End Test Suite (`tests/test_tiny_model_end_to_end.py`):**
   - Tested 100 random prompts (99 short sequences of lengths 1–3 plus 1 long sequence with 16 prompt tokens + 16 generated tokens reaching `MAX_CTX = 32`). Total steps: 326.
   - Evaluated against Python compiled-artifact interpreter (`compiler/interpreter.py`), which was previously verified bit-exact against the `numerics/decoder.py` oracle.
   - Verified 100% token exactness (`res.tok_mismatch == 0`) across all 326 decode steps.
   - Checked 82,944 dumped trace elements bit for bit (`res.trace_mismatch == 0`, `res.diff_result.is_identical == True`).
   - Exercised `verification/trace_diff.py` on an injected fault in `seq0/tok1/L0/q_proj_q.hex` line 5, confirming that `compare_trace_directories` isolates the exact failure coordinates (`seq_id="seq0"`, `token_idx="tok1"`, `layer="L0"`, `op_name="q"`, `tensor_name="proj_q"`, `element_idx=5`).

## Verification & Acceptance
- Ran `pytest tests/test_tiny_model_end_to_end.py -v`:
  - `test_tiny_model_100_prompts_token_and_trace_exact`: PASSED
  - `test_tiny_model_first_divergence_tool_on_injected_fault`: PASSED
- Ran `make check2`: 144/144 tests passing.

## Status
Status: complete
Commit: UD.3: verify token and trace exactness across 100 prompts
