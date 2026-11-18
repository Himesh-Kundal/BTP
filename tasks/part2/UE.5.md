# Task UE.5 — Long-Context Verification up to MAX_CTX (256)

## Objective
Verify token-exact autoregressive decoding and sampled-layer trace equality for a long context sequence up to `MAX_CTX = 256` (prompt length 250, generating 6 tokens to reach position 255) on the full 30-layer SmolLM2-135M RTL accelerator.

## Implementation (`scripts/ue5_long_context.py`)
- Selected WikiText-2 prompt of length 250 tokens from evaluation corpus.
- Ran `CompiledArtifactInterpreter` up to position 255 (256 positions total), dumping golden traces for sampled positions `[0, 1, 127, 254, 255]` on Layer 0.
- Generated top-level Verilator testbench `tb_ue5.v` configured with `TOTAL_STEPS = 256` and timeout of 10B cycles.
- Executed Verilator `-O3` simulation and compared all 256 output tokens and 90 trace files (303,360 elements) against golden references.

## Verification & Results (`results/part2/ue5_long_context.json`)
- **Total Positions Decoded:** `256` (`prompt_len: 250`, `num_generate: 6`).
- **Token Exactness:** `100%` (`tokens_exact: true`, zero mismatches across all 256 positions).
- **Sampled-Layer Trace Parity (`L0` at pos `0, 1, 127, 254, 255`):** `trace_identical: true`, `first_divergence: null` (**90 files / 303,360 elements** checked bit-for-bit).
- **Total Simulated Cycles:** `4,669,315,072` cycles.
- **Simulation Speed:** `2,662.07 kcycles/s` (`2.66 MHz`).
- **Wall Time:** `1754.02s` (~29.2 minutes).
- **Status:** **PASS**.

## Status
Status: complete
Commit: UE.5: verify 256-position long-context decode and sampled traces up to MAX_CTX
