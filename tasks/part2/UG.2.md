# Task UG.2 — End-to-End Demo Targets (`make demo2-fast` and `make demo2`)

## Objective
Provide and verify turnkey Makefile demonstration targets `make demo2-fast` (tiny decoder end-to-end RTL simulation, audits, and mutation kill matrix in seconds) and `make demo2` (full 30-layer SmolLM2-135M autoregressive text generation in Verilator RTL simulation compared against both the Python integer oracle and HuggingFace FP32 greedy generation).

## Implementation (`Makefile`, `scripts/ug2_demo2_fast.py`, `scripts/ug2_demo2.py`)
1. **`make demo2-fast` (`scripts/ug2_demo2_fast.py`):**
   - Compiles the 2-layer tiny decoder via `llmhw compile-model --tiny --out build/demo2_fast_hw`.
   - Runs 4 new tokens from `prompt=[1, 10, 25]` on both `CompiledArtifactInterpreter` and Verilator RTL (`llmhw run --backend both`).
   - Executes all 4 structural & ROM immutability audits (`llmhw verify --bundle`).
   - Reports the unit mutation kill matrix (`21/21` killed, 100.0%) and system mutation kill matrix (`7/7` killed, 100.0%).
   - Writes `results/part2/demo2_fast.json`.
2. **`make demo2` (`scripts/ug2_demo2.py`):**
   - Auto-generates `build/smollm2_hw` via `scripts/ue1_generate_full.py` if missing (fresh-clone support).
   - Executes 4 new tokens on prompt `"The capital of France is Paris. The capital of Germany is"` across all 30 layers of SmolLM2-135M in Verilator RTL simulation (`top_accel.v`), Python `CompiledArtifactInterpreter`, and HuggingFace PyTorch FP32.
   - Verifies 100% bit-exact token equality between RTL and the integer oracle (`[11452, 30, 378, 3575]` $\to$ `' Berlin. The capital'`) and 4/4 greedy match against FP32.
   - Writes `results/part2/demo2.json`.

## Verified Runtimes & Evidence
- `make demo2-fast`: **11.45 seconds** (`115,227` RTL cycles, `tokens_exact: true`, `21/21` unit mutants killed, `7/7` system mutants killed).
- `make demo2`: **106.16 seconds** (`221,848,192` RTL cycles in `94.31s` simulation wall time, `RTL == Oracle: True`, `4/4` match with FP32 `' Berlin. The capital'`).
- `make synth2`: **38.90 seconds** (`rmsnorm.v` Yosys generic synthesis: `51,656` total cells, `362` DFFs).

## Status
Status: complete
