# Task UG.1 — Unified CLI v2 (`compile-model`, `run`, `verify`, `synth`)

## Objective
Extend the unified `llmhw` command-line interface (`cli.py`) to support the full Part II model-specific transformer decoder pipeline alongside Part I linear layer compilation, and verify all subcommands with unit and integration tests in `tests/test_cli.py`.

## Implementation (`cli.py`, `tests/test_cli.py`)
- **`llmhw compile-model`**: Compiles either the deterministic 2-layer tiny decoder fixture (`--tiny`) or the full 30-layer SmolLM2-135M model (`--model`, `--revision`, `--tile-r`, `--tile-c`, `--max-ctx`) into a complete ROM-backed Verilog accelerator bundle (`top_accel.v`, `weights.hex`, `weights_128b.hex`, `requant_params.hex`, `tables_*.hex`, `program.hex`, `layer_bases.hex`, `schedule.json`, `address_map.json`, `memory_plan.json`).
- **`llmhw run`**: Executes autoregressive token generation on any compiled v2 bundle using `--backend interpreter` (fast bit-exact software oracle), `--backend rtl` (Verilator hardware simulation), or `--backend both` (cross-verifies 100% bit-exact parity step-by-step).
- **`llmhw verify --bundle <dir>`**: Runs the four Part II hardware integrity audits (`audit_rom_module_source`, `audit_no_expected_values_in_rtl`, `audit_top_accel_memory_structure`, `audit_rom_hex_images`) plus Verilator RTL vs `CompiledArtifactInterpreter` token-exact verification.
- **`llmhw synth --unit <name>`**: Synthesizes any Part II hardware unit in `rtl/lib/` (automatically resolving `numerics.vh` and sub-modules) via Yosys for `generic` or `ice40` targets.

## Evidence
- `env PYTHONPATH=. uv run pytest tests/test_cli.py tests/test_layering.py -v`: **15/15 tests PASSED**.

## Status
Status: complete
