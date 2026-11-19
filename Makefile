UV ?= $(HOME)/.local/bin/uv
PYTHON ?= PYTHONPATH=. $(UV) run python
PYTEST ?= $(UV) run pytest
RUFF ?= $(UV) run ruff

.PHONY: all env-check check check2 check-rust bench-rust lint test sim sim-tiny regression synth synth2 report-figs demo demo-toy demo2 demo2-fast numerics budgets clean

all: check

env-check:
	@bash scripts/check_env.sh

lint:
	@$(RUFF) check .

test:
	@$(PYTEST) -v

check: lint test
	@echo "=== make check: PASSED ==="

check-rust:
	@cargo clippy --workspace --all-targets -- -D warnings
	@cargo test --workspace --release
	@echo "=== make check-rust: PASSED ==="

bench-rust:
	@cargo run --release --bin llmhw -- differential-bench --out results/part2/rust_port_benchmarks.json

check2: lint
	@$(PYTEST) -v tests/test_smoke.py tests/test_layering.py tests/test_tiny_model.py tests/test_fp32_decoder.py tests/test_eval_harness.py tests/test_numerics_ops.py tests/test_integer_decoder.py tests/test_trace_diff.py tests/test_ir_v2.py tests/test_importer_v2.py tests/test_format_assignment.py tests/test_rom_packing.py tests/test_schedule.py tests/test_memory_planner.py tests/test_interpreter.py tests/test_reference.py tests/test_weight_encoder.py tests/test_ir_complete.py tests/test_tiling.py tests/test_rtl_numerics.py tests/test_weight_audit_v2.py tests/test_rtl_rmsnorm.py tests/test_rtl_rope.py tests/test_rtl_softmax.py tests/test_rtl_swiglu.py tests/test_rtl_residual_embed.py tests/test_rtl_argmax.py tests/test_rtl_kv_cache.py tests/test_rtl_linear_engine.py tests/test_rtl_attention.py tests/test_rtl_sequencer.py tests/test_mutation_suite_v2.py tests/test_top_generator.py tests/test_tb_generator.py tests/test_tiny_model_end_to_end.py tests/test_cross_simulator.py tests/test_top_mutation.py tests/test_audits_v2.py tests/test_cli.py
	@echo "=== make check2: PASSED ==="

numerics:
	@$(PYTEST) -v -k "numerics or reference or tiny_model"

sim:
	@$(PYTHON) scripts/run_toy_sim.py

sim-tiny:
	@$(PYTHON) cli.py compile-model --tiny --out build/tiny_hw && $(PYTHON) cli.py verify --bundle build/tiny_hw --tokens "1,10,25" --max-new-tokens 2 --out results/part2/sim_tiny_verify.json

regression:
	@$(PYTHON) -m verification.regression_runner

synth:
	@$(PYTHON) -m synth.yosys_runner

synth2:
	@$(PYTHON) cli.py synth --unit rmsnorm --target generic --out results/part2/cli_synth_rmsnorm.json

report-figs:
	@$(PYTHON) scripts/generate_report_figs.py

budgets:
	@$(PYTHON) scripts/measure_budgets.py

demo-toy:
	@$(PYTHON) examples/compile_linear.py --toy

demo:
	@$(PYTHON) scripts/run_multi_layer_demo.py

demo2-fast:
	@$(PYTHON) scripts/ug2_demo2_fast.py

demo2:
	@$(PYTHON) scripts/ug2_demo2.py

clean:
	rm -rf build/ artifacts/ *.vcd __pycache__ .pytest_cache

