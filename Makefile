UV ?= $(HOME)/.local/bin/uv
PYTHON ?= PYTHONPATH=. $(UV) run python
PYTEST ?= $(UV) run pytest
RUFF ?= $(UV) run ruff

.PHONY: all env-check check check2 lint test sim sim-tiny regression synth report-figs demo demo-toy demo2 demo2-fast numerics budgets clean

all: check

env-check:
	@bash scripts/check_env.sh

lint:
	@$(RUFF) check .

test:
	@$(PYTEST) -v

check: lint test
	@echo "=== make check: PASSED ==="

check2: lint
	@$(PYTEST) -v tests/test_smoke.py tests/test_layering.py tests/test_tiny_model.py tests/test_reference.py tests/test_weight_encoder.py tests/test_ir_complete.py tests/test_tiling.py
	@echo "=== make check2: PASSED ==="

numerics:
	@$(PYTEST) -v -k "numerics or reference or tiny_model"

sim:
	@$(PYTHON) scripts/run_toy_sim.py

sim-tiny:
	@$(PYTHON) -c "from frontend.tiny_model import dump_tiny_model_artifacts; dump_tiny_model_artifacts(); print('[+] Tiny model artifacts generated and verified.')"

regression:
	@$(PYTHON) -m verification.regression_runner

synth:
	@$(PYTHON) -m synth.yosys_runner

report-figs:
	@$(PYTHON) scripts/generate_report_figs.py

budgets:
	@$(PYTHON) scripts/measure_budgets.py

demo-toy:
	@$(PYTHON) examples/compile_linear.py --toy

demo:
	@$(PYTHON) scripts/run_multi_layer_demo.py

demo2-fast:
	@$(PYTHON) -c "from frontend.tiny_model import tiny_model_forward_fp32; out = tiny_model_forward_fp32([1, 10, 25]); print('Tiny Model Demo PASS: prompt=[1, 10, 25] -> next_token =', out['next_token_id'])"

demo2:
	@echo "[*] SmolLM2 full decode target (available after Phase UE)"

clean:
	rm -rf build/ artifacts/ *.vcd __pycache__ .pytest_cache
