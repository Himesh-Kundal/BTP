UV ?= $(HOME)/.local/bin/uv
PYTHON ?= PYTHONPATH=. $(UV) run python
PYTEST ?= $(UV) run pytest
RUFF ?= $(UV) run ruff

.PHONY: all env-check check lint test sim regression synth report-figs demo demo-toy clean

all: check

env-check:
	@bash scripts/check_env.sh

lint:
	@$(RUFF) check .

test:
	@$(PYTEST) -v

check: lint test
	@echo "=== make check: PASSED ==="

sim:
	@$(PYTHON) scripts/run_toy_sim.py

regression:
	@$(PYTHON) -m verification.regression_runner

synth:
	@$(PYTHON) -m synth.yosys_runner

report-figs:
	@$(PYTHON) scripts/generate_report_figs.py

demo-toy:
	@$(PYTHON) examples/compile_linear.py --toy

demo:
	@$(PYTHON) scripts/run_multi_layer_demo.py

clean:
	rm -rf build/ artifacts/ *.vcd __pycache__ .pytest_cache
