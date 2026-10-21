UV ?= $(HOME)/.local/bin/uv
PYTHON ?= PYTHONPATH=. $(UV) run python
PYTEST ?= $(UV) run pytest
RUFF ?= $(UV) run ruff

.PHONY: all env-check check lint test sim regression synth demo demo-toy clean

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

demo-toy:
	@$(PYTHON) examples/compile_linear.py --toy

demo:
	@echo "Demo target: placeholder for T8.2"

clean:
	rm -rf build/ artifacts/ *.vcd __pycache__ .pytest_cache
