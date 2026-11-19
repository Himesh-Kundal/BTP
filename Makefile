CARGO ?= cargo

.PHONY: all env-check check check2 lint test sim sim-tiny synth synth2 demo demo-fast demo2 demo2-fast bench clean

all: check

env-check:
	@bash scripts/check_env.sh

lint:
	@$(CARGO) clippy --workspace --all-targets -- -D warnings

test:
	@$(CARGO) test --workspace --release

check: lint test
	@echo "=== make check: PASSED (100% Native Rust Workspace) ==="

check2: check
	@echo "=== make check2: PASSED (100% Native Rust Workspace) ==="

sim:
	@$(CARGO) run --release --bin llmhw -- verify --bundle build/tiny_hw --tokens "1,10,25" --max-new-tokens 2

sim-tiny: demo-fast

synth:
	@$(CARGO) run --release --bin llmhw -- synth --unit rmsnorm --target generic --out results/part2/cli_synth_rmsnorm.json

synth2: synth

demo-fast:
	@$(CARGO) run --release --bin llmhw -- demo-fast

demo2-fast: demo-fast

demo:
	@$(CARGO) run --release --bin llmhw -- demo

demo2: demo

bench:
	@$(CARGO) run --release --bin llmhw -- differential-bench --out results/part2/rust_port_benchmarks.json

clean:
	@$(CARGO) clean
	rm -rf build/tiny_hw/ *.vcd
