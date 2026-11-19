#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"
mkdir -p "${RESULTS_DIR}"

echo "=== Checking Environment & Toolchain (Rust Edition) ==="

# 1. Shell / EDA tools
IVERILOG_VER=$(iverilog -V 2>&1 | head -n 1 || true)
VVP_VER=$(vvp -V 2>&1 | head -n 1 || true)
VERILATOR_VER=$(verilator --version 2>&1 | head -n 1 || true)
YOSYS_VER=$(yosys -V 2>&1 | head -n 1 || true)

echo "iverilog:   ${IVERILOG_VER}"
echo "vvp:        ${VVP_VER}"
echo "verilator:  ${VERILATOR_VER}"
echo "yosys:      ${YOSYS_VER}"

for tool in iverilog vvp verilator yosys; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: Missing required EDA tool: $tool" >&2
        exit 1
    fi
done

# 2. Rust toolchain
RUSTC_VER=$(rustc --version 2>&1 | head -n 1 || true)
CARGO_VER=$(cargo --version 2>&1 | head -n 1 || true)
CLIPPY_VER=$(cargo clippy --version 2>&1 | head -n 1 || true)

echo "rustc:      ${RUSTC_VER}"
echo "cargo:      ${CARGO_VER}"
echo "clippy:     ${CLIPPY_VER}"

for tool in rustc cargo; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: Missing required Rust tool: $tool" >&2
        exit 1
    fi
done

# 3. Git commit
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

cat <<EOF > "${RESULTS_DIR}/env.json"
{
  "git_commit": "${GIT_COMMIT}",
  "tools": {
    "iverilog": "${IVERILOG_VER}",
    "vvp": "${VVP_VER}",
    "verilator": "${VERILATOR_VER}",
    "yosys": "${YOSYS_VER}"
  },
  "rust": {
    "rustc": "${RUSTC_VER}",
    "cargo": "${CARGO_VER}",
    "clippy": "${CLIPPY_VER}"
  },
  "model": {
    "name": "HuggingFaceTB/SmolLM2-135M",
    "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
  }
}
EOF

echo "Wrote environment info to ${RESULTS_DIR}/env.json"
echo "=== Environment Check PASSED ==="
