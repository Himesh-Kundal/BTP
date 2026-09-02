#!/usr/bin/env bash
set -euo pipefail

# Find python/uv
UV_BIN="${HOME}/.local/bin/uv"
if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"
mkdir -p "${RESULTS_DIR}"

echo "=== Checking Environment & Toolchain ==="

# 1. Shell / EDA tools
IVERILOG_VER=$(iverilog -V 2>&1 | head -n 1 || true)
VVP_VER=$(vvp -V 2>&1 | head -n 1 || true)
VERILATOR_VER=$(verilator --version 2>&1 | head -n 1 || true)
YOSYS_VER=$(yosys -V 2>&1 | head -n 1 || true)

echo "iverilog:   ${IVERILOG_VER}"
echo "vvp:        ${VVP_VER}"
echo "verilator:  ${VERILATOR_VER}"
echo "yosys:      ${YOSYS_VER}"

# Check for mandatory EDA tools
for tool in iverilog vvp verilator yosys; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: Missing required EDA tool: $tool" >&2
        exit 1
    fi
done

# 2. Python environment via uv
if [ ! -x "${UV_BIN}" ]; then
    echo "ERROR: uv not found at ${UV_BIN}" >&2
    exit 1
fi

RUFF_VER=$("${UV_BIN}" run ruff --version 2>&1 | head -n 1 || true)

PY_INFO=$("${UV_BIN}" run python -c '
import sys, json, torch, transformers, numpy, pytest
info = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "numpy": numpy.__version__,
    "pytest": pytest.__version__
}
print(json.dumps(info))
')

echo "python:     $(echo "$PY_INFO" | grep -o '"python": "[^"]*' | cut -d'"' -f4)"
echo "torch:      $(echo "$PY_INFO" | grep -o '"torch": "[^"]*' | cut -d'"' -f4)"
echo "transformers: $(echo "$PY_INFO" | grep -o '"transformers": "[^"]*' | cut -d'"' -f4)"
echo "numpy:      $(echo "$PY_INFO" | grep -o '"numpy": "[^"]*' | cut -d'"' -f4)"
echo "pytest:     $(echo "$PY_INFO" | grep -o '"pytest": "[^"]*' | cut -d'"' -f4)"
echo "ruff:       ${RUFF_VER}"

# 3. Git commit
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

# 4. Generate results/env.json
"${UV_BIN}" run python -c "
import json, sys

py_info = json.loads('''$PY_INFO''')
py_info['ruff'] = '$RUFF_VER'
env_data = {
    'git_commit': '$GIT_COMMIT',
    'tools': {
        'iverilog': '$IVERILOG_VER',
        'vvp': '$VVP_VER',
        'verilator': '$VERILATOR_VER',
        'yosys': '$YOSYS_VER'
    },
    'python': py_info,
    'model': {
        'name': 'HuggingFaceTB/SmolLM2-135M',
        'revision': 'pinned'
    }
}
with open('${RESULTS_DIR}/env.json', 'w') as f:
    json.dump(env_data, f, indent=2)
print('Wrote environment info to ${RESULTS_DIR}/env.json')
"

echo "=== Environment Check PASSED ==="
