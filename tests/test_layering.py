"""Architectural layering and boundary tests (Golden Rule 5).

Enforces that `ir/`, `compiler/`, `backend/`, and `verification/` are strictly
model-agnostic:
- Never import `torch` or `transformers`.
- Never hardcode SmolLM model names or shapes.
Only `frontend/` and `examples/` are allowed to handle model-specific code.
"""

import ast
from pathlib import Path

PROTECTED_DIRS = ["ir", "compiler", "backend", "verification"]
FORBIDDEN_MODULES = {"torch", "transformers"}
FORBIDDEN_STRINGS = {"SmolLM", "smollm", "SmolLM2"}

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_source_for_forbidden_imports(source: str, filename: str = "<string>") -> list[str]:
    """Inspect Python AST for forbidden module imports."""
    violations = []
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        return [f"SyntaxError in {filename}: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_pkg = alias.name.split(".")[0]
                if base_pkg in FORBIDDEN_MODULES:
                    violations.append(
                        f"{filename}:{node.lineno} imports forbidden module '{alias.name}'"
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            base_pkg = node.module.split(".")[0]
            if base_pkg in FORBIDDEN_MODULES:
                violations.append(
                    f"{filename}:{node.lineno} imports from forbidden module '{node.module}'"
                )

    return violations


def check_source_for_forbidden_strings(source: str, filename: str = "<string>") -> list[str]:
    """Inspect Python AST string constants and identifiers for hardcoded model names."""
    violations = []
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        return [f"SyntaxError in {filename}: {e}"]

    for node in ast.walk(tree):
        # Check string literals
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for pattern in FORBIDDEN_STRINGS:
                if pattern in node.value:
                    violations.append(
                        f"{filename}:{node.lineno} contains hardcoded model string '{pattern}' in literal '{node.value}'"
                    )
        # Check variable/function names
        elif isinstance(node, ast.Name):
            for pattern in FORBIDDEN_STRINGS:
                if pattern.lower() in node.id.lower():
                    violations.append(
                        f"{filename}:{node.lineno} contains hardcoded model identifier '{node.id}'"
                    )

    return violations


def test_layering_in_codebase():
    """Verify that all files in protected directories comply with architectural layering."""
    all_violations = []

    for d in PROTECTED_DIRS:
        dir_path = REPO_ROOT / d
        if not dir_path.exists():
            continue
        for py_file in dir_path.rglob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            rel_path = str(py_file.relative_to(REPO_ROOT))
            all_violations.extend(check_source_for_forbidden_imports(source, rel_path))
            all_violations.extend(check_source_for_forbidden_strings(source, rel_path))

    assert not all_violations, "Architectural layering violations found:\n" + "\n".join(all_violations)


def test_layering_negative_control_import():
    """Ensure the layering checker catches forbidden imports (anti-gaming verification)."""
    dummy_code = "import torch\nimport numpy as np\n"
    violations = check_source_for_forbidden_imports(dummy_code, "dummy.py")
    assert len(violations) == 1
    assert "imports forbidden module 'torch'" in violations[0]

    dummy_from_code = "from transformers import AutoModelForCausalLM\n"
    violations2 = check_source_for_forbidden_imports(dummy_from_code, "dummy.py")
    assert len(violations2) == 1
    assert "imports from forbidden module 'transformers'" in violations2[0]


def test_layering_negative_control_model_strings():
    """Ensure the layering checker catches hardcoded model strings (anti-gaming verification)."""
    dummy_code = 'layer_name = "SmolLM2-135M_q_proj"\n'
    violations = check_source_for_forbidden_strings(dummy_code, "dummy.py")
    assert len(violations) >= 1
    assert any("SmolLM" in v for v in violations)
