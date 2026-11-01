"""Tests for regression test runner (T6.6).

Verifies:
1. End-to-end execution of regression runner across all 7 verification suites.
2. Structured results/regression.json schema and metadata validation.
3. CLI execution via subprocess with --quick and custom output path.
"""

import json
import subprocess
import sys
from pathlib import Path

from verification.regression_runner import run_regression


def test_regression_runner_fast_mode(tmp_path: Path) -> None:
    """Verify regression runner executes cleanly and produces structured JSON in fast mode."""
    repo_root = Path(__file__).resolve().parent.parent
    out_json = tmp_path / "test_reg.json"

    report = run_regression(
        repo_root=repo_root,
        output_path=out_json,
        fast_mode=True,
        verbose=False,
    )

    assert report["summary"]["status"] == "PASS"
    assert report["summary"]["total_suites"] == 7
    assert report["summary"]["failed_tests"] == 0
    assert report["summary"]["total_tests"] > 0
    assert report["summary"]["total_vectors_checked"] > 0

    assert out_json.exists()
    with open(out_json, encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["metadata"]["git_commit"] != ""
    assert "tools" in loaded["metadata"]
    assert "iverilog" in loaded["metadata"]["tools"]
    assert len(loaded["suites"]) == 7

    # Ensure all suites passed
    for suite in loaded["suites"]:
        assert suite["status"] == "PASS", f"Suite {suite['name']} failed"
        assert suite["failed"] == 0


def test_regression_runner_cli(tmp_path: Path) -> None:
    """Verify regression runner CLI entry point returns exit code 0."""
    repo_root = Path(__file__).resolve().parent.parent
    out_json = tmp_path / "cli_reg.json"

    cmd = [
        sys.executable,
        "-m",
        "verification.regression_runner",
        "--quick",
        "--output",
        str(out_json),
        "--quiet",
    ]
    res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Regression runner CLI failed:\n{res.stdout}\n{res.stderr}"
    assert out_json.exists()
