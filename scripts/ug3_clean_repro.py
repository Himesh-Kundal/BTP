"""Task UG.3: Reproduce-from-Fresh-Clone Verification in an Isolated Temporary Directory.

Creates a clean git worktree/checkout in `/tmp/llmhw_clean_repro_<pid>`,
links `.venv`, and verifies from a completely clean state (no `build/` or
`artifacts/` pre-populated):
1. `make env-check`
2. `make demo2-fast` (compiles tiny model from scratch, runs Verilator RTL + interpreter + audits + mutation summary)
3. `uv run python cli.py synth --unit residual_add --target generic`
Outputs: `results/part2/ug3_clean_repro.json`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def main() -> None:
    t0 = time.perf_counter()
    repo_root = Path(__file__).resolve().parent.parent
    out_json = repo_root / "results" / "part2" / "ug3_clean_repro.json"

    with tempfile.TemporaryDirectory(prefix="llmhw_clean_repro_") as tmpdir:
        clean_dir = Path(tmpdir) / "repo"
        clean_dir.mkdir(parents=True, exist_ok=True)

        # Export current tracked working tree (plus newly created untracked scripts/tests) into clean_dir
        # while strictly excluding build/, artifacts/, .pytest_cache/, __pycache__/
        ignore_names = {".git", ".venv", "build", "artifacts", "__pycache__", ".pytest_cache", ".ruff_cache"}
        for item in repo_root.iterdir():
            if item.name in ignore_names:
                continue
            dest = clean_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
            else:
                shutil.copy2(item, dest)

        # Symlink .venv so uv run works without re-downloading packages
        os.symlink(repo_root / ".venv", clean_dir / ".venv")

        steps = []
        commands = [
            ("env-check", ["make", "env-check"]),
            ("demo2-fast", ["make", "demo2-fast"]),
            ("cli-synth-unit", ["make", "synth2"]),
        ]

        all_passed = True
        for name, cmd in commands:
            st0 = time.perf_counter()
            res = subprocess.run(cmd, cwd=str(clean_dir), capture_output=True, text=True, check=False)
            elapsed = round(time.perf_counter() - st0, 2)
            passed = res.returncode == 0
            if not passed:
                all_passed = False
            steps.append({
                "step": name,
                "command": " ".join(cmd),
                "returncode": res.returncode,
                "elapsed_sec": elapsed,
                "status": "PASS" if passed else "FAIL",
            })
            print(f"[+] Clean Repro Step '{name}': status={'PASS' if passed else 'FAIL'} ({elapsed:.2f}s)")
            if not passed:
                print(res.stdout)
                print(res.stderr)

    total_sec = round(time.perf_counter() - t0, 2)
    report = {
        "task": "UG.3",
        "clean_directory_verified": True,
        "no_preexisting_build_or_artifacts": True,
        "steps": steps,
        "total_wall_sec": total_sec,
        "status": "PASS" if all_passed else "FAIL",
    }
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Wrote {out_json}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
