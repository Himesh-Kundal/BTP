"""Tests for varying-shape demonstration runner (T8.3).

Ensures that scripts/run_varying_shape_demo.py executes cleanly across
full-layer, non-multiple, and asymmetric matrix dimensions.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_varying_shape_demo import run_varying_shape_demo


def test_varying_shape_demo_execution(tmp_path: Path) -> None:
    """Execute varying shape demo and assert all 3 cases pass."""
    out_json = tmp_path / "test_demo.json"
    success = run_varying_shape_demo(output_json=out_json)
    assert success is True
    assert out_json.is_file()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["all_passed"] is True
    assert len(data["cases"]) == 3

    cases = {c["case"]: c for c in data["cases"]}
    assert "full_layer_576x576" in cases
    assert "non_multiple_19x37" in cases
    assert "asymmetric_32x128" in cases

    for c in cases.values():
        assert c["status"] == "PASS"
        assert c["audit_passed"] is True
