"""Tests for multi-layer demonstration runner (T8.2).

Ensures that make demo and scripts/run_multi_layer_demo.py execute cleanly,
verifying Claim C1 bit-exact equivalence on all 7 projection types.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_multi_layer_demo import run_layer0_demo


def test_multi_layer_demo_execution(tmp_path: Path) -> None:
    """Execute multi-layer demo with small slice to assert PASS and verify JSON structure."""
    out_json = tmp_path / "test_demo_layer0.json"
    success = run_layer0_demo(
        slice_n=8,
        slice_k=16,
        num_vectors=5,
        output_json=out_json,
    )
    assert success is True
    assert out_json.is_file()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["all_passed"] is True
    assert data["claim_c1_verified"] is True
    assert len(data["projections"]) == 7

    for proj in data["projections"]:
        assert proj["status"] == "PASS"
        assert proj["mismatch"] == 0
        assert proj["port_audit_passed"] is True
        assert proj["const_audit_passed"] is True
