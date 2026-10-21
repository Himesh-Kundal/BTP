"""Tests for synthesis fairness protocol enforcement (T7.2, docs/06 §B2).

Verifies:
1. Signal width conformance (Rule N10): x (8-bit), w (8-bit), prod (16-bit), acc (32-bit).
2. Negative controls for width conformance: corrupted widths trigger detection.
3. Synthesis pair fairness audit: identical flags, targets, and dual baseline accounting.
4. Negative controls for synthesis pair audit: divergent flags or missing accountings trigger failure.
"""

from pathlib import Path

import numpy as np
import pytest

from backend.baseline_p import generate_full_module_p, generate_full_module_p_bus
from backend.module_generator import generate_full_module
from backend.tiling import tile_weight_matrix
from ir.linear import LinearOp
from synth.fairness import audit_synthesis_pair, check_width_conformance
from synth.yosys_runner import synthesize_design


def test_width_conformance_positive() -> None:
    """Verify that Design F, P, and P_bus comply with Rule N10 width mandates."""
    w = np.array([[12, -7, 0, 15], [-3, 8, 22, -1]], dtype=np.int8)
    op = LinearOp(name="fair_test", in_features=4, out_features=2, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=2)

    rtl_f = generate_full_module(op, manifest)
    rtl_p = generate_full_module_p(op, manifest)
    rtl_p_bus = generate_full_module_p_bus(op, manifest)

    assert check_width_conformance(rtl_f, "Design F") == []
    assert check_width_conformance(rtl_p, "Design P") == []
    assert check_width_conformance(rtl_p_bus, "Design P_bus") == []


def test_width_conformance_negative_controls() -> None:
    """Verify that corrupting input, product, or accumulator widths triggers violation detection."""
    w = np.array([[10, 20], [30, 40]], dtype=np.int8)
    op = LinearOp(name="neg_width", in_features=2, out_features=2, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=2)
    rtl = generate_full_module(op, manifest)

    # 1. Corrupt input wire: [7:0] -> [15:0]
    bad_x = rtl.replace("wire signed [7:0] x_", "wire signed [15:0] x_")
    errs_x = check_width_conformance(bad_x)
    assert len(errs_x) > 0
    assert any("Input wire x width is 16 bits" in e for e in errs_x)

    # 2. Corrupt product wire: [15:0] -> [31:0]
    bad_prod = rtl.replace("wire signed [15:0] p_", "wire signed [31:0] p_")
    errs_prod = check_width_conformance(bad_prod)
    assert len(errs_prod) > 0
    assert any("Product wire width is 32 bits" in e for e in errs_prod)

    # 3. Corrupt accumulator wire: [31:0] -> [15:0]
    bad_acc = rtl.replace("wire signed [31:0] y_row_", "wire signed [15:0] y_row_")
    errs_acc = check_width_conformance(bad_acc)
    assert len(errs_acc) > 0
    assert any("Accumulator wire width is 16 bits" in e for e in errs_acc)


def test_audit_synthesis_pair_positive(tmp_path: Path) -> None:
    """Verify that compliant synthesis runs pass the fairness audit."""
    w = np.array([[12, -7], [8, 15]], dtype=np.int8)
    op = LinearOp(name="fair_pair", in_features=2, out_features=2, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=2)

    res_f = synthesize_design(op, manifest, "F", "generic", work_dir=tmp_path / "f")
    res_p = synthesize_design(op, manifest, "P", "generic", work_dir=tmp_path / "p")
    res_p_bus = synthesize_design(op, manifest, "P_bus", "generic", work_dir=tmp_path / "p_bus")

    report = audit_synthesis_pair(res_f, res_p, res_p_bus, strict=True)
    assert report.passed is True
    assert len(report.violations) == 0
    assert report.checks_performed >= 5


def test_audit_synthesis_pair_negative_controls(tmp_path: Path) -> None:
    """Verify that divergent flags, targets, or missing accountings trigger fairness violations."""
    w = np.array([[12, -7], [8, 15]], dtype=np.int8)
    op = LinearOp(name="neg_pair", in_features=2, out_features=2, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=2)

    res_f_gen = synthesize_design(
        op, manifest, "F", "generic", flatten=True, work_dir=tmp_path / "f_gen"
    )
    res_p_ice = synthesize_design(
        op, manifest, "P", "ice40", flatten=True, work_dir=tmp_path / "p_ice"
    )
    res_p_bus_noflat = synthesize_design(
        op, manifest, "P_bus", "generic", flatten=False, work_dir=tmp_path / "p_noflat"
    )

    # 1. Target mismatch (generic vs ice40)
    rep_target = audit_synthesis_pair(res_f_gen, res_p_ice, strict=False)
    assert rep_target.passed is False
    assert any("Target mismatch" in v for v in rep_target.violations)

    # 2. Flatten mismatch (True vs False)
    rep_flatten = audit_synthesis_pair(res_f_gen, res_p_bus=res_p_bus_noflat, strict=False)
    assert rep_flatten.passed is False
    assert any("Flatten mismatch" in v for v in rep_flatten.violations)

    # 3. Missing dual accounting (only P without P_bus)
    rep_missing = audit_synthesis_pair(res_f_gen, res_p_ice, None, strict=False)
    assert rep_missing.passed is False
    assert any("Fairness Rule 4 violation" in v for v in rep_missing.violations)

    # 4. Strict mode raises ValueError
    with pytest.raises(ValueError, match="Fairness Protocol Violation"):
        audit_synthesis_pair(res_f_gen, res_p_ice, strict=True)
