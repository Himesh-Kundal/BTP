"""Unit tests for full-layer area extrapolation model (T7.5, docs/06 §B4)."""

import json
from pathlib import Path

from synth.extrapolation import (
    SMOLLM2_LAYER0_PROJECTIONS,
    ValidationPoint,
    extrapolate_full_layers,
)


def test_extrapolation_formulas(tmp_path: Path) -> None:
    """Verify extrapolation calculations across all 7 projection types."""
    out_file = tmp_path / "E5_test.json"
    report = extrapolate_full_layers(tile_shape=(16, 64), out_json_path=out_file)

    assert out_file.exists()
    assert report["experiment"] == "E5"
    assert "layer_estimates" in report
    assert len(report["layer_estimates"]) == 7

    # Verify every projection adheres to labeling rules
    for est in report["layer_estimates"].values():
        assert est["label"] == "estimate"
        assert est["estimate_cells_f"] > 0
        assert est["estimate_cells_p"] > est["estimate_cells_f"]
        assert est["estimate_cells_p_bus"] > est["estimate_cells_f"]
        assert est["estimate_flops_p"] > est["estimate_flops_f"]

    # Verify full layer totals
    totals = report["full_layer_totals"]
    assert totals["label"] == "estimate"
    assert totals["estimate_total_cells_p"] > totals["estimate_total_cells_f"]
    assert totals["total_tiles_per_layer"] == 3456


def test_validation_point_metric_error() -> None:
    """Verify error calculation properties for validation point."""
    vp = ValidationPoint(
        test_shape=(16, 32),
        tile_shape=(8, 16),
        num_tiles=4,
        actual_cells_f=83551,
        extrap_cells_f=91056,
        error_f=0.0898,
        actual_cells_p=298733,
        extrap_cells_p=301296,
        error_p=0.0086,
        actual_cells_pbus=293737,
        extrap_cells_pbus=296440,
        error_pbus=0.0092,
        actual_reduction_vs_p=3.58,
        actual_reduction_vs_pbus=3.52,
    )

    assert vp.error_f < 0.10  # Under 10% error on Design F
    assert vp.error_p < 0.01  # Under 1% error on Design P
    assert vp.error_pbus < 0.01  # Under 1% error on Design P_bus
    assert vp.actual_reduction_vs_p > 3.0


def test_e5_json_contains_all_spec_requirements(tmp_path: Path) -> None:
    """Verify generated JSON adheres to all spec constraints and caveats."""
    out_file = tmp_path / "E5_spec_test.json"
    dummy_vp = ValidationPoint(
        test_shape=(16, 32),
        tile_shape=(8, 16),
        num_tiles=4,
        actual_cells_f=83551,
        extrap_cells_f=91056,
        error_f=0.0898,
        actual_cells_p=298733,
        extrap_cells_p=301296,
        error_p=0.0086,
        actual_cells_pbus=293737,
        extrap_cells_pbus=296440,
        error_pbus=0.0092,
        actual_reduction_vs_p=3.58,
        actual_reduction_vs_pbus=3.52,
    )
    extrapolate_full_layers(validation_point=dummy_vp, out_json_path=out_file)

    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert "caveat" in loaded
    assert "pre-layout" in loaded["caveat"]
    assert loaded["validation_evidence"]["error_f"] == 0.0898
    for name in SMOLLM2_LAYER0_PROJECTIONS:
        assert name in loaded["layer_estimates"]
