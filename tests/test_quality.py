"""Tests for Claim C2 quantization quality metrics and frozen ADR-0001 thresholds (T3.6)."""

import json
from pathlib import Path

import numpy as np
import pytest

from compiler.quality import (
    compute_cosine_similarity,
    compute_max_abs_error,
    compute_relative_l2,
    compute_sqnr_db,
    evaluate_layer_c2,
)

# Frozen thresholds per ADR-0001
C2_MIN_COSINE_SIM: float = 0.980
C2_MIN_SQNR_DB: float = 15.0
C2_MAX_REL_L2: float = 0.180


def test_quality_metrics_identity():
    """Verify metrics for identical inputs (zero distortion)."""
    y = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float64)

    assert compute_max_abs_error(y, y) == 0.0
    assert compute_relative_l2(y, y) == 0.0
    assert np.isclose(compute_cosine_similarity(y, y), 1.0)
    assert compute_sqnr_db(y, y) == 100.0


def test_quality_metrics_synthetic_noise():
    """Verify metrics on synthetic signals with known analytic SNR."""
    # Signal with mean square = 100
    y_true = np.array([10.0, 10.0, -10.0, -10.0], dtype=np.float64)
    # Noise with error = 1.0, mean square error = 1.0
    y_pred = np.array([11.0, 9.0, -9.0, -11.0], dtype=np.float64)

    assert compute_max_abs_error(y_true, y_pred) == 1.0
    # rel_l2 = sqrt(4) / sqrt(400) = 2 / 20 = 0.1
    assert np.isclose(compute_relative_l2(y_true, y_pred), 0.1)
    # SQNR = 10 * log10(100 / 1) = 20 dB
    assert np.isclose(compute_sqnr_db(y_true, y_pred), 20.0)
    # Cosine similarity: dot = (10*11 + 10*9 + 10*9 + 10*11) = 400 - 4 = 396? Wait:
    # 10*11 + 10*9 + (-10)*(-9) + (-10)*(-11) = 110 + 90 + 90 + 110 = 400.
    # norm_true = sqrt(400) = 20, norm_pred = sqrt(121 + 81 + 81 + 121) = sqrt(404)
    # cos_sim = 400 / (20 * sqrt(404)) = 20 / sqrt(404) = 0.995037
    assert compute_cosine_similarity(y_true, y_pred) > 0.99


def test_frozen_c2_thresholds_on_all_configurations():
    """Verify all 21 configurations in results/quant_quality.json satisfy ADR-0001 thresholds."""
    report_path = Path("results") / "quant_quality.json"
    if not report_path.is_file():
        pytest.skip("results/quant_quality.json not found; run eval_quant_quality.py first")

    with open(report_path) as f:
        data = json.load(f)

    configs = data["configurations"]
    assert len(configs) == 21, f"Expected 21 configurations, found {len(configs)}"

    for cfg_name, metrics in configs.items():
        cos_sim = metrics["cosine_similarity"]
        sqnr = metrics["sqnr_db"]
        rel_l2 = metrics["relative_l2"]

        assert cos_sim >= C2_MIN_COSINE_SIM, (
            f"{cfg_name} failed cosine similarity threshold: {cos_sim:.6f} < {C2_MIN_COSINE_SIM}"
        )
        assert sqnr >= C2_MIN_SQNR_DB, (
            f"{cfg_name} failed SQNR threshold: {sqnr:.2f} dB < {C2_MIN_SQNR_DB} dB"
        )
        assert rel_l2 <= C2_MAX_REL_L2, (
            f"{cfg_name} failed relative L2 error threshold: {rel_l2:.4e} > {C2_MAX_REL_L2}"
        )


def test_evaluate_layer_c2_live_layer0_q_proj():
    """Direct live execution test of evaluate_layer_c2 on layer 0 q_proj."""
    model_dir = Path("artifacts") / "SmolLM2-135M@93efa2f097d5" / "layer_0" / "q_proj"
    w_path = model_dir / "weight.npy"
    x_path = model_dir / "calib" / "x_sample.npy"

    if not w_path.exists() or not x_path.exists():
        pytest.skip("SmolLM2 layer 0 artifacts not available")

    w_float = np.load(w_path)
    x_float = np.load(x_path)

    metrics = evaluate_layer_c2(w_float, x_float, calib_method="absmax")

    assert metrics["cosine_similarity"] >= C2_MIN_COSINE_SIM
    assert metrics["sqnr_db"] >= C2_MIN_SQNR_DB
    assert metrics["relative_l2"] <= C2_MAX_REL_L2
    assert metrics["sqnr_db"] > 30.0  # Layer 0 q_proj typically achieves ~34.8 dB
