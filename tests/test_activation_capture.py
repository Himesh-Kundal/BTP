"""Tests for forward-hook activation capture and calibration reproducibility (frontend/activation_capture.py)."""

from pathlib import Path

import numpy as np
import pytest

from frontend.activation_capture import capture_activations, load_captured_activation
from frontend.model_loader import load_model_and_tokenizer


@pytest.fixture(scope="module")
def loaded_model_and_tokenizer():
    try:
        model, tokenizer = load_model_and_tokenizer()
        return model, tokenizer
    except (OSError, RuntimeError) as e:
        pytest.skip(f"Model unavailable: {e}")


def test_activation_capture_layer0(loaded_model_and_tokenizer, tmp_path: Path):
    """Verify that forward hooks capture activations for all 7 linear projections in layer 0."""
    model, tokenizer = loaded_model_and_tokenizer
    calib_dirs = capture_activations(
        model=model,
        tokenizer=tokenizer,
        layers=[0],
        base_dir=tmp_path,
    )

    assert len(calib_dirs) == 7

    for d in calib_dirs:
        x_act, calib_meta = load_captured_activation(d)
        assert x_act.ndim == 2
        assert x_act.dtype == np.float32
        assert calib_meta["absmax"] > 0.0
        assert "sha256" in calib_meta
        assert len(calib_meta["sha256"]) == 64


def test_activation_capture_deterministic_rerun(loaded_model_and_tokenizer, tmp_path: Path):
    """Verify byte-identical reproducibility (identical SHA256) on rerun."""
    model, tokenizer = loaded_model_and_tokenizer
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"

    capture_activations(model, tokenizer, layers=[0], base_dir=dir1)
    capture_activations(model, tokenizer, layers=[0], base_dir=dir2)

    # Check q_proj in layer 0 across both runs
    q1_dir = next(dir1.glob("**/layer_0/q_proj/calib"))
    q2_dir = next(dir2.glob("**/layer_0/q_proj/calib"))

    x1, meta1 = load_captured_activation(q1_dir)
    x2, meta2 = load_captured_activation(q2_dir)

    assert meta1["sha256"] == meta2["sha256"]
    np.testing.assert_array_equal(x1, x2)
