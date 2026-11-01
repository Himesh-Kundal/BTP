"""Tests for weight artifact dumping and reload verification (frontend/artifact_dumper.py)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from frontend.artifact_dumper import dump_linear_artifact, load_linear_artifact
from frontend.linear_extractor import ExtractedLinear


def test_dump_and_reload_artifact(tmp_path: Path):
    """Verify bit-identical reload of dumped numpy weights and metadata."""
    # Synthetic float32 weight and bias
    w_torch = torch.randn(32, 64, dtype=torch.float32)
    b_torch = torch.randn(32, dtype=torch.float32)

    extracted = ExtractedLinear(
        name="layer0_q_proj",
        layer_idx=0,
        proj_type="q_proj",
        module_path="model.layers.0.self_attn.q_proj",
        in_features=64,
        out_features=32,
        has_bias=True,
        dtype="float32",
        weight=w_torch,
        bias=b_torch,
    )

    art_dir = dump_linear_artifact(
        extracted=extracted,
        base_dir=tmp_path,
        model_id="test_model",
        revision="test_rev12345",
    )

    assert art_dir.exists()
    assert (art_dir / "weight.npy").exists()
    assert (art_dir / "bias.npy").exists()
    assert (art_dir / "meta.json").exists()

    w_loaded, b_loaded, meta = load_linear_artifact(art_dir)

    # Bit-for-bit check
    np.testing.assert_array_equal(w_loaded, w_torch.numpy())
    assert b_loaded is not None
    np.testing.assert_array_equal(b_loaded, b_torch.numpy())

    # Metadata check
    assert meta["shape"] == [32, 64]
    assert meta["in_features"] == 64
    assert meta["out_features"] == 32
    assert meta["has_bias"] is True
    assert "weight_sha256" in meta
    assert len(meta["weight_sha256"]) == 64


def test_artifact_dump_on_smollm_layer0(tmp_path: Path):
    """Dump layer 0 of real SmolLM2-135M and verify reload."""
    from frontend.artifact_dumper import dump_all_artifacts
    from frontend.model_loader import load_model_and_tokenizer

    try:
        model, _ = load_model_and_tokenizer()
    except (OSError, RuntimeError) as e:
        pytest.skip(f"Model unavailable: {e}")

    # Dump only layer 0
    dumped_dirs = dump_all_artifacts(model, layers=[0], base_dir=tmp_path)
    assert len(dumped_dirs) == 7

    for d in dumped_dirs:
        w_loaded, b_loaded, meta = load_linear_artifact(d)
        assert w_loaded.dtype == np.float32
        assert b_loaded is None
        assert meta["layer_idx"] == 0
        assert meta["proj_type"] in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
