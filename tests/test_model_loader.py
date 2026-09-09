"""Tests for model loader (frontend/model_loader.py)."""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_model_loader_pinned_metadata():
    """Verify pinned model constants and metadata loading."""
    from frontend.model_loader import DEFAULT_MODEL_ID, PINNED_REVISION, get_model_metadata

    assert DEFAULT_MODEL_ID == "HuggingFaceTB/SmolLM2-135M"
    assert len(PINNED_REVISION) == 40  # Full git commit SHA

    meta = get_model_metadata(DEFAULT_MODEL_ID, PINNED_REVISION)
    assert meta["model_id"] == DEFAULT_MODEL_ID
    assert meta["revision"] == PINNED_REVISION
    assert "hidden_size" in meta["config"]
    assert meta["config"]["hidden_size"] == 576


def test_load_model_offline_or_online():
    """Load model in float32 on CPU and verify configuration."""
    from frontend.model_loader import load_model_and_tokenizer

    try:
        model, tokenizer = load_model_and_tokenizer()
    except (OSError, RuntimeError) as e:
        pytest.skip(f"Model download unavailable: {e}")

    assert model is not None
    assert tokenizer is not None
    assert not model.training  # Must be in eval mode
    assert sum(p.numel() for p in model.parameters()) > 100_000_000


def test_model_json_creation(tmp_path: Path):
    """Verify results/model.json structure and contents."""
    from frontend.model_loader import record_model_metadata

    out_file = tmp_path / "model.json"
    data = record_model_metadata(out_path=out_file)

    assert out_file.exists()
    assert data["model_id"] == "HuggingFaceTB/SmolLM2-135M"
    assert data["revision"] == "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"

    with open(out_file, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["config"]["num_hidden_layers"] == 30
    assert loaded["config"]["intermediate_size"] == 1536
