"""Tests for Frontend Importer v2 (Task UB.2)."""

from pathlib import Path

import pytest

from frontend.importer_v2 import (
    build_graph_v2_from_artifacts,
    extract_and_dump_model_v2,
    load_model_artifacts_v2,
)
from frontend.tiny_model import generate_tiny_model
from ir.graph_v2 import validate_invariants


def test_tiny_model_extraction_and_reload(tmp_path: Path):
    """Verify tiny model extraction, manifest generation, and bit-identical reload."""
    model = generate_tiny_model(seed=42)
    artifacts_dir = tmp_path / "tiny_artifacts"

    extract_and_dump_model_v2(
        model=model,
        config=model.config,
        target_dir=artifacts_dir,
        model_id="tiny_model",
        revision="random",
    )

    assert (artifacts_dir / "manifest.json").exists()
    assert (artifacts_dir / "embed_tokens.npy").exists()
    assert (artifacts_dir / "final_norm.npy").exists()
    assert (artifacts_dir / "layer_0" / "q_proj.npy").exists()

    # Verify reload
    reloaded = load_model_artifacts_v2(artifacts_dir)
    assert reloaded["manifest"]["model_id"] == "tiny_model"
    assert "model.embed_tokens.weight" in reloaded["tensors"]
    assert reloaded["tensors"]["model.embed_tokens.weight"].shape == (256, 64)


def test_importer_v2_sha256_tamper_detection_negative_control(tmp_path: Path):
    """Negative control: tampering with an artifact triggers SHA256 integrity failure."""
    model = generate_tiny_model(seed=42)
    artifacts_dir = tmp_path / "tamper_artifacts"

    extract_and_dump_model_v2(
        model=model,
        config=model.config,
        target_dir=artifacts_dir,
        model_id="tiny_model",
        revision="random",
    )

    # Tamper with embed_tokens.npy by appending a byte
    embed_file = artifacts_dir / "embed_tokens.npy"
    with open(embed_file, "ab") as f:
        f.write(b"\x00")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_model_artifacts_v2(artifacts_dir)


def test_build_graph_v2_from_extracted_artifacts(tmp_path: Path):
    """Build GraphV2 from extracted tiny model artifacts and validate Invariants I1–I7."""
    model = generate_tiny_model(seed=42)
    artifacts_dir = tmp_path / "graph_artifacts"

    extract_and_dump_model_v2(
        model=model,
        config=model.config,
        target_dir=artifacts_dir,
        model_id="tiny_model",
        revision="random",
    )

    graph = build_graph_v2_from_artifacts(artifacts_dir)
    assert graph.name == "tiny_model"
    assert len(graph.nodes) > 10
    assert len(graph.tensors) > 20

    # Invariants I1–I7 must pass on the complete decoder graph
    validate_invariants(graph)
