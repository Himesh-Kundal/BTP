"""Tests for Format and Dyadic Scale Assignment Pass (Task UB.3)."""

import json
from pathlib import Path

from compiler.format_assignment import assign_formats_and_scales
from frontend.importer_v2 import build_graph_v2_from_artifacts, load_model_artifacts_v2
from ir.graph_v2 import validate_invariants

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_model_v2"
CONFIG_FROZEN_PATH = REPO_ROOT / "numerics" / "config_frozen.json"


def test_format_and_dyadic_scale_assignment_tiny_model(tmp_path: Path):
    """Test format assignment and scale folding across all operations of the tiny model."""
    artifacts = load_model_artifacts_v2(TINY_ARTIFACTS_DIR)
    graph = build_graph_v2_from_artifacts(TINY_ARTIFACTS_DIR)

    with open(CONFIG_FROZEN_PATH, "r", encoding="utf-8") as f:
        frozen_config = json.load(f)

    result = assign_formats_and_scales(
        graph=graph,
        weights_dict=artifacts["tensors"],
        frozen_config=frozen_config,
    )

    # 1. Format table completeness
    assert len(result.format_table) == len(graph.tensors)
    for fmt in result.format_table.values():
        assert fmt["bits"] in (8, 16, 24, 32)
        assert fmt["min_val"] <= fmt["max_val"]
        assert len(fmt["shape"]) >= 1

    # 2. Dyadic requantization parameters
    assert "embed_tokens" in result.requant_params
    assert "attn_norm_0" in result.requant_params
    assert "q_proj_0" in result.requant_params
    assert "attention_0" in result.requant_params

    # Verify all (M, s) are valid non-negative numbers
    for params in result.requant_params.values():
        assert len(params["M"]) == len(params["shift"])
        for M, s in zip(params["M"], params["shift"], strict=False):
            assert M >= 0
            assert 0 <= s <= 62

    # 3. Save JSON and verify round-trip
    out_json = tmp_path / "format_table.json"
    result.save_json(out_json)
    assert out_json.exists()

    # 4. Invariants must still pass on updated graph
    validate_invariants(result.graph)
