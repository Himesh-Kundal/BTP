"""Tests for linear extractor and architecture parser (frontend/linear_extractor.py)."""

import pytest
import torch

from frontend.linear_extractor import PROJECTION_TYPES, extract_linear_modules
from frontend.model_loader import load_model_and_tokenizer


@pytest.fixture(scope="module")
def loaded_model():
    """Load model once for tests."""
    model, _ = load_model_and_tokenizer()
    return model


def test_projection_types_count():
    """Confirm exactly 7 projection types are recognized."""
    assert len(PROJECTION_TYPES) == 7
    expected = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert set(PROJECTION_TYPES) == expected


def test_extract_linear_modules(loaded_model):
    """Verify that all linear modules across all layers are extracted and classified."""
    extracted = extract_linear_modules(loaded_model)

    # 30 layers * 7 projection types = 210 linear modules
    assert len(extracted) == 210

    # Verify layer 0 has all 7 projection types
    l0_modules = [m for m in extracted if m.layer_idx == 0]
    assert len(l0_modules) == 7

    proj_map = {m.proj_type: m for m in l0_modules}
    assert set(proj_map.keys()) == set(PROJECTION_TYPES)

    # Expected shapes:
    # q_proj: N=576, K=576
    assert proj_map["q_proj"].out_features == 576
    assert proj_map["q_proj"].in_features == 576
    assert not proj_map["q_proj"].has_bias

    # k_proj, v_proj: N=192, K=576
    assert proj_map["k_proj"].out_features == 192
    assert proj_map["k_proj"].in_features == 576
    assert proj_map["v_proj"].out_features == 192
    assert proj_map["v_proj"].in_features == 576

    # o_proj: N=576, K=576
    assert proj_map["o_proj"].out_features == 576
    assert proj_map["o_proj"].in_features == 576

    # gate_proj, up_proj: N=1536, K=576
    assert proj_map["gate_proj"].out_features == 1536
    assert proj_map["gate_proj"].in_features == 576
    assert proj_map["up_proj"].out_features == 1536
    assert proj_map["up_proj"].in_features == 576

    # down_proj: N=576, K=1536
    assert proj_map["down_proj"].out_features == 576
    assert proj_map["down_proj"].in_features == 1536


def test_extract_from_synthetic_module():
    """Verify extractor works on arbitrary synthetic PyTorch module."""
    class DummyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(16, 32, bias=True)
            self.down_proj = torch.nn.Linear(32, 16, bias=False)

    block = DummyBlock()
    extracted = extract_linear_modules(block)
    assert len(extracted) == 2
    types = {m.proj_type for m in extracted}
    assert types == {"q_proj", "down_proj"}
    q = next(m for m in extracted if m.proj_type == "q_proj")
    assert q.has_bias
    assert q.bias is not None
