"""Mathematical validation test: X @ W.T matches PyTorch module output within 1e-5."""

from pathlib import Path

import pytest
import torch

from frontend.activation_capture import load_captured_activation
from frontend.artifact_dumper import load_linear_artifact
from frontend.linear_extractor import PROJECTION_TYPES
from frontend.model_loader import load_model_and_tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def loaded_model():
    try:
        model, _ = load_model_and_tokenizer()
        return model
    except (OSError, RuntimeError) as e:
        pytest.skip(f"Model unavailable: {e}")


@pytest.mark.parametrize("layer_idx", [0, 14, 29])
@pytest.mark.parametrize("proj_type", PROJECTION_TYPES)
def test_linear_module_matches_matmul(loaded_model, layer_idx: int, proj_type: str):
    """Verify that module(x) == x @ W.T within 1e-5 float tolerance across layers and projections."""
    # Find module inside PyTorch model
    decoder_layer = loaded_model.model.layers[layer_idx]
    if proj_type in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        module = getattr(decoder_layer.self_attn, proj_type)
    else:
        module = getattr(decoder_layer.mlp, proj_type)

    # Locate dumped artifacts
    artifact_root = REPO_ROOT / "artifacts" / "SmolLM2-135M@93efa2f097d5" / f"layer_{layer_idx}" / proj_type
    if not artifact_root.exists():
        pytest.skip(f"Artifact directory {artifact_root} not found. Run artifact dump first.")

    w_np, b_np, _ = load_linear_artifact(artifact_root)
    x_np, _ = load_captured_activation(artifact_root / "calib")

    # Convert to torch float32 tensors
    x_t = torch.from_numpy(x_np)
    w_t = torch.from_numpy(w_np)
    b_t = torch.from_numpy(b_np) if b_np is not None else None

    # Compute reference matmul: X @ W.T + b
    y_manual = torch.matmul(x_t, w_t.T)
    if b_t is not None:
        y_manual = y_manual + b_t

    # Compute actual PyTorch module output
    with torch.no_grad():
        y_module = module(x_t)

    # Max absolute difference
    max_abs_diff = torch.max(torch.abs(y_manual - y_module)).item()
    assert max_abs_diff <= 1e-5, (
        f"Layer {layer_idx} {proj_type}: max abs diff {max_abs_diff} exceeds 1e-5"
    )
