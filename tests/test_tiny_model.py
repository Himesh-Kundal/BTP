"""Verification tests for Tiny Random Model generator (Task U0.5)."""

from pathlib import Path

import numpy as np

from frontend.tiny_model import (
    dump_tiny_model_artifacts,
    generate_tiny_model,
    tiny_model_forward_fp32,
)


def test_tiny_model_determinism():
    """Verify that model weights are 100% bit-identical across runs with identical seed."""
    m1 = generate_tiny_model(seed=42)
    m2 = generate_tiny_model(seed=42)

    for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=True):
        np.testing.assert_array_equal(p1.detach().cpu().numpy(), p2.detach().cpu().numpy())


def test_tiny_model_architectural_contracts():
    """Verify all dimensions match Task U0.5 specifications."""
    model = generate_tiny_model(seed=42)
    cfg = model.config

    assert cfg.hidden_size == 64
    assert cfg.num_hidden_layers == 2
    assert cfg.num_attention_heads == 4
    assert cfg.num_key_value_heads == 2
    assert cfg.intermediate_size == 128
    assert cfg.vocab_size == 256
    assert cfg.max_position_embeddings == 32
    assert cfg.tie_word_embeddings is True
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()

    # Check 7 projections per layer
    for layer in model.model.layers:
        assert layer.self_attn.q_proj.weight.shape == (64, 64)
        assert layer.self_attn.k_proj.weight.shape == (32, 64)
        assert layer.self_attn.v_proj.weight.shape == (32, 64)
        assert layer.self_attn.o_proj.weight.shape == (64, 64)
        assert layer.mlp.gate_proj.weight.shape == (128, 64)
        assert layer.mlp.up_proj.weight.shape == (128, 64)
        assert layer.mlp.down_proj.weight.shape == (64, 128)
        assert layer.self_attn.q_proj.bias is None
        assert layer.mlp.gate_proj.bias is None


def test_tiny_model_artifact_dumping(tmp_path: Path):
    """Verify that artifacts dump cleanly with correct files and manifest."""
    manifest = dump_tiny_model_artifacts(target_dir=tmp_path, seed=42)

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "embed_tokens.npy").exists()
    assert (tmp_path / "lm_head.npy").exists()
    assert (tmp_path / "final_norm.npy").exists()

    for l_idx in range(2):
        layer_dir = tmp_path / f"layer_{l_idx}"
        assert (layer_dir / "input_layernorm.npy").exists()
        assert (layer_dir / "post_attention_layernorm.npy").exists()
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            assert (layer_dir / proj / "weight.npy").exists()

    # Check manifest metadata
    assert manifest["config"]["vocab_size"] == 256
    assert manifest["config"]["hidden_size"] == 64


def test_tiny_model_forward_fp32():
    """Verify that PyTorch reference forward pass produces deterministic logits and argmax."""
    prompt = [5, 42, 100, 200]
    out1 = tiny_model_forward_fp32(prompt, seed=42)
    out2 = tiny_model_forward_fp32(prompt, seed=42)

    logits = out1["logits"]
    assert logits.shape == (4, 256)
    assert 0 <= out1["next_token_id"] < 256
    np.testing.assert_array_equal(out1["logits"], out2["logits"])
    assert out1["next_token_id"] == out2["next_token_id"]
