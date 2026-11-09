"""Deterministic Tiny Random Transformer Model Generator for BTP Part II (Task U0.5).

Generates a minimal, HuggingFace-compatible Llama-style autoregressive transformer
with the identical operator graph as SmolLM2-135M, but scaled down for ultra-fast
pipeline verification and testing:
- d = 64 (hidden dimension)
- L = 2 (transformer layers)
- n_q = 4 (query attention heads)
- n_kv = 2 (key/value heads, GQA 2:1)
- head_dim = 16
- d_ff = 128 (SwiGLU intermediate dimension)
- vocab_size = 256
- max_position_embeddings = 32
- tied word embeddings: embed_tokens == lm_head (256 x 64)
- bias = False
- rms_norm_eps = 1e-5
- hidden_act = silu
- rope_theta = 10000.0
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_model@random"


def get_tiny_model_config() -> LlamaConfig:
    """Return the pinned LlamaConfig for the tiny random model."""
    return LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        tie_word_embeddings=True,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
    )


def generate_tiny_model(seed: int = 42) -> LlamaForCausalLM:
    """Deterministically instantiate a tiny LlamaForCausalLM model with fixed random weights."""
    torch.manual_seed(seed)
    config = get_tiny_model_config()
    model = LlamaForCausalLM(config)
    model.eval()

    # Re-initialize weights with normal distribution scaled for stable variance
    with torch.no_grad():
        for param in model.parameters():
            torch.nn.init.normal_(param, mean=0.0, std=0.04)

        # Explicitly enforce tied embedding pointer
        model.lm_head.weight = model.model.embed_tokens.weight

        # RMSNorm weights initialized around 1.0
        for layer in model.model.layers:
            torch.nn.init.ones_(layer.input_layernorm.weight)
            torch.nn.init.ones_(layer.post_attention_layernorm.weight)
        torch.nn.init.ones_(model.model.norm.weight)

    return model


def dump_tiny_model_artifacts(
    target_dir: Path = TINY_ARTIFACTS_DIR, seed: int = 42
) -> dict:
    """Generate and dump all tiny model weight arrays, config, and metadata to disk."""
    target_dir.mkdir(parents=True, exist_ok=True)
    model = generate_tiny_model(seed=seed)
    config = model.config

    manifest = {
        "model_slug": "tiny_model",
        "revision": "random",
        "seed": seed,
        "config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.hidden_size // config.num_attention_heads,
            "max_position_embeddings": config.max_position_embeddings,
            "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": getattr(config, "rope_theta", 10000.0),
            "tie_word_embeddings": config.tie_word_embeddings,
        },
        "modules": {},
    }

    # Save embed_tokens and lm_head (tied)
    embed_w = model.model.embed_tokens.weight.detach().cpu().numpy()
    np.save(target_dir / "embed_tokens.npy", embed_w)
    np.save(target_dir / "lm_head.npy", embed_w)
    manifest["modules"]["embed_tokens"] = {
        "shape": list(embed_w.shape),
        "path": "embed_tokens.npy",
    }
    manifest["modules"]["lm_head"] = {
        "shape": list(embed_w.shape),
        "path": "lm_head.npy",
        "tied_to": "embed_tokens",
    }

    # Save final layernorm
    norm_w = model.model.norm.weight.detach().cpu().numpy()
    np.save(target_dir / "final_norm.npy", norm_w)
    manifest["modules"]["final_norm"] = {
        "shape": list(norm_w.shape),
        "path": "final_norm.npy",
    }

    # Save each layer
    for l_idx, layer in enumerate(model.model.layers):
        layer_dir = target_dir / f"layer_{l_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        # Projections
        projs = {
            "q_proj": layer.self_attn.q_proj.weight.detach().cpu().numpy(),
            "k_proj": layer.self_attn.k_proj.weight.detach().cpu().numpy(),
            "v_proj": layer.self_attn.v_proj.weight.detach().cpu().numpy(),
            "o_proj": layer.self_attn.o_proj.weight.detach().cpu().numpy(),
            "gate_proj": layer.mlp.gate_proj.weight.detach().cpu().numpy(),
            "up_proj": layer.mlp.up_proj.weight.detach().cpu().numpy(),
            "down_proj": layer.mlp.down_proj.weight.detach().cpu().numpy(),
        }
        for name, arr in projs.items():
            proj_dir = layer_dir / name
            proj_dir.mkdir(parents=True, exist_ok=True)
            np.save(proj_dir / "weight.npy", arr)
            manifest["modules"][f"layer_{l_idx}.{name}"] = {
                "shape": list(arr.shape),
                "path": f"layer_{l_idx}/{name}/weight.npy",
            }

        # Norms
        in_norm = layer.input_layernorm.weight.detach().cpu().numpy()
        np.save(layer_dir / "input_layernorm.npy", in_norm)
        post_norm = layer.post_attention_layernorm.weight.detach().cpu().numpy()
        np.save(layer_dir / "post_attention_layernorm.npy", post_norm)

        manifest["modules"][f"layer_{l_idx}.input_layernorm"] = {
            "shape": list(in_norm.shape),
            "path": f"layer_{l_idx}/input_layernorm.npy",
        }
        manifest["modules"][f"layer_{l_idx}.post_attention_layernorm"] = {
            "shape": list(post_norm.shape),
            "path": f"layer_{l_idx}/post_attention_layernorm.npy",
        }

    with open(target_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def tiny_model_forward_fp32(
    input_ids: list[int], seed: int = 42
) -> dict[str, np.ndarray]:
    """Run fp32 forward pass using PyTorch reference to produce baseline test outputs."""
    model = generate_tiny_model(seed=seed)
    with torch.no_grad():
        t_in = torch.tensor([input_ids], dtype=torch.long)
        out = model(t_in, output_hidden_states=True)
        logits = out.logits[0].cpu().numpy()  # [seq_len, vocab_size]
        argmax_tokens = np.argmax(logits, axis=-1).tolist()

    return {
        "logits": logits,
        "next_token_id": int(argmax_tokens[-1]),
        "all_argmax_ids": argmax_tokens,
    }
