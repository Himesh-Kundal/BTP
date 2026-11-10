"""Frontend Importer v2 for SmolLM2 and Tiny Model Architectures.

Extracts complete model artifacts (embedding, all layer norms, 7 linear projections
per layer, tied LM head, RoPE parameters, RMSNorm eps) with SHA256 checksums,
and constructs Graph IR v2 instances.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ir.graph_v2 import EdgeFormat, GraphV2, OpKind, TensorKind


def compute_file_sha256(path: Path) -> str:
    """Compute standard SHA256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_and_dump_model_v2(
    model: Any,
    config: Any,
    target_dir: Path,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    """Extract and dump all weights, norms, and config metadata to target_dir with SHA256 hashes."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "config": {
            "vocab_size": int(config.vocab_size),
            "hidden_size": int(config.hidden_size),
            "intermediate_size": int(config.intermediate_size),
            "num_hidden_layers": int(config.num_hidden_layers),
            "num_attention_heads": int(config.num_attention_heads),
            "num_key_value_heads": int(config.num_key_value_heads),
            "head_dim": int(config.hidden_size // config.num_attention_heads),
            "max_position_embeddings": int(config.max_position_embeddings),
            "rms_norm_eps": float(config.rms_norm_eps),
            "rope_theta": float(getattr(config, "rope_theta", 100000.0)),
            "tie_word_embeddings": bool(getattr(config, "tie_word_embeddings", True)),
        },
        "tensors": {},
    }

    def _save_tensor(name: str, tensor: torch.Tensor, rel_path: str, tied_to: str | None = None) -> None:
        file_path = target_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        arr = tensor.detach().float().cpu().numpy()
        np.save(file_path, arr)
        sha = compute_file_sha256(file_path)
        manifest["tensors"][name] = {
            "path": rel_path,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha256": sha,
            "tied_to": tied_to,
        }

    # 1. Embeddings & Tied Head
    _save_tensor("model.embed_tokens.weight", model.model.embed_tokens.weight, "embed_tokens.npy")
    if getattr(config, "tie_word_embeddings", True):
        # Tied: reference embed_tokens
        file_path = target_dir / "lm_head.npy"
        arr = model.model.embed_tokens.weight.detach().float().cpu().numpy()
        np.save(file_path, arr)
        sha = compute_file_sha256(file_path)
        manifest["tensors"]["lm_head.weight"] = {
            "path": "lm_head.npy",
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha256": sha,
            "tied_to": "model.embed_tokens.weight",
        }
    else:
        _save_tensor("lm_head.weight", model.lm_head.weight, "lm_head.npy")

    # 2. Final Norm
    _save_tensor("model.norm.weight", model.model.norm.weight, "final_norm.npy")

    # 3. Layers
    for l_idx, layer in enumerate(model.model.layers):
        layer_prefix = f"model.layers.{l_idx}"
        l_dir = f"layer_{l_idx}"

        # Norms
        _save_tensor(
            f"{layer_prefix}.input_layernorm.weight",
            layer.input_layernorm.weight,
            f"{l_dir}/input_layernorm.npy",
        )
        _save_tensor(
            f"{layer_prefix}.post_attention_layernorm.weight",
            layer.post_attention_layernorm.weight,
            f"{l_dir}/post_attention_layernorm.npy",
        )

        # 7 Projections
        projs = {
            "self_attn.q_proj.weight": layer.self_attn.q_proj.weight,
            "self_attn.k_proj.weight": layer.self_attn.k_proj.weight,
            "self_attn.v_proj.weight": layer.self_attn.v_proj.weight,
            "self_attn.o_proj.weight": layer.self_attn.o_proj.weight,
            "mlp.gate_proj.weight": layer.mlp.gate_proj.weight,
            "mlp.up_proj.weight": layer.mlp.up_proj.weight,
            "mlp.down_proj.weight": layer.mlp.down_proj.weight,
        }
        for p_name, p_tensor in projs.items():
            proj_slug = p_name.split(".")[-2]  # e.g. q_proj
            _save_tensor(
                f"{layer_prefix}.{p_name}",
                p_tensor,
                f"{l_dir}/{proj_slug}.npy",
            )

    manifest_file = target_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return manifest


def load_model_artifacts_v2(target_dir: Path) -> dict[str, Any]:
    """Reload and cryptographically verify all artifacts from manifest.json."""
    target_dir = Path(target_dir)
    manifest_file = target_dir / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    loaded_tensors: dict[str, np.ndarray] = {}
    for name, info in manifest["tensors"].items():
        file_path = target_dir / info["path"]
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact file missing: {file_path}")
        computed_sha = compute_file_sha256(file_path)
        if computed_sha != info["sha256"]:
            raise ValueError(
                f"SHA256 mismatch for {name}: expected {info['sha256']}, got {computed_sha}"
            )
        loaded_tensors[name] = np.load(file_path)

    return {
        "manifest": manifest,
        "config": manifest["config"],
        "tensors": loaded_tensors,
    }


def build_graph_v2_from_artifacts(target_dir: Path) -> GraphV2:
    """Build a complete GraphV2 instance representing the decoder step from artifacts."""
    artifacts = load_model_artifacts_v2(target_dir)
    config = artifacts["config"]
    manifest = artifacts["manifest"]

    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    vocab_size = config["vocab_size"]
    num_layers = config["num_hidden_layers"]
    num_q_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    max_ctx = config["max_position_embeddings"]

    g = GraphV2(
        name=manifest.get("model_id", "transformer_decoder"),
        metadata={"config": config},
    )

    # Edge formats (INT16 for residual, INT8 for weights and normed activations)
    fmt_tok = EdgeFormat(bits=16, signed=False, scale=1.0, fixed_point="UINT16")
    fmt_res = EdgeFormat(bits=16, signed=True, scale=0.01, fixed_point="INT16")
    fmt_act = EdgeFormat(bits=8, signed=True, scale=0.05, fixed_point="INT8")
    fmt_w = EdgeFormat(bits=8, signed=True, scale=0.002, fixed_point="INT8")
    fmt_kv = EdgeFormat(bits=8, signed=True, scale=0.05, fixed_point="INT8")

    # 1. Token Input
    g.add_tensor("tok_in", TensorKind.DYNAMIC, [1], fmt_tok)

    # 2. Embedding
    embed_info = manifest["tensors"]["model.embed_tokens.weight"]
    g.add_tensor(
        "model.embed_tokens.weight",
        TensorKind.CONSTANT,
        [vocab_size, hidden_size],
        fmt_w,
        data_file=embed_info["path"],
        sha256=embed_info["sha256"],
    )

    # h_0
    g.add_tensor("h_0", TensorKind.DYNAMIC, [1, hidden_size], fmt_res)
    g.add_node(
        node_id="embed_tokens",
        op=OpKind.Embed,
        inputs=["tok_in", "model.embed_tokens.weight"],
        outputs=["h_0"],
        provenance={"module": "model.embed_tokens"},
    )

    # 3. Transformer Layers
    h_curr = "h_0"
    for l_idx in range(num_layers):
        l_prefix = f"model.layers.{l_idx}"

        # Attention RMSNorm
        in_norm_info = manifest["tensors"][f"{l_prefix}.input_layernorm.weight"]
        g.add_tensor(
            f"{l_prefix}.input_layernorm.weight",
            TensorKind.CONSTANT,
            [hidden_size],
            fmt_w,
            data_file=in_norm_info["path"],
            sha256=in_norm_info["sha256"],
        )
        h_attn_norm = f"h_attn_norm_{l_idx}"
        g.add_tensor(h_attn_norm, TensorKind.DYNAMIC, [1, hidden_size], fmt_act)
        g.add_node(
            node_id=f"attn_norm_{l_idx}",
            op=OpKind.RMSNorm,
            inputs=[h_curr, f"{l_prefix}.input_layernorm.weight"],
            outputs=[h_attn_norm],
            attributes={"eps": config["rms_norm_eps"]},
            provenance={"module": f"{l_prefix}.input_layernorm"},
        )

        # Q, K, V projections
        for proj, out_dim in [("q_proj", num_q_heads * head_dim), ("k_proj", num_kv_heads * head_dim), ("v_proj", num_kv_heads * head_dim)]:
            info = manifest["tensors"][f"{l_prefix}.self_attn.{proj}.weight"]
            g.add_tensor(
                f"{l_prefix}.self_attn.{proj}.weight",
                TensorKind.CONSTANT,
                [out_dim, hidden_size],
                fmt_w,
                data_file=info["path"],
                sha256=info["sha256"],
            )
            g.add_tensor(f"{proj}_{l_idx}", TensorKind.DYNAMIC, [1, out_dim], fmt_act)
            g.add_node(
                node_id=f"{proj}_{l_idx}",
                op=OpKind.Linear,
                inputs=[h_attn_norm, f"{l_prefix}.self_attn.{proj}.weight"],
                outputs=[f"{proj}_{l_idx}"],
                provenance={"module": f"{l_prefix}.self_attn.{proj}"},
            )

        # RoPE
        g.add_tensor(f"q_rope_{l_idx}", TensorKind.DYNAMIC, [1, num_q_heads * head_dim], fmt_act)
        g.add_tensor(f"k_rope_{l_idx}", TensorKind.DYNAMIC, [1, num_kv_heads * head_dim], fmt_act)
        g.add_node(
            node_id=f"q_rope_{l_idx}",
            op=OpKind.RoPE,
            inputs=[f"q_proj_{l_idx}"],
            outputs=[f"q_rope_{l_idx}"],
            attributes={"head_dim": head_dim, "theta": config["rope_theta"]},
            provenance={"module": f"{l_prefix}.self_attn.rotary_emb"},
        )
        g.add_node(
            node_id=f"k_rope_{l_idx}",
            op=OpKind.RoPE,
            inputs=[f"k_proj_{l_idx}"],
            outputs=[f"k_rope_{l_idx}"],
            attributes={"head_dim": head_dim, "theta": config["rope_theta"]},
            provenance={"module": f"{l_prefix}.self_attn.rotary_emb"},
        )

        # KV cache state and Attention
        g.add_tensor(
            f"kv_cache_{l_idx}",
            TensorKind.STATE,
            [2, num_kv_heads, max_ctx, head_dim],
            fmt_kv,
        )
        g.add_tensor(f"attn_context_{l_idx}", TensorKind.DYNAMIC, [1, hidden_size], fmt_act)
        g.add_node(
            node_id=f"attention_{l_idx}",
            op=OpKind.Attention,
            inputs=[f"q_rope_{l_idx}", f"k_rope_{l_idx}", f"v_proj_{l_idx}", f"kv_cache_{l_idx}"],
            outputs=[f"attn_context_{l_idx}"],
            attributes={
                "num_q_heads": num_q_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "max_ctx": max_ctx,
            },
            provenance={"module": f"{l_prefix}.self_attn"},
        )

        # O proj
        o_info = manifest["tensors"][f"{l_prefix}.self_attn.o_proj.weight"]
        g.add_tensor(
            f"{l_prefix}.self_attn.o_proj.weight",
            TensorKind.CONSTANT,
            [hidden_size, hidden_size],
            fmt_w,
            data_file=o_info["path"],
            sha256=o_info["sha256"],
        )
        g.add_tensor(f"o_proj_{l_idx}", TensorKind.DYNAMIC, [1, hidden_size], fmt_res)
        g.add_node(
            node_id=f"o_proj_{l_idx}",
            op=OpKind.Linear,
            inputs=[f"attn_context_{l_idx}", f"{l_prefix}.self_attn.o_proj.weight"],
            outputs=[f"o_proj_{l_idx}"],
            provenance={"module": f"{l_prefix}.self_attn.o_proj"},
        )

        # Residual add 1
        h_mid = f"h_mid_{l_idx}"
        g.add_tensor(h_mid, TensorKind.DYNAMIC, [1, hidden_size], fmt_res)
        g.add_node(
            node_id=f"res_add_attn_{l_idx}",
            op=OpKind.Add,
            inputs=[h_curr, f"o_proj_{l_idx}"],
            outputs=[h_mid],
            provenance={"module": f"{l_prefix}.residual_1"},
        )

        # Post-attn layernorm
        post_norm_info = manifest["tensors"][f"{l_prefix}.post_attention_layernorm.weight"]
        g.add_tensor(
            f"{l_prefix}.post_attention_layernorm.weight",
            TensorKind.CONSTANT,
            [hidden_size],
            fmt_w,
            data_file=post_norm_info["path"],
            sha256=post_norm_info["sha256"],
        )
        h_mlp_norm = f"h_mlp_norm_{l_idx}"
        g.add_tensor(h_mlp_norm, TensorKind.DYNAMIC, [1, hidden_size], fmt_act)
        g.add_node(
            node_id=f"mlp_norm_{l_idx}",
            op=OpKind.RMSNorm,
            inputs=[h_mid, f"{l_prefix}.post_attention_layernorm.weight"],
            outputs=[h_mlp_norm],
            attributes={"eps": config["rms_norm_eps"]},
            provenance={"module": f"{l_prefix}.post_attention_layernorm"},
        )

        # Gate and Up projections
        gate_info = manifest["tensors"][f"{l_prefix}.mlp.gate_proj.weight"]
        up_info = manifest["tensors"][f"{l_prefix}.mlp.up_proj.weight"]
        g.add_tensor(
            f"{l_prefix}.mlp.gate_proj.weight",
            TensorKind.CONSTANT,
            [intermediate_size, hidden_size],
            fmt_w,
            data_file=gate_info["path"],
            sha256=gate_info["sha256"],
        )
        g.add_tensor(
            f"{l_prefix}.mlp.up_proj.weight",
            TensorKind.CONSTANT,
            [intermediate_size, hidden_size],
            fmt_w,
            data_file=up_info["path"],
            sha256=up_info["sha256"],
        )
        g.add_tensor(f"gate_{l_idx}", TensorKind.DYNAMIC, [1, intermediate_size], fmt_act)
        g.add_tensor(f"up_{l_idx}", TensorKind.DYNAMIC, [1, intermediate_size], fmt_act)
        g.add_node(
            node_id=f"gate_proj_{l_idx}",
            op=OpKind.Linear,
            inputs=[h_mlp_norm, f"{l_prefix}.mlp.gate_proj.weight"],
            outputs=[f"gate_{l_idx}"],
            provenance={"module": f"{l_prefix}.mlp.gate_proj"},
        )
        g.add_node(
            node_id=f"up_proj_{l_idx}",
            op=OpKind.Linear,
            inputs=[h_mlp_norm, f"{l_prefix}.mlp.up_proj.weight"],
            outputs=[f"up_{l_idx}"],
            provenance={"module": f"{l_prefix}.mlp.up_proj"},
        )

        # SiLU and Elementwise Mul (SwiGLU)
        g.add_tensor(f"silu_{l_idx}", TensorKind.DYNAMIC, [1, intermediate_size], fmt_act)
        g.add_node(
            node_id=f"silu_{l_idx}",
            op=OpKind.SiLU,
            inputs=[f"gate_{l_idx}"],
            outputs=[f"silu_{l_idx}"],
            provenance={"module": f"{l_prefix}.mlp.act_fn"},
        )
        g.add_tensor(f"swiglu_{l_idx}", TensorKind.DYNAMIC, [1, intermediate_size], fmt_act)
        g.add_node(
            node_id=f"swiglu_{l_idx}",
            op=OpKind.EltMul,
            inputs=[f"silu_{l_idx}", f"up_{l_idx}"],
            outputs=[f"swiglu_{l_idx}"],
            provenance={"module": f"{l_prefix}.mlp.mul"},
        )

        # Down proj
        down_info = manifest["tensors"][f"{l_prefix}.mlp.down_proj.weight"]
        g.add_tensor(
            f"{l_prefix}.mlp.down_proj.weight",
            TensorKind.CONSTANT,
            [hidden_size, intermediate_size],
            fmt_w,
            data_file=down_info["path"],
            sha256=down_info["sha256"],
        )
        g.add_tensor(f"down_{l_idx}", TensorKind.DYNAMIC, [1, hidden_size], fmt_res)
        g.add_node(
            node_id=f"down_proj_{l_idx}",
            op=OpKind.Linear,
            inputs=[f"swiglu_{l_idx}", f"{l_prefix}.mlp.down_proj.weight"],
            outputs=[f"down_{l_idx}"],
            provenance={"module": f"{l_prefix}.mlp.down_proj"},
        )

        # Residual add 2
        h_next = f"h_{l_idx + 1}"
        g.add_tensor(h_next, TensorKind.DYNAMIC, [1, hidden_size], fmt_res)
        g.add_node(
            node_id=f"res_add_mlp_{l_idx}",
            op=OpKind.Add,
            inputs=[h_mid, f"down_{l_idx}"],
            outputs=[h_next],
            provenance={"module": f"{l_prefix}.residual_2"},
        )

        h_curr = h_next

    # 4. Final Norm
    final_norm_info = manifest["tensors"]["model.norm.weight"]
    g.add_tensor(
        "model.norm.weight",
        TensorKind.CONSTANT,
        [hidden_size],
        fmt_w,
        data_file=final_norm_info["path"],
        sha256=final_norm_info["sha256"],
    )
    g.add_tensor("h_final_norm", TensorKind.DYNAMIC, [1, hidden_size], fmt_act)
    g.add_node(
        node_id="final_norm",
        op=OpKind.RMSNorm,
        inputs=[h_curr, "model.norm.weight"],
        outputs=["h_final_norm"],
        attributes={"eps": config["rms_norm_eps"]},
        provenance={"module": "model.norm"},
    )

    # 5. LM Head
    lm_head_info = manifest["tensors"]["lm_head.weight"]
    g.add_tensor(
        "lm_head.weight",
        TensorKind.CONSTANT,
        [vocab_size, hidden_size],
        fmt_w,
        data_file=lm_head_info["path"],
        sha256=lm_head_info["sha256"],
    )
    g.add_tensor("logits", TensorKind.DYNAMIC, [1, vocab_size], fmt_res)
    g.add_node(
        node_id="lm_head",
        op=OpKind.Linear,
        inputs=["h_final_norm", "lm_head.weight"],
        outputs=["logits"],
        provenance={"module": "lm_head"},
    )

    # 6. ArgMax
    g.add_tensor("tok_out", TensorKind.DYNAMIC, [1], fmt_tok)
    g.add_node(
        node_id="argmax",
        op=OpKind.ArgMax,
        inputs=["logits"],
        outputs=["tok_out"],
        provenance={"module": "argmax"},
    )

    return g
