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

    calib_file = target_dir / "calib_scales.json"
    calib_scales: dict[str, Any] = {}
    if calib_file.exists():
        with open(calib_file, "r", encoding="utf-8") as f:
            calib_scales = json.load(f)

    return {
        "manifest": manifest,
        "config": manifest["config"],
        "tensors": loaded_tensors,
        "calib_scales": calib_scales,
    }


def calibrate_and_migrate_artifacts_v2(
    model: Any,
    calibration_windows: list[list[int]],
    target_dir: Path,
    s_gate: float = 0.10,
    swig_shift: int = 6,
    score_shift: int = 1,
) -> dict[str, Any]:
    """Run forward calibration, apply compile-time SmoothQuant & outlier migration, and update artifacts."""
    target_dir = Path(target_dir)
    manifest_file = target_dir / "manifest.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cfg = manifest["config"]
    num_q_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])

    ch_max: dict[str, np.ndarray] = {}
    t_max: dict[str, float] = {}

    def _rec_ch(name: str, arr: np.ndarray) -> None:
        a = np.abs(arr)
        m = np.max(a, axis=(0, 1))
        ch_max[name] = np.maximum(ch_max.get(name, np.zeros_like(m)), m)
        t_max[name] = max(t_max.get(name, 1e-6), float(np.max(a)))

    handles = []
    for i, lyr in enumerate(model.model.layers):
        handles.append(
            lyr.input_layernorm.register_forward_hook(
                lambda m, inp, out, i=i: _rec_ch(f"an_{i}", out.detach().float().cpu().numpy())
            )
        )
        handles.append(
            lyr.self_attn.q_proj.register_forward_hook(
                lambda m, inp, out, i=i: _rec_ch(f"q_{i}", out.detach().float().cpu().numpy())
            )
        )
        handles.append(
            lyr.self_attn.k_proj.register_forward_hook(
                lambda m, inp, out, i=i: _rec_ch(f"k_{i}", out.detach().float().cpu().numpy())
            )
        )
        handles.append(
            lyr.self_attn.v_proj.register_forward_hook(
                lambda m, inp, out, i=i: _rec_ch(f"v_{i}", out.detach().float().cpu().numpy())
            )
        )
        handles.append(
            lyr.post_attention_layernorm.register_forward_hook(
                lambda m, inp, out, i=i: _rec_ch(f"mn_{i}", out.detach().float().cpu().numpy())
            )
        )
        handles.append(
            lyr.mlp.gate_proj.register_forward_hook(
                lambda m, inp, out, i=i: _rec_ch(f"gate_{i}", out.detach().float().cpu().numpy())
            )
        )
        handles.append(
            lyr.mlp.up_proj.register_forward_hook(
                lambda m, inp, out, i=i: _rec_ch(f"up_{i}", out.detach().float().cpu().numpy())
            )
        )
        handles.append(
            lyr.mlp.down_proj.register_forward_pre_hook(
                lambda m, inp, i=i: _rec_ch(f"swig_{i}", inp[0].detach().float().cpu().numpy())
            )
        )
    handles.append(
        model.model.norm.register_forward_hook(
            lambda m, inp, out: _rec_ch("fn", out.detach().float().cpu().numpy())
        )
    )

    with torch.no_grad():
        for win in calibration_windows:
            model(torch.tensor([win], dtype=torch.long))
    for h in handles:
        h.remove()

    def _update_npy(tensor_key: str, arr: np.ndarray) -> None:
        info = manifest["tensors"][tensor_key]
        fpath = target_dir / info["path"]
        np.save(fpath, arr.astype(np.float32))
        info["sha256"] = compute_file_sha256(fpath)

    target_sq_sk = 8.0 / ((256 << score_shift) * 1.4426950408889634)
    s_q_base = float(np.sqrt(target_sq_sk))
    s_k_base = float(np.sqrt(target_sq_sk))
    gqa_rep = num_q_heads // num_kv_heads
    half_hd = head_dim // 2

    calib_scales: dict[str, Any] = {"__s_gate__": float(s_gate)}

    for i, lyr in enumerate(model.model.layers):
        l_prefix = f"model.layers.{i}"
        in_g = lyr.input_layernorm.weight.detach().float().cpu().numpy().copy()
        post_g = lyr.post_attention_layernorm.weight.detach().float().cpu().numpy().copy()
        wq = lyr.self_attn.q_proj.weight.detach().float().cpu().numpy().copy()
        wk = lyr.self_attn.k_proj.weight.detach().float().cpu().numpy().copy()
        wv = lyr.self_attn.v_proj.weight.detach().float().cpu().numpy().copy()
        wg = lyr.mlp.gate_proj.weight.detach().float().cpu().numpy().copy()
        wu = lyr.mlp.up_proj.weight.detach().float().cpu().numpy().copy()
        wd = lyr.mlp.down_proj.weight.detach().float().cpu().numpy().copy()

        # 1. SmoothQuant migration on input_layernorm <-> (wq, wk, wv)
        an_cm = np.maximum(ch_max[f"an_{i}"], 1e-5)
        w_qkv_cm = np.maximum(np.max(np.abs(np.vstack([wq, wk, wv])), axis=0), 1e-5)
        d_an = (an_cm / w_qkv_cm) ** 0.5
        d_an = d_an / np.median(d_an)
        in_g_tilde = in_g / d_an
        wq_tilde = wq * d_an[None, :]
        wk_tilde = wk * d_an[None, :]
        wv_tilde = wv * d_an[None, :]
        s_an = float(np.max(an_cm / d_an)) / 127.0

        # 2. SmoothQuant migration on post_attention_layernorm <-> (wg, wu)
        mn_cm = np.maximum(ch_max[f"mn_{i}"], 1e-5)
        w_gu_cm = np.maximum(np.max(np.abs(np.vstack([wg, wu])), axis=0), 1e-5)
        d_mn = (mn_cm / w_gu_cm) ** 0.5
        d_mn = d_mn / np.median(d_mn)
        post_g_tilde = post_g / d_mn
        wg_tilde = wg * d_mn[None, :]
        wu_tilde = wu * d_mn[None, :]
        s_mn = float(np.max(mn_cm / d_mn)) / 127.0

        # 3. Outlier channel migration on (gate, up) <-> down_proj
        g_cm = ch_max[f"gate_{i}"]
        u_cm = ch_max[f"up_{i}"]
        sw_cm = ch_max[f"swig_{i}"]
        beta_g = np.maximum(1.0, g_cm / (126.0 * s_gate))
        u_cap = float(np.percentile(u_cm, 98.5))
        sw_cap = float(np.percentile(sw_cm / beta_g, 98.5))
        alpha_u = np.maximum(
            1.0,
            np.maximum(u_cm / max(u_cap, 1e-6), (sw_cm / beta_g) / max(sw_cap, 1e-6)),
        )
        wd_tilde = wd * (beta_g * alpha_u)[None, :]
        s_up = max(
            float(np.max(u_cm / alpha_u)) / 127.0,
            float(np.max(sw_cm / (beta_g * alpha_u))) / (127.0 * (1 << swig_shift) * s_gate),
        )
        s_swiglu = float(1 << swig_shift) * s_gate * s_up

        # 4. RoPE-invariant per-pair Q <-> K migration
        q_cm = ch_max[f"q_{i}"].reshape(num_kv_heads, gqa_rep, 2, half_hd)
        k_cm = ch_max[f"k_{i}"].reshape(num_kv_heads, 2, half_hd)
        q_pair_max = np.maximum(np.max(q_cm, axis=(1, 2)), 1e-4)
        k_pair_max = np.maximum(np.max(k_cm, axis=1), 1e-4)
        kappa = np.sqrt(q_pair_max / k_pair_max)
        kappa_q = np.tile(kappa[:, None, :], (1, gqa_rep, 2)).reshape(num_q_heads * head_dim)
        kappa_k = np.tile(kappa, (1, 2)).reshape(num_kv_heads * head_dim)

        s_v = float(t_max[f"v_{i}"]) / 127.0

        # Save migrated weights
        _update_npy(f"{l_prefix}.input_layernorm.weight", in_g_tilde)
        _update_npy(f"{l_prefix}.post_attention_layernorm.weight", post_g_tilde)
        _update_npy(f"{l_prefix}.self_attn.q_proj.weight", wq_tilde)
        _update_npy(f"{l_prefix}.self_attn.k_proj.weight", wk_tilde)
        _update_npy(f"{l_prefix}.self_attn.v_proj.weight", wv_tilde)
        _update_npy(f"{l_prefix}.mlp.gate_proj.weight", wg_tilde)
        _update_npy(f"{l_prefix}.mlp.up_proj.weight", wu_tilde)
        _update_npy(f"{l_prefix}.mlp.down_proj.weight", wd_tilde)

        # Record tensor and per-channel scales
        calib_scales[f"h_attn_norm_{i}"] = s_an
        calib_scales[f"q_proj_{i}"] = s_q_base
        calib_scales[f"q_rope_{i}"] = s_q_base
        calib_scales[f"k_proj_{i}"] = s_k_base
        calib_scales[f"k_rope_{i}"] = s_k_base
        calib_scales[f"v_proj_{i}"] = s_v
        calib_scales[f"attn_context_{i}"] = s_v
        calib_scales[f"h_mlp_norm_{i}"] = s_mn
        calib_scales[f"gate_{i}"] = float(s_gate)
        calib_scales[f"silu_{i}"] = float(s_gate)
        calib_scales[f"up_{i}"] = s_up
        calib_scales[f"swiglu_{i}"] = s_swiglu

        calib_scales[f"__ch_scale__q_proj_{i}"] = (s_q_base * kappa_q).tolist()
        calib_scales[f"__ch_scale__k_proj_{i}"] = (s_k_base / kappa_k).tolist()
        calib_scales[f"__ch_scale__gate_proj_{i}"] = (s_gate * beta_g).tolist()
        calib_scales[f"__ch_scale__up_proj_{i}"] = (s_up * alpha_u).tolist()

    # 5. Final norm <-> lm_head migration
    embed_w = model.model.embed_tokens.weight.detach().float().cpu().numpy().copy()
    fn_g = model.model.norm.weight.detach().float().cpu().numpy().copy()
    fn_cm = np.maximum(ch_max["fn"], 1e-5)
    w_lm_cm = np.maximum(np.max(np.abs(embed_w), axis=0), 1e-5)
    d_fn = (fn_cm / w_lm_cm) ** 0.5
    d_fn = d_fn / np.median(d_fn)
    fn_g_tilde = fn_g / d_fn
    lm_w_tilde = embed_w * d_fn[None, :]
    s_fn = float(np.max(fn_cm / d_fn)) / 127.0

    _update_npy("model.norm.weight", fn_g_tilde)
    _update_npy("lm_head.weight", lm_w_tilde)
    calib_scales["h_final_norm"] = s_fn

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    calib_file = target_dir / "calib_scales.json"
    with open(calib_file, "w", encoding="utf-8") as f:
        json.dump(calib_scales, f, indent=2, sort_keys=True)

    return calib_scales


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
