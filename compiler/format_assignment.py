"""Format and Dyadic Scale Assignment Pass (Task UB.3).

Assigns exact bit-widths, container formats, and dyadic requantization parameters (M, s)
to all edges in Graph IR v2 according to the frozen configuration.
Folds gamma, 1/sqrt(d), and per-channel/per-row scales into dyadic constants.

Strict Layering (Golden Rule 6):
Only pure Python, NumPy, and IR dataclasses are used. Never import torch or transformers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ir.graph_v2 import EdgeFormat, GraphV2, OpKind, TensorKind
from numerics.ops import float_to_dyadic_scale


@dataclass
class FormatAssignmentResult:
    graph: GraphV2
    format_table: dict[str, Any]
    requant_params: dict[str, dict[str, list[int]]]

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "format_table": self.format_table,
            "requant_params": self.requant_params,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)


def assign_formats_and_scales(
    graph: GraphV2,
    weights_dict: dict[str, np.ndarray],
    frozen_config: dict[str, Any],
) -> FormatAssignmentResult:
    """Execute format and dyadic scale assignment pass on Graph IR v2."""
    fmt_cfg = frozen_config.get("formats", {})
    res_bits = fmt_cfg.get("residual_stream", {}).get("bits", 16)
    w_bits = fmt_cfg.get("linear_weights", {}).get("bits", 8)
    act_bits = fmt_cfg.get("linear_activations", {}).get("bits", 8)

    # Standard scales from calibration / numerics config
    s_act = 0.05
    s_res = 0.01

    fmt_token = EdgeFormat(bits=16, signed=False, scale=1.0, fixed_point="UINT16")
    fmt_res = EdgeFormat(
        bits=res_bits,
        signed=True,
        scale=s_res,
        fixed_point=f"INT{res_bits}",
    )
    fmt_act = EdgeFormat(
        bits=act_bits,
        signed=True,
        scale=s_act,
        fixed_point=f"INT{act_bits}",
    )
    fmt_weight = EdgeFormat(
        bits=w_bits,
        signed=True,
        scale=0.002,
        fixed_point=f"INT{w_bits}",
    )
    fmt_rope = EdgeFormat(
        bits=16,
        signed=True,
        scale=1.0 / 16384.0,
        fixed_point="Q1.14",
    )

    format_table: dict[str, Any] = {}
    requant_params: dict[str, dict[str, list[int]]] = {}

    # 1. Update tensor formats
    for name, tensor in graph.tensors.items():
        if tensor.kind == TensorKind.STATE:
            tensor.format = fmt_act
        elif tensor.kind == TensorKind.CONSTANT:
            if "weight" in name:
                tensor.format = fmt_weight
            elif "rope" in name:
                tensor.format = fmt_rope
            else:
                tensor.format = fmt_weight
        elif name in ("tok_in", "tok_out"):
            tensor.format = fmt_token
        elif "logits" in name or name.startswith("h_") or "down_" in name or "o_proj_" in name:
            tensor.format = fmt_res
        else:
            tensor.format = fmt_act

        format_table[name] = {
            "bits": tensor.format.bits,
            "signed": tensor.format.signed,
            "scale": tensor.format.scale,
            "fixed_point": tensor.format.fixed_point,
            "min_val": tensor.format.min_val,
            "max_val": tensor.format.max_val,
            "shape": tensor.shape,
        }

    # 2. Derive dyadic scales for each node
    for node in graph.nodes:
        if node.op == OpKind.Embed:
            # Per-row embedding dyadic scales into residual stream
            w_name = node.inputs[1]
            w_arr = weights_dict.get(w_name)
            if w_arr is not None:
                row_max = np.max(np.abs(w_arr), axis=-1)
                row_max = np.maximum(row_max, 1e-8)
                row_scales = row_max / 127.0
                m_list: list[int] = []
                s_list: list[int] = []
                for s_w in row_scales:
                    real_scale = s_w / s_res
                    M, s = float_to_dyadic_scale(real_scale)
                    m_list.append(M)
                    s_list.append(s)
                requant_params[node.id] = {"M": m_list, "shift": s_list}

        elif node.op == OpKind.RMSNorm:
            # Fold gamma into dyadic parameters
            g_name = node.inputs[1]
            gamma = weights_dict.get(g_name)
            if gamma is not None:
                m_list = []
                s_list = []
                for g_val in gamma:
                    M, s = float_to_dyadic_scale(float(g_val))
                    m_list.append(M)
                    s_list.append(s)
                requant_params[node.id] = {"M": m_list, "shift": s_list}

        elif node.op == OpKind.Linear:
            # Fold per-channel weight scales into (M, s)
            w_name = node.inputs[1]
            w_arr = weights_dict.get(w_name)
            if w_arr is not None:
                ch_max = np.max(np.abs(w_arr), axis=-1)
                ch_max = np.maximum(ch_max, 1e-8)
                ch_scales = ch_max / 127.0
                # Target output scale
                out_tensor = graph.tensors[node.outputs[0]]
                s_out = out_tensor.format.scale
                s_in = graph.tensors[node.inputs[0]].format.scale

                m_list = []
                s_list = []
                for s_w in ch_scales:
                    real_scale = (s_in * s_w) / s_out
                    M, s = float_to_dyadic_scale(float(real_scale))
                    m_list.append(M)
                    s_list.append(s)
                requant_params[node.id] = {"M": m_list, "shift": s_list}

        elif node.op == OpKind.Attention:
            # Fold 1/sqrt(head_dim) into attention scale
            hd = node.attributes.get("head_dim", 64)
            scale_factor = 1.0 / np.sqrt(hd)
            M, s = float_to_dyadic_scale(float(scale_factor))
            requant_params[node.id] = {"M": [M], "shift": [s]}

    return FormatAssignmentResult(
        graph=graph,
        format_table=format_table,
        requant_params=requant_params,
    )
