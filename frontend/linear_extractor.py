"""Extract and classify linear projection modules from pretrained transformer models."""

import re
from dataclasses import dataclass

import torch

PROJECTION_TYPES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class ExtractedLinear:
    """Represents a classified linear projection layer extracted from a model."""

    name: str
    layer_idx: int
    proj_type: str
    module_path: str
    in_features: int  # K
    out_features: int  # N
    has_bias: bool
    dtype: str
    weight: torch.Tensor
    bias: torch.Tensor | None = None


def extract_linear_modules(model: torch.nn.Module) -> list[ExtractedLinear]:
    """Recursively search model for the 7 standard transformer linear projections.

    Extracts:
        - Self-attention: q_proj, k_proj, v_proj, o_proj
        - MLP / SwiGLU: gate_proj, up_proj, down_proj
    """
    extracted: list[ExtractedLinear] = []

    for path, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        matched_type = None
        for p_type in PROJECTION_TYPES:
            if path.endswith(f".{p_type}") or path == p_type:
                matched_type = p_type
                break

        if matched_type is None:
            # Exclude non-decoder linear projections (e.g. lm_head)
            continue

        # Extract layer index from module path (e.g. model.layers.14.self_attn.q_proj)
        layer_match = re.search(r"layers\.(\d+)\.", path)
        layer_idx = int(layer_match.group(1)) if layer_match else 0

        has_bias = module.bias is not None
        bias_tensor = module.bias.detach().clone() if has_bias else None
        weight_tensor = module.weight.detach().clone()

        extracted.append(
            ExtractedLinear(
                name=f"layer{layer_idx}_{matched_type}",
                layer_idx=layer_idx,
                proj_type=matched_type,
                module_path=path,
                in_features=module.in_features,
                out_features=module.out_features,
                has_bias=has_bias,
                dtype=str(module.weight.dtype).replace("torch.", ""),
                weight=weight_tensor,
                bias=bias_tensor,
            )
        )

    # Sort deterministically by layer_idx, then projection type order
    type_order = {t: i for i, t in enumerate(PROJECTION_TYPES)}
    extracted.sort(key=lambda m: (m.layer_idx, type_order.get(m.proj_type, 99)))
    return extracted
