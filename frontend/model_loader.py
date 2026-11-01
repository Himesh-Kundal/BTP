"""Model loader and provenance tracking for Hugging Face transformer models."""

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
PINNED_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


def get_model_metadata(
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = PINNED_REVISION,
) -> dict[str, Any]:
    """Retrieve model configuration and metadata at pinned revision without downloading full weights."""
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    return {
        "model_id": model_id,
        "revision": revision,
        "config": config.to_dict(),
    }


def load_model_and_tokenizer(
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = PINNED_REVISION,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.nn.Module, Any]:
    """Load pretrained transformer and tokenizer in evaluation mode on CPU."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


def record_model_metadata(
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = PINNED_REVISION,
    out_path: str | Path = "results/model.json",
) -> dict[str, Any]:
    """Inspect model configuration and write provenance information to results/model.json."""
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    meta = get_model_metadata(model_id, revision)
    config_dict = meta["config"]

    provenance = {
        "model_id": model_id,
        "revision": revision,
        "model_type": config_dict.get("model_type", "llama"),
        "architecture": config_dict.get("architectures", ["LlamaForCausalLM"])[0],
        "config": {
            "hidden_size": config_dict.get("hidden_size"),
            "intermediate_size": config_dict.get("intermediate_size"),
            "num_hidden_layers": config_dict.get("num_hidden_layers"),
            "num_attention_heads": config_dict.get("num_attention_heads"),
            "num_key_value_heads": config_dict.get("num_key_value_heads"),
            "head_dim": config_dict.get("head_dim", config_dict.get("hidden_size", 576) // config_dict.get("num_attention_heads", 9)),
            "tie_word_embeddings": config_dict.get("tie_word_embeddings", True),
            "vocab_size": config_dict.get("vocab_size"),
        },
    }

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    return provenance
