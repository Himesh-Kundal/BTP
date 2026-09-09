"""Forward-hook activation capture on a fixed calibration set."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from frontend.artifact_dumper import compute_file_sha256
from frontend.linear_extractor import PROJECTION_TYPES
from frontend.model_loader import DEFAULT_MODEL_ID, PINNED_REVISION

DEFAULT_CALIB_PROMPTS = [
    "The theory of computation provides foundational insights into algorithm complexity.",
    "In hardware design, fixed-point arithmetic allows specialization of multiplier arrays.",
    "Large language models rely heavily on dense matrix-vector multiplications.",
    "The quick brown fox jumps over the lazy dog.",
]
DEFAULT_CALIB_SEED = 42


def capture_activations(
    model: torch.nn.Module,
    tokenizer: Any,
    layers: list[int] | None = None,
    prompts: list[str] | None = None,
    seed: int = DEFAULT_CALIB_SEED,
    base_dir: Path = Path("artifacts"),
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = PINNED_REVISION,
) -> list[Path]:
    """Capture input activations for linear projections using forward hooks on a fixed calibration set.

    Saves:
        artifacts/<model>@<rev>/layer_<idx>/<proj_type>/calib/x_sample.npy
        artifacts/<model>@<rev>/layer_<idx>/<proj_type>/calib/calib.json
    """
    if prompts is None:
        prompts = list(DEFAULT_CALIB_PROMPTS)

    # Set fixed seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.eval()

    target_layers = set(layers) if layers is not None else None
    hooks = []
    captured_tensors: dict[str, list[torch.Tensor]] = {}

    for path, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        matched_type = None
        for p_type in PROJECTION_TYPES:
            if path.endswith(f".{p_type}") or path == p_type:
                matched_type = p_type
                break

        if matched_type is None:
            continue

        # Extract layer index
        import re

        layer_match = re.search(r"layers\.(\d+)\.", path)
        layer_idx = int(layer_match.group(1)) if layer_match else 0

        if target_layers is not None and layer_idx not in target_layers:
            continue

        key = f"layer_{layer_idx}/{matched_type}"
        captured_tensors[key] = []

        def make_hook(mod_key: str):
            def hook_fn(_mod, inp, _out):
                # inp[0] is shape [batch, seq_len, in_features]
                captured_tensors[mod_key].append(inp[0].detach().cpu().to(torch.float32))

            return hook_fn

        hook_handle = module.register_forward_hook(make_hook(key))
        hooks.append(hook_handle)

    # Run forward pass on calibration prompts
    try:
        with torch.no_grad():
            for text in prompts:
                inputs = tokenizer(text, return_tensors="pt")
                model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    # Process and save captured activations
    model_slug = model_id.split("/")[-1]
    rev_slug = revision[:12]
    calib_dirs: list[Path] = []

    for key, tensor_list in captured_tensors.items():
        if not tensor_list:
            continue

        # Concatenate along sequence/batch dimension -> [M, K]
        combined = torch.cat(tensor_list, dim=1 if tensor_list[0].ndim == 3 else 0)
        x_flat = combined.reshape(-1, combined.shape[-1]).numpy()

        target_calib_dir = base_dir / f"{model_slug}@{rev_slug}" / key / "calib"
        target_calib_dir.mkdir(parents=True, exist_ok=True)

        sample_file = target_calib_dir / "x_sample.npy"
        np.save(sample_file, x_flat)
        sha256_hash = compute_file_sha256(sample_file)

        abs_vals = np.abs(x_flat)
        calib_meta = {
            "model_id": model_id,
            "revision": revision,
            "key": key,
            "prompts": prompts,
            "seed": seed,
            "shape": list(x_flat.shape),
            "absmax": float(np.max(abs_vals)),
            "percentile_99": float(np.percentile(abs_vals, 99.0)),
            "percentile_99_9": float(np.percentile(abs_vals, 99.9)),
            "sha256": sha256_hash,
        }

        calib_json = target_calib_dir / "calib.json"
        with open(calib_json, "w", encoding="utf-8") as f:
            json.dump(calib_meta, f, indent=2)

        calib_dirs.append(target_calib_dir)

    return calib_dirs


def load_captured_activation(calib_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load captured activation sample and metadata with integrity check."""
    calib_p = Path(calib_dir)
    sample_file = calib_p / "x_sample.npy"
    meta_file = calib_p / "calib.json"

    if not sample_file.exists() or not meta_file.exists():
        raise FileNotFoundError(f"Missing activation files in {calib_p}")

    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)

    # Validate hash
    actual_sha = compute_file_sha256(sample_file)
    if actual_sha != meta["sha256"]:
        raise ValueError(f"Activation sample SHA256 mismatch in {calib_p}")

    x_act = np.load(sample_file)
    return x_act, meta
