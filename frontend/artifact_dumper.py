"""Dump and reload weight artifacts (weight.npy, meta.json) with SHA256 integrity checks."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from frontend.linear_extractor import ExtractedLinear, extract_linear_modules
from frontend.model_loader import DEFAULT_MODEL_ID, PINNED_REVISION


def compute_file_sha256(path: Path) -> str:
    """Compute standard SHA256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def dump_linear_artifact(
    extracted: ExtractedLinear,
    base_dir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = PINNED_REVISION,
) -> Path:
    """Dump a single linear layer's weight, bias, and metadata to disk."""
    model_slug = model_id.split("/")[-1]
    rev_slug = revision[:12]
    target_dir = base_dir / f"{model_slug}@{rev_slug}" / f"layer_{extracted.layer_idx}" / extracted.proj_type
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save weight.npy (raw numpy array)
    weight_np = extracted.weight.detach().cpu().numpy()
    weight_file = target_dir / "weight.npy"
    np.save(weight_file, weight_np)
    weight_sha = compute_file_sha256(weight_file)

    # 2. Save bias.npy if present
    bias_sha = None
    if extracted.has_bias and extracted.bias is not None:
        bias_np = extracted.bias.detach().cpu().numpy()
        bias_file = target_dir / "bias.npy"
        np.save(bias_file, bias_np)
        bias_sha = compute_file_sha256(bias_file)

    # 3. Save meta.json
    meta: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "name": extracted.name,
        "layer_idx": extracted.layer_idx,
        "proj_type": extracted.proj_type,
        "module_path": extracted.module_path,
        "in_features": extracted.in_features,
        "out_features": extracted.out_features,
        "shape": [extracted.out_features, extracted.in_features],
        "dtype": str(weight_np.dtype),
        "weight_sha256": weight_sha,
        "has_bias": extracted.has_bias,
        "bias_sha256": bias_sha,
    }

    meta_file = target_dir / "meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return target_dir


def load_linear_artifact(target_dir: Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Reload dumped artifact and verify SHA256 integrity."""
    target_p = Path(target_dir)
    weight_file = target_p / "weight.npy"
    meta_file = target_p / "meta.json"

    if not weight_file.exists() or not meta_file.exists():
        raise FileNotFoundError(f"Missing weight.npy or meta.json in {target_p}")

    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)

    # Verify SHA256 integrity
    actual_weight_sha = compute_file_sha256(weight_file)
    if actual_weight_sha != meta["weight_sha256"]:
        raise ValueError(
            f"Weight SHA256 mismatch in {target_p}: recorded {meta['weight_sha256']} vs actual {actual_weight_sha}"
        )

    weight = np.load(weight_file)

    bias = None
    if meta.get("has_bias"):
        bias_file = target_p / "bias.npy"
        if not bias_file.exists():
            raise FileNotFoundError(f"Expected bias.npy in {target_p}")
        actual_bias_sha = compute_file_sha256(bias_file)
        if actual_bias_sha != meta["bias_sha256"]:
            raise ValueError("Bias SHA256 mismatch")
        bias = np.load(bias_file)

    return weight, bias, meta


def dump_all_artifacts(
    model: torch.nn.Module,
    layers: list[int] | None = None,
    base_dir: Path = Path("artifacts"),
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = PINNED_REVISION,
) -> list[Path]:
    """Extract and dump linear artifacts for specified layers (or all layers if None)."""
    extracted_modules = extract_linear_modules(model)
    if layers is not None:
        target_layers = set(layers)
        extracted_modules = [m for m in extracted_modules if m.layer_idx in target_layers]

    dumped_dirs: list[Path] = []
    for mod in extracted_modules:
        d = dump_linear_artifact(mod, base_dir=base_dir, model_id=model_id, revision=revision)
        dumped_dirs.append(d)
    return dumped_dirs
