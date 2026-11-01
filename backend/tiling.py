"""Matrix tiling, zero-padding, and manifest generation (T4.2).

Slices INT8 weight matrices into parametric Nt x Kt tiles (default 8x16),
pads matrix dimensions with zeros when dimensions are not exact multiples,
and produces the tile manifest tracking tile geometries, non-zero counts,
and cryptographic SHA256 hashes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclasses.dataclass
class TileSpec:
    """Specification for a single matrix tile."""

    i: int
    j: int
    module_name: str
    nonzero_count: int
    weight_sha256: str
    weights: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Convert tile specification to dictionary representation."""
        return {
            "i": self.i,
            "j": self.j,
            "module_name": self.module_name,
            "nonzero_count": self.nonzero_count,
            "weight_sha256": self.weight_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], weights: np.ndarray | None = None) -> TileSpec:
        """Construct TileSpec from dictionary."""
        return cls(
            i=data["i"],
            j=data["j"],
            module_name=data["module_name"],
            nonzero_count=data["nonzero_count"],
            weight_sha256=data["weight_sha256"],
            weights=weights if weights is not None else np.zeros((0, 0), dtype=np.int8),
        )


@dataclasses.dataclass
class TileManifest:
    """Manifest describing tiling geometry, tile hierarchy, and integrity hashes."""

    layer_name: str
    n: int
    k: int
    n_padded: int
    k_padded: int
    tile_n: int
    tile_k: int
    num_tiles_n: int
    num_tiles_k: int
    total_tiles: int
    tiles: list[TileSpec]
    manifest_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest metadata to dictionary."""
        return {
            "layer_name": self.layer_name,
            "n": self.n,
            "k": self.k,
            "n_padded": self.n_padded,
            "k_padded": self.k_padded,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "num_tiles_n": self.num_tiles_n,
            "num_tiles_k": self.num_tiles_k,
            "total_tiles": self.total_tiles,
            "tiles": [t.to_dict() for t in self.tiles],
            "manifest_hash": self.manifest_hash,
        }

    def save(self, path: Path | str) -> None:
        """Save manifest to JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path | str) -> TileManifest:
        """Load manifest from JSON file."""
        target = Path(path)
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
        tiles = [TileSpec.from_dict(t) for t in data["tiles"]]
        return cls(
            layer_name=data["layer_name"],
            n=data["n"],
            k=data["k"],
            n_padded=data["n_padded"],
            k_padded=data["k_padded"],
            tile_n=data["tile_n"],
            tile_k=data["tile_k"],
            num_tiles_n=data["num_tiles_n"],
            num_tiles_k=data["num_tiles_k"],
            total_tiles=data["total_tiles"],
            tiles=tiles,
            manifest_hash=data.get("manifest_hash", ""),
        )


def pad_matrix(w: np.ndarray, tile_n: int, tile_k: int) -> tuple[np.ndarray, int, int]:
    """Pad weight matrix with zeros to exact multiples of (tile_n, tile_k).

    Args:
        w: Weight array of shape (N, K).
        tile_n: Tile height (Nt).
        tile_k: Tile width (Kt).

    Returns:
        Tuple of (w_padded, n_padded, k_padded).
    """
    n, k = w.shape
    n_padded = ((n + tile_n - 1) // tile_n) * tile_n
    k_padded = ((k + tile_k - 1) // tile_k) * tile_k

    if n_padded == n and k_padded == k:
        return w.copy(), n, k

    w_padded = np.zeros((n_padded, k_padded), dtype=w.dtype)
    w_padded[:n, :k] = w
    return w_padded, n_padded, k_padded


def pad_activations(x: np.ndarray, tile_k: int) -> tuple[np.ndarray, int]:
    """Pad input activations with zeros along feature dimension K to multiple of tile_k.

    Args:
        x: Activation array of shape (M, K) or (K,).
        tile_k: Tile width (Kt).

    Returns:
        Tuple of (x_padded, k_padded).
    """
    if x.ndim == 1:
        k = x.shape[0]
        k_padded = ((k + tile_k - 1) // tile_k) * tile_k
        if k_padded == k:
            return x.copy(), k
        x_padded = np.zeros(k_padded, dtype=x.dtype)
        x_padded[:k] = x
        return x_padded, k_padded

    m, k = x.shape
    k_padded = ((k + tile_k - 1) // tile_k) * tile_k
    if k_padded == k:
        return x.copy(), k

    x_padded = np.zeros((m, k_padded), dtype=x.dtype)
    x_padded[:, :k] = x
    return x_padded, k_padded


def unpad_outputs(y_padded: np.ndarray, original_n: int) -> np.ndarray:
    """Strip padded output channels back to original out_features N.

    Args:
        y_padded: Output array of shape (..., N').
        original_n: Original out_features N.

    Returns:
        Array of shape (..., N) with padding removed.
    """
    return y_padded[..., :original_n]


def tile_weight_matrix(
    w: np.ndarray,
    tile_n: int = 8,
    tile_k: int = 16,
    layer_name: str = "linear",
    module_prefix: str = "llmhw",
) -> tuple[TileManifest, list[list[TileSpec]]]:
    """Slice weight matrix into Nt x Kt tiles and construct TileManifest.

    Args:
        w: Weight array of shape (N, K).
        tile_n: Height of each tile (Nt), default 8.
        tile_k: Width of each tile (Kt), default 16.
        layer_name: Identifier for the layer.
        module_prefix: Prefix for generated module names.

    Returns:
        Tuple of (manifest, tile_grid) where tile_grid is a 2D list of TileSpec [num_tiles_n][num_tiles_k].
    """
    n, k = w.shape
    w_padded, n_padded, k_padded = pad_matrix(w, tile_n, tile_k)

    num_tiles_n = n_padded // tile_n
    num_tiles_k = k_padded // tile_k
    total_tiles = num_tiles_n * num_tiles_k

    tiles_list: list[TileSpec] = []
    tile_grid: list[list[TileSpec]] = []

    # Clean layer name for Verilog module identifier
    clean_layer_name = layer_name.replace(".", "_").replace("-", "_").lower()

    for i in range(num_tiles_n):
        row_tiles: list[TileSpec] = []
        for j in range(num_tiles_k):
            sub_w = w_padded[i * tile_n : (i + 1) * tile_n, j * tile_k : (j + 1) * tile_k]
            # Ensure C-contiguous array for deterministic byte hashing
            sub_w_c = np.ascontiguousarray(sub_w, dtype=np.int8)
            sha = hashlib.sha256(sub_w_c.tobytes()).hexdigest()
            nonzero = int(np.count_nonzero(sub_w_c))
            mod_name = f"{module_prefix}_{clean_layer_name}_tile_i{i}_j{j}"

            spec = TileSpec(
                i=i,
                j=j,
                module_name=mod_name,
                nonzero_count=nonzero,
                weight_sha256=sha,
                weights=sub_w_c,
            )
            row_tiles.append(spec)
            tiles_list.append(spec)
        tile_grid.append(row_tiles)

    # Compute manifest hash from canonical JSON
    manifest_data = {
        "layer_name": clean_layer_name,
        "n": n,
        "k": k,
        "n_padded": n_padded,
        "k_padded": k_padded,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "total_tiles": total_tiles,
        "tiles": [t.to_dict() for t in tiles_list],
    }
    canonical_json = json.dumps(manifest_data, sort_keys=True).encode("utf-8")
    manifest_hash = hashlib.sha256(canonical_json).hexdigest()

    manifest = TileManifest(
        layer_name=clean_layer_name,
        n=n,
        k=k,
        n_padded=n_padded,
        k_padded=k_padded,
        tile_n=tile_n,
        tile_k=tile_k,
        num_tiles_n=num_tiles_n,
        num_tiles_k=num_tiles_k,
        total_tiles=total_tiles,
        tiles=tiles_list,
        manifest_hash=manifest_hash,
    )

    return manifest, tile_grid
