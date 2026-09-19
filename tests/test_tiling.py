"""Tests for matrix tiling, zero-padding, and manifest generation (T4.2)."""

import json
from pathlib import Path

import numpy as np
import pytest

from backend.tiling import (
    TileManifest,
    TileSpec,
    pad_activations,
    pad_matrix,
    tile_weight_matrix,
    unpad_outputs,
)


def test_pad_matrix_exact_multiple():
    """Verify padding when matrix dimensions are already exact multiples."""
    w = np.arange(8 * 16, dtype=np.int8).reshape(8, 16)
    w_pad, n_pad, k_pad = pad_matrix(w, tile_n=8, tile_k=16)

    assert n_pad == 8
    assert k_pad == 16
    assert w_pad.shape == (8, 16)
    np.testing.assert_array_equal(w_pad, w)


def test_pad_matrix_non_multiple():
    """Verify zero padding when matrix dimensions require rounding up."""
    # N=10, K=19 -> N'=16, K'=32 with tile (8, 16)
    rng = np.random.default_rng(42)
    w = rng.integers(-100, 100, size=(10, 19), dtype=np.int8)

    w_pad, n_pad, k_pad = pad_matrix(w, tile_n=8, tile_k=16)

    assert n_pad == 16
    assert k_pad == 32
    assert w_pad.shape == (16, 32)
    assert w_pad.dtype == np.int8

    # Unpadded region is bit-identical
    np.testing.assert_array_equal(w_pad[:10, :19], w)

    # Padding regions are strictly zero
    np.testing.assert_array_equal(w_pad[10:, :], 0)
    np.testing.assert_array_equal(w_pad[:, 19:], 0)


def test_pad_activations_and_unpad_outputs():
    """Verify activation padding and output unpadding."""
    # M=3, K=19 -> K'=32 with tile_k=16
    rng = np.random.default_rng(101)
    x = rng.integers(-50, 50, size=(3, 19), dtype=np.int8)

    x_pad, k_pad = pad_activations(x, tile_k=16)
    assert k_pad == 32
    assert x_pad.shape == (3, 32)
    assert x_pad.dtype == np.int8
    np.testing.assert_array_equal(x_pad[:, :19], x)
    np.testing.assert_array_equal(x_pad[:, 19:], 0)

    # Output unpadding: M=3, N'=16 -> N=10
    y_pad = rng.integers(-1000, 1000, size=(3, 16), dtype=np.int32)
    y_unpad = unpad_outputs(y_pad, original_n=10)
    assert y_unpad.shape == (3, 10)
    np.testing.assert_array_equal(y_unpad, y_pad[:, :10])


def test_tile_weight_matrix_reconstruction_property():
    """Property test: assembling all tiles reproduces the padded matrix exactly."""
    rng = np.random.default_rng(2026)
    # Test on SmolLM2-like shape slice: N=25, K=35 with tile (8, 16)
    # N'=32 (4 tiles), K'=48 (3 tiles) -> 12 tiles
    w = rng.integers(-127, 127, size=(25, 35), dtype=np.int8)

    manifest, tile_grid = tile_weight_matrix(w, tile_n=8, tile_k=16, layer_name="layer0_test")

    assert manifest.n == 25
    assert manifest.k == 35
    assert manifest.n_padded == 32
    assert manifest.k_padded == 48
    assert manifest.num_tiles_n == 4
    assert manifest.num_tiles_k == 3
    assert manifest.total_tiles == 12
    assert len(manifest.tiles) == 12

    # Reconstruct matrix from grid
    reconstructed = np.zeros((32, 48), dtype=np.int8)
    for i in range(4):
        for j in range(3):
            tile = tile_grid[i][j]
            assert tile.i == i
            assert tile.j == j
            assert tile.weights.shape == (8, 16)
            assert tile.nonzero_count == int(np.count_nonzero(tile.weights))
            reconstructed[i * 8 : (i + 1) * 8, j * 16 : (j + 1) * 16] = tile.weights

    # Padded slice matches original matrix
    np.testing.assert_array_equal(reconstructed[:25, :35], w)
    # Padding rows and cols are strictly zero
    np.testing.assert_array_equal(reconstructed[25:, :], 0)
    np.testing.assert_array_equal(reconstructed[:, 35:], 0)


def test_tile_manifest_serialization(tmp_path: Path):
    """Verify TileManifest serializes and deserializes losslessly."""
    w = np.array([[1, 2, 0, -4], [0, 0, 7, 8]], dtype=np.int8)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=2, layer_name="mini_test")

    json_path = tmp_path / "manifest.json"
    manifest.save(json_path)
    assert json_path.exists()

    with open(json_path) as f:
        data = json.load(f)

    assert data["layer_name"] == "mini_test"
    assert data["n"] == 2
    assert data["k"] == 4
    assert data["tile_n"] == 2
    assert data["tile_k"] == 2
    assert len(data["tiles"]) == 2

    # Reload and compare
    reloaded = TileManifest.load(json_path)
    assert reloaded.layer_name == manifest.layer_name
    assert reloaded.manifest_hash == manifest.manifest_hash
    assert len(reloaded.tiles) == len(manifest.tiles)
    for t1, t2 in zip(manifest.tiles, reloaded.tiles, strict=True):
        assert t1.i == t2.i
        assert t1.j == t2.j
        assert t1.module_name == t2.module_name
        assert t1.nonzero_count == t2.nonzero_count
        assert t1.weight_sha256 == t2.weight_sha256


def test_tiling_exact_smollm_layer0_shape():
    """Verify tiling on SmolLM2 576x576 standard projection shape."""
    rng = np.random.default_rng(777)
    w = rng.integers(-127, 127, size=(576, 576), dtype=np.int8)

    manifest, tile_grid = tile_weight_matrix(w, tile_n=8, tile_k=16, layer_name="layer0_q_proj")
    # 576 / 8 = 72, 576 / 16 = 36 -> 72 * 36 = 2592 tiles
    assert manifest.num_tiles_n == 72
    assert manifest.num_tiles_k == 36
    assert manifest.total_tiles == 2592
    assert len(tile_grid) == 72
    assert len(tile_grid[0]) == 36
    assert manifest.n_padded == 576
    assert manifest.k_padded == 576
