"""Tests for RTL generation determinism and header hygiene (T4.7)."""

import hashlib
import re
from pathlib import Path

import numpy as np

from backend.module_generator import generate_full_module
from backend.tiling import tile_weight_matrix
from ir.linear import LinearOp


def test_rtl_generation_determinism_byte_identical(tmp_path: Path):
    """Verify that generating full module RTL twice from the same inputs is byte-identical."""
    rng = np.random.default_rng(42)
    # 16x32 matrix tiled into 8x16 (2x2 tiles)
    w = rng.integers(-127, 127, size=(16, 32), dtype=np.int8)
    b = rng.integers(-1000, 1000, size=(16,), dtype=np.int32)

    op = LinearOp(name="det_test_layer", in_features=32, out_features=16, weight=w, bias=b, is_quantized=True)
    manifest1, _ = tile_weight_matrix(w, tile_n=8, tile_k=16, layer_name=op.name)
    manifest2, _ = tile_weight_matrix(w, tile_n=8, tile_k=16, layer_name=op.name)

    # Run 1
    rtl1 = generate_full_module(op, manifest1, tile_n=8, tile_k=16, latency=1)
    file1 = tmp_path / "run1" / "layer.v"
    file1.parent.mkdir(parents=True)
    file1.write_text(rtl1, encoding="utf-8")
    sha1 = hashlib.sha256(file1.read_bytes()).hexdigest()

    # Run 2
    rtl2 = generate_full_module(op, manifest2, tile_n=8, tile_k=16, latency=1)
    file2 = tmp_path / "run2" / "layer.v"
    file2.parent.mkdir(parents=True)
    file2.write_text(rtl2, encoding="utf-8")
    sha2 = hashlib.sha256(file2.read_bytes()).hexdigest()

    # Byte-identical assert
    assert sha1 == sha2, "Generated RTL hashes do not match across runs!"
    assert rtl1 == rtl2, "RTL source differs between identical generation runs!"


def test_rtl_headers_omit_timestamps_and_paths():
    """Verify that generated RTL headers do NOT contain timestamps or absolute file paths."""
    w = np.array([[1, 2], [3, 4]], dtype=np.int8)
    op = LinearOp(name="clean_header_test", in_features=2, out_features=2, weight=w, is_quantized=True)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=2, layer_name=op.name)

    rtl = generate_full_module(op, manifest, tile_n=2, tile_k=2, latency=1)

    # 1. No timestamps: YYYY-MM-DD or HH:MM:SS
    assert not re.search(r"\b20\d{2}-\d{2}-\d{2}\b", rtl), "Timestamp (date) found in generated RTL!"
    assert not re.search(r"\b[0-2][0-9]:[0-5][0-9]:[0-5][0-9]\b", rtl), "Timestamp (time) found in generated RTL!"

    # 2. No absolute paths
    assert not re.search(r"/home/\w+", rtl), "Absolute path /home/ found in generated RTL!"
    assert not re.search(r"/tmp/\w+", rtl), "Absolute path /tmp/ found in generated RTL!"
    assert not re.search(r"/Users/\w+", rtl), "Absolute path /Users/ found in generated RTL!"

    # 3. Contains manifest SHA256
    assert f"Manifest SHA256: {manifest.manifest_hash}" in rtl


def test_tile_manifest_determinism():
    """Verify that TileManifest generation and hash computation is strictly deterministic."""
    rng = np.random.default_rng(777)
    w = rng.integers(-100, 100, size=(24, 48), dtype=np.int8)

    m1, _ = tile_weight_matrix(w, tile_n=8, tile_k=16, layer_name="det_manifest")
    m2, _ = tile_weight_matrix(w, tile_n=8, tile_k=16, layer_name="det_manifest")

    assert m1.manifest_hash == m2.manifest_hash
    assert m1.to_dict() == m2.to_dict()
