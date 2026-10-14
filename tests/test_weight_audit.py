"""Tests for static weight audit and constant extractor (T6.4).

Verifies:
1. RTL weight parsing from single-core combinational modules.
2. RTL weight parsing from multi-tile structured modules (including non-multiple padded shapes).
3. RTL weight parsing from real SmolLM2 layer modules.
4. SHA256 integrity verification against manifest hash.
5. Negative controls: mutated constant, flipped sign, and dropped term trigger failure.
"""

from pathlib import Path

import numpy as np
import pytest

from backend.codegen import generate_fixed_core
from backend.module_generator import generate_full_module
from backend.tiling import tile_weight_matrix
from compiler.quantization import quantize_weights
from ir.linear import LinearOp
from verification.weight_audit import audit_weights_match, parse_rtl_weights


def test_parse_rtl_weights_toy_core() -> None:
    """Verify weight parsing on a 1x3 single-module combinational core."""
    w = np.array([[10, -5, 1]], dtype=np.int8)
    op = LinearOp(name="toy_audit", in_features=3, out_features=1, weight=w)
    rtl = generate_fixed_core(op)

    w_extracted = parse_rtl_weights(rtl, expected_n=1, expected_k=3)
    assert w_extracted.shape == (1, 3)
    assert np.array_equal(w_extracted, w)

    report = audit_weights_match(rtl, w, strict=True)
    assert report["passed"] is True
    assert report["mismatches"] == 0
    assert report["total_weights_checked"] == 3


def test_parse_rtl_weights_multi_tile_synthetic() -> None:
    """Verify weight parsing on a 4x8 multi-tile module (2x2 grid of 2x4 tiles)."""
    rng = np.random.default_rng(42)
    w = rng.integers(-127, 128, size=(4, 8), dtype=np.int8)
    # Ensure sparsity: include 0, 1, -1
    w[0, 0] = 0
    w[0, 1] = 1
    w[0, 2] = -1

    op = LinearOp(name="multi_audit", in_features=8, out_features=4, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=4)
    rtl = generate_full_module(op, manifest)

    w_extracted = parse_rtl_weights(rtl, expected_n=4, expected_k=8, manifest=manifest)
    assert w_extracted.shape == (4, 8)
    assert np.array_equal(w_extracted, w)

    report = audit_weights_match(rtl, w, manifest=manifest, strict=True)
    assert report["passed"] is True
    assert report["mismatches"] == 0
    assert report["total_weights_checked"] == 32


def test_parse_rtl_weights_non_multiple_shape() -> None:
    """Verify weight parsing on non-multiple 5x11 shape padded with tile 4x4."""
    rng = np.random.default_rng(100)
    w = rng.integers(-127, 128, size=(5, 11), dtype=np.int8)

    op = LinearOp(name="non_mult_audit", in_features=11, out_features=5, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=4, tile_k=4)
    rtl = generate_full_module(op, manifest)

    w_extracted = parse_rtl_weights(rtl, expected_n=5, expected_k=11, manifest=manifest)
    assert w_extracted.shape == (5, 11)
    assert np.array_equal(w_extracted, w)

    report = audit_weights_match(rtl, w, manifest=manifest, strict=True)
    assert report["passed"] is True
    assert report["mismatches"] == 0


def test_parse_rtl_weights_real_smollm_layer() -> None:
    """Verify static weight audit on real SmolLM2 Layer 0 q_proj slice."""
    repo_root = Path(__file__).resolve().parent.parent
    art_path = (
        repo_root / "artifacts" / "SmolLM2-135M@93efa2f097d5" / "layer_0" / "q_proj" / "weight.npy"
    )
    raw_w = np.load(art_path)
    w_q, _ = quantize_weights(raw_w[:8, :16])

    op = LinearOp(name="real_audit_q", in_features=16, out_features=8, weight=w_q)
    manifest, _ = tile_weight_matrix(w_q, tile_n=8, tile_k=16)
    rtl = generate_full_module(op, manifest)

    w_extracted = parse_rtl_weights(rtl, expected_n=8, expected_k=16, manifest=manifest)
    assert np.array_equal(w_extracted, w_q)

    report = audit_weights_match(rtl, w_q, manifest=manifest, strict=True)
    assert report["passed"] is True
    assert report["mismatches"] == 0


def test_negative_control_mutated_constant() -> None:
    """Negative control: mutating a single constant literal must trigger audit failure."""
    w = np.array([[10, 20], [30, 40]], dtype=np.int8)
    op = LinearOp(name="neg_const", in_features=2, out_features=2, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=2)
    rtl = generate_full_module(op, manifest)

    # Corrupt constant 8'sd20 to 8'sd21
    bad_rtl = rtl.replace("8'sd20", "8'sd21")
    assert bad_rtl != rtl

    report = audit_weights_match(bad_rtl, w, manifest=manifest, strict=False)
    assert report["passed"] is False
    assert report["mismatches"] >= 1

    with pytest.raises(ValueError, match="Weight audit mismatch"):
        audit_weights_match(bad_rtl, w, manifest=manifest, strict=True)


def test_negative_control_flipped_sign() -> None:
    """Negative control: flipping sign of a weight must trigger audit failure."""
    w = np.array([[15, -25], [35, -45]], dtype=np.int8)
    op = LinearOp(name="neg_sign", in_features=2, out_features=2, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=2)
    rtl = generate_full_module(op, manifest)

    # Flip sign: replace -8'sd25 with 8'sd25
    bad_rtl = rtl.replace("-8'sd25", "8'sd25")
    assert bad_rtl != rtl

    report = audit_weights_match(bad_rtl, w, manifest=manifest, strict=False)
    assert report["passed"] is False
    assert report["mismatches"] >= 1


def test_negative_control_dropped_term() -> None:
    """Negative control: dropping a product term must trigger audit failure."""
    w = np.array([[10, 20], [30, 40]], dtype=np.int8)
    op = LinearOp(name="neg_drop", in_features=2, out_features=2, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=2)
    rtl = generate_full_module(op, manifest)

    # Drop wire declaration for p_0_0
    lines = [line for line in rtl.splitlines() if "p_0_0" not in line or "wire" not in line]
    bad_rtl = "\n".join(lines)
    assert bad_rtl != rtl

    report = audit_weights_match(bad_rtl, w, manifest=manifest, strict=False)
    assert report["passed"] is False
    assert report["mismatches"] >= 1
