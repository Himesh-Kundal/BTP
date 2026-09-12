"""Tests for complete IR specification, dataclasses, serialization, and artifact loading (T3.1)."""

import json
from pathlib import Path

import numpy as np
import pytest

from ir.graph import Graph
from ir.linear import LinearOp, load_linear_from_artifact
from ir.tensor import TensorKind, TensorRef


def test_tensor_ref_creation_and_attributes():
    """Verify TensorRef creation, SHA256 calculation, and array interface."""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    t = TensorRef(
        name="test_tensor",
        shape=data.shape,
        dtype="int8",
        kind=TensorKind.CONSTANT,
        data=data,
    )
    assert t.name == "test_tensor"
    assert t.shape == (2, 3)
    assert t.dtype == "int8"
    assert t.kind == TensorKind.CONSTANT
    assert t.sha256 is not None
    assert len(t.sha256) == 64
    assert t[0, 1] == 2
    assert np.array_equal(np.asarray(t), data)


def test_tensor_ref_serialization_and_sidecar(tmp_path: Path):
    """Verify saving tensor to .npy sidecar and verifying SHA256 on reload."""
    data = np.array([[10, -20], [30, -40]], dtype=np.int32)
    t = TensorRef(
        name="bias_tensor",
        shape=data.shape,
        dtype="int32",
        kind=TensorKind.CONSTANT,
        data=data,
    )
    sidecar_path = t.save_array(tmp_path)
    assert sidecar_path.is_file()
    assert t.path == sidecar_path.name

    # Test dictionary serialization
    d = t.to_dict()
    assert d["name"] == "bias_tensor"
    assert d["shape"] == [2, 2]
    assert d["dtype"] == "int32"
    assert d["kind"] == "CONSTANT"
    assert d["sha256"] == t.sha256

    # Reload from dict and path
    t_reloaded = TensorRef.from_dict(d, base_dir=tmp_path)
    assert t_reloaded.data is not None
    assert np.array_equal(t_reloaded.data, data)
    assert t_reloaded.sha256 == t.sha256

    # Tampered file triggers hash mismatch
    np.save(sidecar_path, np.array([[99, 99], [99, 99]], dtype=np.int32))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        TensorRef.from_dict(d, base_dir=tmp_path)


def test_linear_op_complete_fields_and_invariants():
    """Verify full LinearOp dataclass, metadata fields, and Invariants I1-I3."""
    w_data = np.array([[3, 7, -2]], dtype=np.int8)
    b_data = np.array([12], dtype=np.int32)
    w_scale = np.array([0.05], dtype=np.float64)

    op = LinearOp(
        name="full_linear",
        in_features=3,
        out_features=1,
        weight=w_data,
        bias=b_data,
        weight_scale=w_scale,
        act_scale=0.1,
        act_dtype="int8",
        acc_dtype="int32",
        acc_bits_min=16,
        quant={
            "scheme": "symmetric",
            "rounding": "round_half_to_even",
            "stat": "absmax",
        },
        hw={"tile_n": 8, "tile_k": 16, "arch": "fixed"},
        source={"model": "SmolLM2-135M", "layer": 0, "module": "q_proj"},
    )

    # Check invariants
    assert op.weight.shape == (1, 3)
    assert op.bias is not None and op.bias.shape == (1,)
    assert op.weight.kind == TensorKind.CONSTANT  # Invariant I3
    assert op.acc_bits_min == 16
    assert op.quant["scheme"] == "symmetric"
    assert op.hw["arch"] == "fixed"
    assert op.source["module"] == "q_proj"


def test_linear_op_serialization_roundtrip_lossless(tmp_path: Path):
    """Invariant I4: Serialization round-trip to JSON and .npy sidecars is lossless."""
    w_data = np.array([[12, -45, 0], [127, -127, 34]], dtype=np.int8)
    b_data = np.array([1000, -2000], dtype=np.int32)
    w_scale = np.array([0.02, 0.04], dtype=np.float64)

    op = LinearOp(
        name="roundtrip_linear",
        in_features=3,
        out_features=2,
        weight=w_data,
        bias=b_data,
        weight_scale=w_scale,
        act_scale=0.15,
        acc_bits_min=20,
        quant={"scheme": "symmetric"},
        hw={"tile_n": 8, "tile_k": 16},
        source={"model": "SmolLM2-135M"},
    )

    json_path = op.save(tmp_path)
    assert json_path.is_file()

    # Verify JSON structure
    with open(json_path) as f:
        data = json.load(f)
    assert data["op"] == "linear"
    assert data["name"] == "roundtrip_linear"
    assert data["in_features"] == 3
    assert data["out_features"] == 2

    # Load back and verify bit-identity
    reloaded_op = LinearOp.load(json_path)
    assert reloaded_op.name == op.name
    assert reloaded_op.in_features == op.in_features
    assert reloaded_op.out_features == op.out_features
    assert np.array_equal(reloaded_op.weight.data, op.weight.data)
    assert reloaded_op.bias is not None and op.bias is not None
    assert np.array_equal(reloaded_op.bias.data, op.bias.data)
    assert reloaded_op.weight_scale is not None and op.weight_scale is not None
    assert np.array_equal(reloaded_op.weight_scale, op.weight_scale)
    assert reloaded_op.act_scale == op.act_scale
    assert reloaded_op.acc_bits_min == op.acc_bits_min
    assert reloaded_op.quant == op.quant
    assert reloaded_op.hw == op.hw
    assert reloaded_op.source == op.source


def test_graph_container(tmp_path: Path):
    """Verify Graph container with multiple operators."""
    w1 = np.array([[1, 2], [3, 4]], dtype=np.int8)
    w2 = np.array([[5, 6]], dtype=np.int8)
    op1 = LinearOp(name="layer1", in_features=2, out_features=2, weight=w1)
    op2 = LinearOp(name="layer2", in_features=2, out_features=1, weight=w2)

    graph = Graph(name="toy_mlp", operators=[op1, op2])
    assert len(graph.operators) == 2
    assert graph.get_operator("layer1") is op1
    assert graph.get_operator("layer2") is op2

    # Save and reload graph
    graph_path = graph.save(tmp_path)
    assert graph_path.is_file()

    reloaded_graph = Graph.load(graph_path)
    assert reloaded_graph.name == "toy_mlp"
    assert len(reloaded_graph.operators) == 2
    assert np.array_equal(
        reloaded_graph.get_operator("layer1").weight.data, w1
    )
    assert np.array_equal(
        reloaded_graph.get_operator("layer2").weight.data, w2
    )


def test_load_linear_from_artifact():
    """Verify loading real dumped layer artifact into unquantized LinearOp."""
    artifact_dir = (
        Path("artifacts")
        / "SmolLM2-135M@93efa2f097d5"
        / "layer_0"
        / "q_proj"
    )
    if not artifact_dir.exists():
        pytest.skip("SmolLM2-135M layer 0 q_proj artifact not found on disk")

    op = load_linear_from_artifact(artifact_dir, name="layer_0_q_proj")
    assert op.name == "layer_0_q_proj"
    assert op.in_features == 576
    assert op.out_features == 576
    assert op.weight.dtype == "float32"
    assert op.weight.shape == (576, 576)
    assert op.weight.kind == TensorKind.CONSTANT
    assert op.source["module_path"] == "model.layers.0.self_attn.q_proj"
    assert op.bias is None
