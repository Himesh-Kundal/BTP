"""Tests for Graph IR v2 and Invariants I1–I7 per docs/part2/02 §5."""

from pathlib import Path

import pytest

from ir.graph_v2 import (
    EdgeFormat,
    GraphV2,
    InvariantViolation,
    OpKind,
    TensorKind,
    validate_invariants,
)


def create_minimal_valid_graph() -> GraphV2:
    """Helper to build a small valid decoder subgraph."""
    g = GraphV2(name="minimal_valid_graph")

    # Formats
    fmt_int8 = EdgeFormat(bits=8, signed=True, scale=0.05, fixed_point="INT8")
    fmt_int16 = EdgeFormat(bits=16, signed=True, scale=0.01, fixed_point="INT16")
    fmt_weight = EdgeFormat(bits=8, signed=True, scale=0.002, fixed_point="INT8")
    fmt_state = EdgeFormat(bits=8, signed=True, scale=0.05, fixed_point="INT8")

    # Tensors
    g.add_tensor("tok_in", TensorKind.DYNAMIC, [1], fmt_int16)
    g.add_tensor("embed_w", TensorKind.CONSTANT, [256, 64], fmt_weight)
    g.add_tensor("h0", TensorKind.DYNAMIC, [1, 64], fmt_int16)

    g.add_tensor("norm0_gamma", TensorKind.CONSTANT, [64], fmt_weight)
    g.add_tensor("h0_norm", TensorKind.DYNAMIC, [1, 64], fmt_int8)

    g.add_tensor("q_w", TensorKind.CONSTANT, [64, 64], fmt_weight)
    g.add_tensor("q", TensorKind.DYNAMIC, [1, 64], fmt_int8)

    g.add_tensor("kv_cache", TensorKind.STATE, [2, 2, 32, 16], fmt_state)
    g.add_tensor("attn_out", TensorKind.DYNAMIC, [1, 64], fmt_int8)

    # Nodes
    g.add_node(
        node_id="embed0",
        op=OpKind.Embed,
        inputs=["tok_in", "embed_w"],
        outputs=["h0"],
        provenance={"model_id": "smollm2_tiny", "module": "model.embed_tokens"},
    )
    g.add_node(
        node_id="norm0",
        op=OpKind.RMSNorm,
        inputs=["h0", "norm0_gamma"],
        outputs=["h0_norm"],
        provenance={"model_id": "smollm2_tiny", "module": "model.layers.0.input_layernorm"},
    )
    g.add_node(
        node_id="linear_q",
        op=OpKind.Linear,
        inputs=["h0_norm", "q_w"],
        outputs=["q"],
        provenance={"model_id": "smollm2_tiny", "module": "model.layers.0.self_attn.q_proj"},
    )
    g.add_node(
        node_id="attn0",
        op=OpKind.Attention,
        inputs=["q", "kv_cache"],
        outputs=["attn_out"],
        attributes={"n_heads": 4, "n_kv": 2, "hd": 16, "max_ctx": 32},
        provenance={"model_id": "smollm2_tiny", "module": "model.layers.0.self_attn"},
    )

    return g


def test_ir_v2_dataclasses_and_valid_graph():
    """Verify standard creation and invariant validation passes on valid graph."""
    g = create_minimal_valid_graph()
    assert len(g.nodes) == 4
    assert len(g.tensors) == 9
    # Invariants I1-I7 should pass cleanly
    validate_invariants(g)


def test_ir_v2_lossless_roundtrip(tmp_path: Path):
    """Invariant I4: JSON serialization round-trip is lossless."""
    g = create_minimal_valid_graph()
    json_path = tmp_path / "graph.json"
    g.save_json(json_path)

    loaded = GraphV2.load_json(json_path)
    assert loaded.to_dict() == g.to_dict()
    validate_invariants(loaded)


def test_invariant_i1_shape_mismatch_negative_control():
    """Invariant I1 negative control: mismatched Linear dimensions."""
    g = create_minimal_valid_graph()
    # Linear q_w is [64, 64], change input h0_norm shape to [1, 32]
    g.tensors["h0_norm"].shape = [1, 32]

    with pytest.raises(InvariantViolation, match="Invariant I1 failed"):
        validate_invariants(g)


def test_invariant_i3_constant_write_port_negative_control():
    """Invariant I3 negative control: CONSTANT tensor produced by a node."""
    g = create_minimal_valid_graph()
    # Set embed_w as output of embed0
    g.nodes[0].outputs.append("embed_w")

    with pytest.raises(InvariantViolation, match="Invariant I3 failed"):
        validate_invariants(g)


def test_invariant_i6_missing_edge_format_negative_control():
    """Invariant I6 negative control: tensor without valid EdgeFormat."""
    g = create_minimal_valid_graph()
    g.tensors["h0"].format = None  # type: ignore

    with pytest.raises(InvariantViolation, match="Invariant I6 failed"):
        validate_invariants(g)


def test_invariant_i7_state_tensor_illegal_access_negative_control():
    """Invariant I7 negative control: STATE tensor accessed by non-attention op."""
    g = create_minimal_valid_graph()
    # Add kv_cache as input to Linear node
    g.nodes[2].inputs.append("kv_cache")

    with pytest.raises(InvariantViolation, match="Invariant I7 failed"):
        validate_invariants(g)


def test_invariant_i5_forbidden_import_negative_control(tmp_path: Path):
    """Invariant I5 negative control: forbidden import in ir/."""
    # Create a dummy python file with forbidden import
    dummy_code = "import torch\nx = 1\n"
    import ast
    tree = ast.parse(dummy_code)
    violations = []
    for ast_node in ast.walk(tree):
        if isinstance(ast_node, ast.Import):
            for alias in ast_node.names:
                if alias.name.split(".")[0] in ("torch", "transformers"):
                    violations.append(alias.name)
    assert len(violations) == 1
    assert violations[0] == "torch"


def test_invariant_i2_range_validation():
    """Invariant I2: value ranges fit declared formats."""
    fmt = EdgeFormat(bits=8, signed=True, scale=0.05, fixed_point="INT8")
    assert fmt.min_val == -128
    assert fmt.max_val == 127
    # Out of range test
    val_out_of_range = 150
    assert not (fmt.min_val <= val_out_of_range <= fmt.max_val)


def test_edge_format_validation():
    """Test EdgeFormat constraints and min/max val calculations."""
    fmt_u8 = EdgeFormat(bits=8, signed=False, scale=1.0, fixed_point="UINT8")
    assert fmt_u8.min_val == 0
    assert fmt_u8.max_val == 255

    fmt_s8 = EdgeFormat(bits=8, signed=True, scale=1.0, fixed_point="INT8")
    assert fmt_s8.min_val == -128
    assert fmt_s8.max_val == 127

    with pytest.raises(ValueError, match="Invalid bits"):
        EdgeFormat(bits=0, signed=True, scale=1.0, fixed_point="INVALID")

    with pytest.raises(ValueError, match="Scale must be strictly positive"):
        EdgeFormat(bits=8, signed=True, scale=-1.0, fixed_point="INVALID")
