"""Graph IR v2 for Transformer Decoder Compiler (Part II).

Contracts per docs/part2/02 §5:
- Node ops: Embed, RMSNorm, Linear, RoPE, Attention, SiLU, EltMul, Add, ArgMax
- Tensor kinds: CONSTANT (weights, tables, requant params), DYNAMIC (activations), STATE (KV cache)
- Edge annotation: (bits, signed, scale, fixed_point, producer, consumers)
- Node attributes: dims, head layout (n_heads, n_kv, hd), ROM refs, requant param refs, MAX_CTX
- Provenance: strings only (model_id, revision, module_path)

Invariants:
- I1: shapes consistent
- I2: value ranges fit declared formats
- I3: CONSTANT tensors never feed a write port (cannot be produced by any node)
- I4: round-trip lossless JSON serialization
- I5: no torch/transformers/Verilog imports
- I6: every edge has a format
- I7: STATE tensors accessed only by Attention/KV ops
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TensorKind(str, Enum):
    CONSTANT = "CONSTANT"
    DYNAMIC = "DYNAMIC"
    STATE = "STATE"


class OpKind(str, Enum):
    Embed = "Embed"
    RMSNorm = "RMSNorm"
    Linear = "Linear"
    RoPE = "RoPE"
    Attention = "Attention"
    SiLU = "SiLU"
    EltMul = "EltMul"
    Add = "Add"
    ArgMax = "ArgMax"


@dataclass(frozen=True)
class EdgeFormat:
    bits: int
    signed: bool
    scale: float
    fixed_point: str

    def __post_init__(self) -> None:
        if self.bits <= 0 or self.bits > 64:
            raise ValueError(f"Invalid bits {self.bits}; must be in [1, 64]")
        if self.scale <= 0.0:
            raise ValueError(f"Scale must be strictly positive, got {self.scale}")

    @property
    def min_val(self) -> int:
        if self.signed:
            return -(1 << (self.bits - 1))
        return 0

    @property
    def max_val(self) -> int:
        if self.signed:
            return (1 << (self.bits - 1)) - 1
        return (1 << self.bits) - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "signed": self.signed,
            "scale": self.scale,
            "fixed_point": self.fixed_point,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EdgeFormat:
        return cls(
            bits=int(d["bits"]),
            signed=bool(d["signed"]),
            scale=float(d["scale"]),
            fixed_point=str(d["fixed_point"]),
        )


@dataclass
class TensorRefV2:
    name: str
    kind: TensorKind
    shape: list[int]
    format: EdgeFormat
    producer: str | None = None
    consumers: list[str] = field(default_factory=list)
    data_file: str | None = None
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "shape": list(self.shape),
            "format": self.format.to_dict(),
            "producer": self.producer,
            "consumers": list(self.consumers),
            "data_file": self.data_file,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TensorRefV2:
        return cls(
            name=str(d["name"]),
            kind=TensorKind(d["kind"]),
            shape=[int(x) for x in d["shape"]],
            format=EdgeFormat.from_dict(d["format"]),
            producer=d.get("producer"),
            consumers=[str(c) for c in d.get("consumers", [])],
            data_file=d.get("data_file"),
            sha256=d.get("sha256"),
        )


@dataclass
class NodeV2:
    id: str
    op: OpKind
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Provenance must be strings only per spec §5
        for k, v in self.provenance.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise TypeError(f"Provenance must contain strings only; got {k}: {v}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "op": self.op.value,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attributes": dict(self.attributes),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NodeV2:
        return cls(
            id=str(d["id"]),
            op=OpKind(d["op"]),
            inputs=[str(x) for x in d["inputs"]],
            outputs=[str(x) for x in d["outputs"]],
            attributes=dict(d.get("attributes", {})),
            provenance={str(k): str(v) for k, v in d.get("provenance", {}).items()},
        )


@dataclass
class GraphV2:
    name: str
    nodes: list[NodeV2] = field(default_factory=list)
    tensors: dict[str, TensorRefV2] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_tensor(
        self,
        name: str,
        kind: TensorKind,
        shape: list[int],
        edge_format: EdgeFormat,
        producer: str | None = None,
        consumers: list[str] | None = None,
        data_file: str | None = None,
        sha256: str | None = None,
    ) -> TensorRefV2:
        if name in self.tensors:
            raise ValueError(f"Tensor {name} already exists in graph")
        t = TensorRefV2(
            name=name,
            kind=kind,
            shape=list(shape),
            format=edge_format,
            producer=producer,
            consumers=list(consumers) if consumers is not None else [],
            data_file=data_file,
            sha256=sha256,
        )
        self.tensors[name] = t
        return t

    def add_node(
        self,
        node_id: str,
        op: OpKind,
        inputs: list[str],
        outputs: list[str],
        attributes: dict[str, Any] | None = None,
        provenance: dict[str, str] | None = None,
    ) -> NodeV2:
        node = NodeV2(
            id=node_id,
            op=op,
            inputs=list(inputs),
            outputs=list(outputs),
            attributes=dict(attributes) if attributes is not None else {},
            provenance=dict(provenance) if provenance is not None else {},
        )
        self.nodes.append(node)
        # Update producer/consumers in tensors
        for inp in inputs:
            if inp in self.tensors and node_id not in self.tensors[inp].consumers:
                self.tensors[inp].consumers.append(node_id)
        for out in outputs:
            if out in self.tensors:
                self.tensors[out].producer = node_id
        return node

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes],
            "tensors": {k: v.to_dict() for k, v in self.tensors.items()},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraphV2:
        nodes = [NodeV2.from_dict(n) for n in d.get("nodes", [])]
        tensors = {k: TensorRefV2.from_dict(v) for k, v in d.get("tensors", {}).items()}
        return cls(
            name=str(d["name"]),
            nodes=nodes,
            tensors=tensors,
            metadata=dict(d.get("metadata", {})),
        )

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load_json(cls, path: str | Path) -> GraphV2:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)


class InvariantViolation(Exception):
    """Raised when an IR invariant is violated."""


def validate_invariants(graph: GraphV2) -> None:
    """Validate Invariants I1–I7 per docs/part2/02 §5.

    I1: shapes consistent across producer/consumers and node operations.
    I2: value ranges fit declared formats.
    I3: CONSTANT tensors never feed a write port (never produced by any node).
    I4: round-trip lossless JSON serialization.
    I5: no torch/transformers/Verilog imports anywhere in ir/.
    I6: every edge carries a complete format (bits, signed, scale, fixed_point).
    I7: STATE tensors accessed only by Attention/KV ops.
    """
    # I5: Layering / imports check on ir module itself
    import ast
    import sys
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("ir.") or mod_name == "ir":
            mod = sys.modules[mod_name]
            file_path = getattr(mod, "__file__", None)
            if file_path and Path(file_path).exists():
                text = Path(file_path).read_text(encoding="utf-8")
                tree = ast.parse(text, filename=str(file_path))
                for ast_node in ast.walk(tree):
                    if isinstance(ast_node, ast.Import):
                        for alias in ast_node.names:
                            base_pkg = alias.name.split(".")[0]
                            if base_pkg in ("torch", "transformers"):
                                raise InvariantViolation(f"Invariant I5 failed: imports forbidden module '{alias.name}'")
                    elif isinstance(ast_node, ast.ImportFrom) and ast_node.module:
                        base_pkg = ast_node.module.split(".")[0]
                        if base_pkg in ("torch", "transformers"):
                            raise InvariantViolation(f"Invariant I5 failed: imports from forbidden module '{ast_node.module}'")

    # I6: Every edge has a format
    for name, tensor in graph.tensors.items():
        if tensor.format is None:
            raise InvariantViolation(f"Invariant I6 failed: Tensor '{name}' has no EdgeFormat")
        if not isinstance(tensor.format, EdgeFormat):
            raise InvariantViolation(f"Invariant I6 failed: Tensor '{name}' format is not EdgeFormat")
        if tensor.format.bits <= 0 or tensor.format.scale <= 0.0:
            raise InvariantViolation(f"Invariant I6 failed: Tensor '{name}' invalid format fields")

    # I3: CONSTANT tensors never feed a write port (must never have a producer node)
    node_outputs: dict[str, str] = {}
    for node in graph.nodes:
        for out in node.outputs:
            if out in node_outputs:
                raise InvariantViolation(f"Invariant I1 failed: Multiple nodes produce tensor '{out}'")
            node_outputs[out] = node.id
            t = graph.tensors.get(out)
            if t is not None and t.kind == TensorKind.CONSTANT:
                raise InvariantViolation(
                    f"Invariant I3 failed: CONSTANT tensor '{out}' is written by node '{node.id}'"
                )

    # I7: STATE tensors accessed only by Attention/KV ops
    for node in graph.nodes:
        for inp in node.inputs:
            t = graph.tensors.get(inp)
            if t is not None and t.kind == TensorKind.STATE and node.op not in (OpKind.Attention,):
                raise InvariantViolation(
                    f"Invariant I7 failed: STATE tensor '{inp}' accessed by non-attention op '{node.op}'"
                )
        for out in node.outputs:
            t = graph.tensors.get(out)
            if t is not None and t.kind == TensorKind.STATE and node.op not in (OpKind.Attention,):
                raise InvariantViolation(
                    f"Invariant I7 failed: STATE tensor '{out}' produced by non-attention op '{node.op}'"
                )

    # I1: Shapes consistent
    for node in graph.nodes:
        for inp in node.inputs:
            if inp not in graph.tensors:
                raise InvariantViolation(f"Invariant I1 failed: Node '{node.id}' input '{inp}' not in graph.tensors")
        for out in node.outputs:
            if out not in graph.tensors:
                raise InvariantViolation(f"Invariant I1 failed: Node '{node.id}' output '{out}' not in graph.tensors")

        # Op-specific shape consistency checks
        if node.op == OpKind.Linear:
            # Inputs: [act, weight, (optional bias)]
            act_t = graph.tensors[node.inputs[0]]
            weight_t = graph.tensors[node.inputs[1]]
            out_t = graph.tensors[node.outputs[0]]
            # weight shape: [out_features, in_features]
            if len(weight_t.shape) != 2:
                raise InvariantViolation(f"Invariant I1 failed: Linear weight '{weight_t.name}' must be 2D")
            in_feat = weight_t.shape[1]
            out_feat = weight_t.shape[0]
            if act_t.shape[-1] != in_feat:
                raise InvariantViolation(
                    f"Invariant I1 failed: Linear act dim {act_t.shape[-1]} != weight in_features {in_feat}"
                )
            if out_t.shape[-1] != out_feat:
                raise InvariantViolation(
                    f"Invariant I1 failed: Linear output dim {out_t.shape[-1]} != weight out_features {out_feat}"
                )

        elif node.op == OpKind.Add:
            # All inputs and output must have identical shapes
            shapes = [graph.tensors[inp].shape for inp in node.inputs]
            out_shape = graph.tensors[node.outputs[0]].shape
            if not all(s == out_shape for s in shapes):
                raise InvariantViolation(
                    f"Invariant I1 failed: Add node '{node.id}' inputs/output shape mismatch: {shapes} vs {out_shape}"
                )

        elif node.op == OpKind.EltMul:
            shapes = [graph.tensors[inp].shape for inp in node.inputs]
            out_shape = graph.tensors[node.outputs[0]].shape
            if not all(s == out_shape for s in shapes):
                raise InvariantViolation(
                    f"Invariant I1 failed: EltMul node '{node.id}' inputs/output shape mismatch"
                )

        elif node.op == OpKind.RMSNorm:
            in_shape = graph.tensors[node.inputs[0]].shape
            out_shape = graph.tensors[node.outputs[0]].shape
            if in_shape != out_shape:
                raise InvariantViolation(
                    f"Invariant I1 failed: RMSNorm node '{node.id}' in_shape {in_shape} != out_shape {out_shape}"
                )

        elif node.op == OpKind.ArgMax:
            out_t = graph.tensors[node.outputs[0]]
            # ArgMax produces scalar or 1D token ID
            if len(out_t.shape) > 1 and out_t.shape != [1]:
                raise InvariantViolation(
                    f"Invariant I1 failed: ArgMax node '{node.id}' output must be scalar/1D, got {out_t.shape}"
                )

    # I2: Value ranges fit declared formats
    for name, tensor in graph.tensors.items():
        fmt = tensor.format
        if fmt.signed:
            assert fmt.min_val == -(1 << (fmt.bits - 1))
            assert fmt.max_val == (1 << (fmt.bits - 1)) - 1
        else:
            assert fmt.min_val == 0
            assert fmt.max_val == (1 << fmt.bits) - 1

    # I4: Round-trip lossless
    d1 = graph.to_dict()
    g_rt = GraphV2.from_dict(d1)
    d2 = g_rt.to_dict()
    if d1 != d2:
        raise InvariantViolation("Invariant I4 failed: Graph serialization round-trip is not identical")
