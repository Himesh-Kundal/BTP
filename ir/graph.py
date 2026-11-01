"""Computation graph representation for the IR."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ir.linear import LinearOp
from ir.operators import BaseOp
from ir.tensor import TensorRef


@dataclass
class Graph:
    """Represents a sequence or computation graph of operators."""

    name: str
    operators: list[BaseOp] = field(default_factory=list)
    inputs: list[TensorRef] = field(default_factory=list)
    outputs: list[TensorRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_operator(self, name: str) -> BaseOp | None:
        """Retrieve an operator by its unique name."""
        for op in self.operators:
            if op.name == name:
                return op
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert graph metadata to serializable dictionary."""
        return {
            "name": self.name,
            "operators": [op.to_dict() for op in self.operators],
            "inputs": [inp.to_dict() for inp in self.inputs],
            "outputs": [out.to_dict() for out in self.outputs],
            "metadata": self.metadata,
        }

    def save(self, directory: Path | str) -> Path:
        """Save graph and its constituent operators to directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save each operator
        for op in self.operators:
            if isinstance(op, LinearOp):
                op.save(dir_path)

        graph_file = dir_path / f"{self.name}_graph.json"
        with open(graph_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return graph_file

    @classmethod
    def load(cls, graph_json: Path | str) -> "Graph":
        """Load graph and constituent operators from file."""
        file_path = Path(graph_json)
        base_dir = file_path.parent
        with open(file_path) as f:
            d = json.load(f)

        operators: list[BaseOp] = []
        for op_dict in d.get("operators", []):
            op_type = op_dict.get("op")
            op_name = op_dict.get("name")
            if op_type == "linear":
                op_json = base_dir / f"{op_name}.json"
                if op_json.is_file():
                    operators.append(LinearOp.load(op_json))
                else:
                    # Construct from dict
                    operators.append(
                        LinearOp(
                            name=op_name,
                            in_features=op_dict["in_features"],
                            out_features=op_dict["out_features"],
                            weight=TensorRef.from_dict(op_dict["weight"], base_dir=base_dir),
                        )
                    )

        return cls(
            name=d["name"],
            operators=operators,
            inputs=[TensorRef.from_dict(inp, base_dir=base_dir) for inp in d.get("inputs", [])],
            outputs=[TensorRef.from_dict(out, base_dir=base_dir) for out in d.get("outputs", [])],
            metadata=d.get("metadata", {}),
        )
