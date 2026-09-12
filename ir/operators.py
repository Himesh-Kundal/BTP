"""Base operator definitions for the hardware-independent IR."""

from dataclasses import dataclass, field
from typing import Any

from ir.tensor import TensorRef


@dataclass
class BaseOp:
    """Base class for all intermediate representation operations."""

    name: str
    op: str
    inputs: list[TensorRef] = field(default_factory=list)
    outputs: list[TensorRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert operator to serializable dictionary."""
        return {
            "name": self.name,
            "op": self.op,
            "inputs": [inp.to_dict() for inp in self.inputs],
            "outputs": [out.to_dict() for out in self.outputs],
            "metadata": self.metadata,
        }
