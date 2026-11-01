"""Hardware-independent and framework-independent IR representation for Linear operations."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LinearOp:
    """Represents a dense linear operation Y = X @ W.T + b.

    Adheres to Invariants:
    - I1: Shapes consistent between input_shape, weight, and bias.
    - I2: Quantized weights strictly in [-127, 127] with dtype int8.
    - I5: No torch or framework dependencies.
    """

    name: str
    in_features: int
    out_features: int
    weight: np.ndarray  # int8, shape [N, K]
    bias: np.ndarray | None = None  # int32, shape [N]
    input_shape: tuple[int, ...] = (1, 0)
    weight_scale: np.ndarray | None = None  # float64, shape [N]
    act_scale: float | None = None
    act_dtype: str = "int8"
    acc_dtype: str = "int32"
    acc_bits_min: int = 32
    kind: str = "CONSTANT"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # Default input shape if not explicitly configured with non-zero K
        if self.input_shape == (1, 0):
            self.input_shape = (1, self.in_features)

        # Invariant I2: Type checking
        if not isinstance(self.weight, np.ndarray) or self.weight.dtype != np.int8:
            raise TypeError(
                f"Weight must be a numpy ndarray with dtype int8, got {type(self.weight)} "
                f"with dtype {getattr(self.weight, 'dtype', None)}"
            )

        # Invariant I1: Shape checks
        expected_w_shape = (self.out_features, self.in_features)
        if self.weight.shape != expected_w_shape:
            raise ValueError(
                f"Weight shape {self.weight.shape} does not match "
                f"(out_features={self.out_features}, in_features={self.in_features})"
            )

        if self.bias is not None:
            if not isinstance(self.bias, np.ndarray) or self.bias.dtype != np.int32:
                raise TypeError(f"Bias must be a numpy ndarray with dtype int32, got {type(self.bias)}")
            if self.bias.shape != (self.out_features,):
                raise ValueError(
                    f"bias shape {self.bias.shape} does not match (out_features={self.out_features},)"
                )

        if self.input_shape[-1] != self.in_features:
            raise ValueError(
                f"input_shape feature dimension {self.input_shape[-1]} does not match "
                f"in_features {self.in_features}"
            )

        # Invariant I2: Quantized weight range strictly within [-127, 127]
        # Specifically, -128 is not allowed for weights per rule N2
        if np.any(self.weight < -127) or np.any(self.weight > 127):
            min_val = int(np.min(self.weight))
            max_val = int(np.max(self.weight))
            raise ValueError(
                f"Quantized weights contain values outside valid range [-127, 127]: "
                f"min={min_val}, max={max_val}"
            )
