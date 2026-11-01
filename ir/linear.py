"""Hardware-independent and framework-independent IR representation for Linear operations."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ir.operators import BaseOp
from ir.tensor import TensorKind, TensorRef


@dataclass
class LinearOp(BaseOp):
    """Represents a dense linear operation Y = X @ W.T + b.

    Adheres to Invariants:
    - I1: Shapes consistent between input_shape, weight, bias, and weight_scale.
    - I2: Quantized weights strictly in [-127, 127] with dtype int8.
    - I3: Weight has kind CONSTANT.
    - I4: Serialization round-trip is lossless.
    - I5: No torch or framework dependencies.
    """

    in_features: int = 0
    out_features: int = 0
    weight: TensorRef = field(default_factory=lambda: TensorRef("w", (0, 0), "int8"))
    bias: TensorRef | None = None
    input_shape: tuple[int, ...] = (1, 0)
    weight_scale: np.ndarray | None = None  # float64, shape [N]
    act_scale: float | None = None
    act_dtype: str = "int8"
    acc_dtype: str = "int32"
    acc_bits_min: int | None = 32
    kind: str = "CONSTANT"
    quant: dict[str, Any] = field(default_factory=dict)
    hw: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    is_quantized: bool = True

    def __init__(
        self,
        name: str,
        in_features: int,
        out_features: int,
        weight: TensorRef | np.ndarray,
        bias: TensorRef | np.ndarray | None = None,
        input_shape: tuple[int, ...] = (1, 0),
        weight_scale: np.ndarray | None = None,
        act_scale: float | None = None,
        act_dtype: str = "int8",
        acc_dtype: str = "int32",
        acc_bits_min: int | None = 32,
        kind: str = "CONSTANT",
        metadata: dict[str, Any] | None = None,
        quant: dict[str, Any] | None = None,
        hw: dict[str, Any] | None = None,
        source: dict[str, Any] | None = None,
        is_quantized: bool | None = None,
    ) -> None:
        super().__init__(name=name, op="linear", metadata=metadata or {})
        self.in_features = in_features
        self.out_features = out_features
        self.kind = kind
        self.input_shape = input_shape
        self.weight_scale = weight_scale
        self.act_scale = act_scale
        self.act_dtype = act_dtype
        self.acc_dtype = acc_dtype
        self.acc_bits_min = acc_bits_min
        self.quant = quant or {}
        self.hw = hw or {}
        self.source = source or {}

        # Convert numpy array weight to TensorRef if needed
        if isinstance(weight, np.ndarray):
            self.weight = TensorRef(
                name=f"{name}_w",
                shape=weight.shape,
                dtype=str(weight.dtype),
                kind=TensorKind(kind),
                data=weight,
            )
        elif isinstance(weight, TensorRef):
            self.weight = weight
        else:
            raise TypeError(
                f"Weight must be a numpy ndarray or TensorRef, got {type(weight)}"
            )

        # Convert numpy array bias to TensorRef if needed
        if isinstance(bias, np.ndarray):
            self.bias = TensorRef(
                name=f"{name}_b",
                shape=bias.shape,
                dtype=str(bias.dtype),
                kind=TensorKind.CONSTANT,
                data=bias,
            )
        elif isinstance(bias, TensorRef):
            self.bias = bias
        else:
            self.bias = None

        # Determine quantization status (defaults to True per Invariant I2)
        if is_quantized is not None:
            self.is_quantized = is_quantized
        else:
            self.is_quantized = True

        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """Validate Invariants I1, I2, I3."""
        # Default input shape if not configured
        if self.input_shape == (1, 0):
            self.input_shape = (1, self.in_features)

        # Invariant I1: Shape checks
        expected_w_shape = (self.out_features, self.in_features)
        if self.weight.shape != expected_w_shape:
            raise ValueError(
                f"Weight shape {self.weight.shape} does not match "
                f"(out_features={self.out_features}, in_features={self.in_features})"
            )

        if self.bias is not None and self.bias.shape != (self.out_features,):
            raise ValueError(
                f"bias shape {self.bias.shape} does not match (out_features={self.out_features},)"
            )

        if self.weight_scale is not None and self.weight_scale.shape != (self.out_features,):
            raise ValueError(
                f"weight_scale shape {self.weight_scale.shape} does not match "
                f"(out_features={self.out_features},)"
            )

        if self.input_shape[-1] != self.in_features:
            raise ValueError(
                f"input_shape feature dimension {self.input_shape[-1]} does not match "
                f"in_features {self.in_features}"
            )

        # Invariant I2: Quantized weight checks
        if self.is_quantized:
            if self.weight.dtype != "int8":
                raise TypeError(
                    f"Quantized weight must have dtype int8, got {self.weight.dtype}"
                )
            if self.weight.data is not None and (
                np.any(self.weight.data < -127) or np.any(self.weight.data > 127)
            ):
                min_val = int(np.min(self.weight.data))
                max_val = int(np.max(self.weight.data))
                raise ValueError(
                    f"Quantized weights contain values outside valid range [-127, 127]: "
                    f"min={min_val}, max={max_val}"
                )
            if self.bias is not None and self.bias.dtype != "int32":
                raise TypeError(f"Quantized bias must have dtype int32, got {self.bias.dtype}")
        else:
            # Unquantized weights (e.g. from artifact before quantization)
            if self.weight.dtype not in ("float32", "float64"):
                raise TypeError(
                    f"Unquantized weight must have float32/float64 dtype, got {self.weight.dtype}"
                )

        # Invariant I3: Fixed-weight must be marked CONSTANT
        if self.hw.get("arch") == "fixed" and self.weight.kind != TensorKind.CONSTANT:
            raise ValueError("Design F requires weight tensor kind to be CONSTANT")

    def to_dict(self) -> dict[str, Any]:
        """Serialize LinearOp metadata to dictionary."""
        d: dict[str, Any] = {
            "name": self.name,
            "op": self.op,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "input_shape": list(self.input_shape),
            "weight": self.weight.to_dict(),
            "bias": self.bias.to_dict() if self.bias is not None else None,
            "act_scale": self.act_scale,
            "act_dtype": self.act_dtype,
            "acc_dtype": self.acc_dtype,
            "acc_bits_min": self.acc_bits_min,
            "is_quantized": self.is_quantized,
            "kind": self.kind,
            "quant": self.quant,
            "hw": self.hw,
            "source": self.source,
        }
        if self.weight_scale is not None:
            d["weight_scale"] = {
                "shape": list(self.weight_scale.shape),
                "dtype": str(self.weight_scale.dtype),
                "sha256": hashlib.sha256(np.ascontiguousarray(self.weight_scale).tobytes()).hexdigest(),
                "path": f"{self.name}_weight_scale.npy",
            }
        return d

    def save(self, directory: Path | str) -> Path:
        """Save LinearOp JSON and sidecar arrays to directory (Invariant I4)."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save weight and bias arrays
        self.weight.save_array(dir_path)
        if self.bias is not None:
            self.bias.save_array(dir_path)

        if self.weight_scale is not None:
            ws_path = dir_path / f"{self.name}_weight_scale.npy"
            np.save(ws_path, self.weight_scale)

        json_path = dir_path / f"{self.name}.json"
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return json_path

    @classmethod
    def load(cls, json_path: Path | str) -> "LinearOp":
        """Load LinearOp from JSON and sidecar arrays (Invariant I4)."""
        path = Path(json_path)
        base_dir = path.parent
        with open(path) as f:
            d = json.load(f)

        weight_ref = TensorRef.from_dict(d["weight"], base_dir=base_dir)
        bias_ref = TensorRef.from_dict(d["bias"], base_dir=base_dir) if d.get("bias") else None

        weight_scale: np.ndarray | None = None
        if "weight_scale" in d and d["weight_scale"] is not None:
            ws_meta = d["weight_scale"]
            ws_file = base_dir / ws_meta["path"]
            if ws_file.is_file():
                weight_scale = np.load(ws_file)
                actual_sha = hashlib.sha256(np.ascontiguousarray(weight_scale).tobytes()).hexdigest()
                if actual_sha != ws_meta["sha256"]:
                    raise ValueError(f"SHA256 mismatch for weight_scale in {ws_file}")

        return cls(
            name=d["name"],
            in_features=d["in_features"],
            out_features=d["out_features"],
            weight=weight_ref,
            bias=bias_ref,
            input_shape=tuple(d["input_shape"]),
            weight_scale=weight_scale,
            act_scale=d.get("act_scale"),
            act_dtype=d.get("act_dtype", "int8"),
            acc_dtype=d.get("acc_dtype", "int32"),
            acc_bits_min=d.get("acc_bits_min"),
            kind=d.get("kind", "CONSTANT"),
            quant=d.get("quant", {}),
            hw=d.get("hw", {}),
            source=d.get("source", {}),
            is_quantized=d.get("is_quantized", True),
        )


def load_linear_from_artifact(artifact_dir: Path | str, name: str | None = None) -> LinearOp:
    """Load an unquantized LinearOp directly from dumped frontend artifacts."""
    art_path = Path(artifact_dir)
    meta_path = art_path / "meta.json"
    weight_path = art_path / "weight.npy"

    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found in {art_path}")
    if not weight_path.is_file():
        raise FileNotFoundError(f"weight.npy not found in {art_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    weight_data = np.load(weight_path)
    bias_data: np.ndarray | None = None
    bias_path = art_path / "bias.npy"
    if bias_path.is_file():
        bias_data = np.load(bias_path)

    op_name = name or art_path.name
    out_features, in_features = weight_data.shape

    weight_ref = TensorRef(
        name=f"{op_name}_weight",
        shape=weight_data.shape,
        dtype=str(weight_data.dtype),
        kind=TensorKind.CONSTANT,
        data=weight_data,
        path="weight.npy",
        sha256=meta.get("weight_sha256"),
    )

    bias_ref: TensorRef | None = None
    if bias_data is not None:
        bias_ref = TensorRef(
            name=f"{op_name}_bias",
            shape=bias_data.shape,
            dtype=str(bias_data.dtype),
            kind=TensorKind.CONSTANT,
            data=bias_data,
            path="bias.npy",
        )

    return LinearOp(
        name=op_name,
        in_features=in_features,
        out_features=out_features,
        weight=weight_ref,
        bias=bias_ref,
        is_quantized=False,
        source=meta,
    )
