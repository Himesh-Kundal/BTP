"""Hardware-independent and framework-independent tensor references for IR."""

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class TensorKind(str, Enum):
    """Categorization of tensors in the IR."""

    CONSTANT = "CONSTANT"
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    VARIABLE = "VARIABLE"


@dataclass
class TensorRef:
    """Reference to a tensor operand or parameter in the IR.

    Contains metadata (name, shape, dtype, kind, sha256 checksum) and optional
    in-memory array data or an on-disk relative/absolute path to a .npy sidecar.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    kind: TensorKind = TensorKind.CONSTANT
    data: np.ndarray | None = None
    path: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.shape, list):
            self.shape = tuple(self.shape)
        if isinstance(self.kind, str):
            self.kind = TensorKind(self.kind)

        if self.data is not None:
            if not isinstance(self.data, np.ndarray):
                raise TypeError(f"Tensor data must be a np.ndarray, got {type(self.data)}")
            if self.shape != self.data.shape:
                raise ValueError(
                    f"TensorRef shape {self.shape} does not match data shape {self.data.shape}"
                )
            if self.sha256 is None:
                self.sha256 = self.compute_sha256()

    def tobytes(self) -> bytes:
        """Return raw contiguous bytes of in-memory data."""
        if self.data is None:
            raise ValueError("Cannot get bytes of TensorRef without in-memory data")
        return np.ascontiguousarray(self.data).tobytes()

    def compute_sha256(self) -> str:
        """Compute SHA256 hex digest of the raw bytes."""
        return hashlib.sha256(self.tobytes()).hexdigest()

    def __getitem__(self, item: Any) -> Any:
        if self.data is None:
            raise ValueError(f"Cannot index TensorRef '{self.name}' without in-memory data")
        return self.data[item]

    def __array__(self, dtype: Any = None) -> np.ndarray:
        if self.data is None:
            raise ValueError(f"Cannot convert TensorRef '{self.name}' to array without in-memory data")
        if dtype is None:
            return self.data
        return self.data.astype(dtype)

    def save_array(self, directory: Path | str) -> Path:
        """Save tensor data as a .npy file in the specified directory."""
        if self.data is None:
            raise ValueError("Cannot save TensorRef without in-memory data")
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{self.name}.npy"
        np.save(file_path, self.data)
        self.path = str(file_path.name)
        self.sha256 = self.compute_sha256()
        return file_path

    def to_dict(self) -> dict[str, Any]:
        """Serialize TensorRef metadata to a dictionary."""
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "kind": self.kind.value,
            "path": self.path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], base_dir: Path | str | None = None) -> "TensorRef":
        """Deserialize TensorRef from a dictionary, optionally loading array data from sidecar."""
        data: np.ndarray | None = None
        path_str = d.get("path")
        expected_sha = d.get("sha256")

        if path_str is not None and base_dir is not None:
            full_path = Path(base_dir) / path_str
            if full_path.is_file():
                data = np.load(full_path)
                # Verify SHA256 integrity
                actual_sha = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
                if expected_sha is not None and actual_sha != expected_sha:
                    raise ValueError(
                        f"SHA256 mismatch for tensor '{d.get('name')}' in {full_path}: "
                        f"expected {expected_sha}, got {actual_sha}"
                    )

        return cls(
            name=d["name"],
            shape=tuple(d["shape"]),
            dtype=d["dtype"],
            kind=TensorKind(d.get("kind", TensorKind.CONSTANT.value)),
            data=data,
            path=path_str,
            sha256=expected_sha,
        )
