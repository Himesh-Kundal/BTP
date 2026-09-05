"""Vector writer for testbench stimulation and golden comparison."""

from pathlib import Path

import numpy as np


def write_hex_vectors(
    x: np.ndarray,
    y_exp: np.ndarray,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Write input and expected output hex vector files.

    Per spec docs/02 §7:
    - x.hex: M * K lines, row-major, 2 hex digits per int8 (two's complement)
    - y_exp.hex: M * N lines, row-major, 8 hex digits per int32 (two's complement)

    Args:
        x: Input array of shape [M, K] in int8.
        y_exp: Expected output array of shape [M, N] in int32 or int64.
        out_dir: Directory where hex files will be written.

    Returns:
        (x_path, y_path): Paths to x.hex and y_exp.hex.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    x_arr = np.asarray(x, dtype=np.int8)
    y_arr = np.asarray(y_exp, dtype=np.int64)

    if x_arr.ndim != 2:
        raise ValueError(f"x must be 2D [M, K], got shape {x_arr.shape}")
    if y_arr.ndim != 2:
        raise ValueError(f"y_exp must be 2D [M, N], got shape {y_arr.shape}")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"Vector count mismatch: x has {x_arr.shape[0]}, y has {y_arr.shape[0]}")

    x_file = out_path / "x.hex"
    with open(x_file, "w", encoding="utf-8") as f:
        f.writelines(f"{int(val) & 0xFF:02x}\n" for val in x_arr.flat)

    y_file = out_path / "y_exp.hex"
    with open(y_file, "w", encoding="utf-8") as f:
        f.writelines(f"{int(val) & 0xFFFFFFFF:08x}\n" for val in y_arr.flat)

    return x_file, y_file
