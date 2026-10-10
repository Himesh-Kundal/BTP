"""Test-matrix generator for verification hardening (T6.1, docs/06 §A).

Generates systematic test configurations covering:
1. Level L1 dimensions and tile geometries:
   N in {1, 3, 4, 8, 9}, K in {1, 5, 16, 17, 33}
   Tile grids: (1, 1), (4, 16), (8, 16), (8, 8)
2. Weight classes:
   uniform_random, extremes_pos, extremes_neg, extremes_alt,
   all_zero, single_nonzero, sparse_random, near_gaussian,
   with synthetic or extreme bias.
3. Input classes:
   class1_random_int8, class2_quant_range, class3_extremes,
   class4_one_hot, class5_worst_case, class6_alternating_sparse.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.module_generator import generate_full_module
from backend.testbench import generate_wrapper_testbench
from backend.tiling import pad_activations, tile_weight_matrix
from backend.vectors import write_hex_vectors
from ir.linear import LinearOp
from verification.reference import linear_reference_int64
from verification.simulator import SimResult, run_simulation

WEIGHT_CLASSES = [
    "uniform_random",
    "extremes_pos",
    "extremes_neg",
    "extremes_alt",
    "all_zero",
    "single_nonzero",
    "sparse_random",
    "near_gaussian",
]

INPUT_CLASSES = [
    "class1_random_int8",
    "class2_quant_range",
    "class3_extremes",
    "class4_one_hot",
    "class5_worst_case",
    "class6_alternating_sparse",
]


@dataclass(frozen=True)
class TestMatrixConfig:
    """Configuration for a single verification test-matrix point."""

    __test__ = False

    n: int
    k: int
    tile_n: int
    tile_k: int
    weight_class: str = "uniform_random"
    input_class: str = "class1_random_int8"
    bias_class: str = "none"
    seed: int = 42

    @property
    def name(self) -> str:
        """Unique identifier string for this matrix point."""
        return (
            f"n{self.n}_k{self.k}_t{self.tile_n}x{self.tile_k}_"
            f"{self.weight_class}_{self.input_class}_{self.bias_class}"
        )


def generate_weights(
    n: int,
    k: int,
    weight_class: str = "uniform_random",
    seed: int = 42,
    bias_class: str = "none",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Generate an (N, K) INT8 weight matrix and optional INT32 bias vector.

    Weight classes:
    - 'uniform_random': INT8 in [-127, 127]
    - 'extremes_pos': all +127
    - 'extremes_neg': all -127
    - 'extremes_alt': alternating +-127 pattern
    - 'all_zero': all 0
    - 'single_nonzero': exactly one element is 127 (at row 0, col 0)
    - 'sparse_random': 80% zeros, remainder INT8 [-127, 127]
    - 'near_gaussian': rounded normal distribution clipped to [-127, 127]

    Bias classes:
    - 'none': None
    - 'synthetic': INT32 values in [-1000, 1000]
    - 'extreme': INT32 values near +- (2^20 - 1)
    """
    rng = np.random.default_rng(seed)

    if weight_class == "uniform_random":
        w = rng.integers(-127, 128, size=(n, k), dtype=np.int8)
    elif weight_class == "extremes_pos":
        w = np.full((n, k), 127, dtype=np.int8)
    elif weight_class == "extremes_neg":
        w = np.full((n, k), -127, dtype=np.int8)
    elif weight_class == "extremes_alt":
        rows, cols = np.indices((n, k))
        signs = np.where((rows + cols) % 2 == 0, 127, -127)
        w = signs.astype(np.int8)
    elif weight_class == "all_zero":
        w = np.zeros((n, k), dtype=np.int8)
    elif weight_class == "single_nonzero":
        w = np.zeros((n, k), dtype=np.int8)
        w[0, 0] = 127
    elif weight_class == "sparse_random":
        raw = rng.integers(-127, 128, size=(n, k), dtype=np.int8)
        mask = rng.random(size=(n, k)) < 0.8  # 80% zero
        raw[mask] = 0
        w = raw
    elif weight_class == "near_gaussian":
        samples = rng.normal(loc=0.0, scale=35.0, size=(n, k))
        w = np.clip(np.round(samples), -127, 127).astype(np.int8)
    else:
        raise ValueError(f"Unknown weight_class: {weight_class}")

    bias: np.ndarray | None = None
    if bias_class == "none":
        bias = None
    elif bias_class == "synthetic":
        bias = rng.integers(-1000, 1001, size=(n,), dtype=np.int32)
    elif bias_class == "extreme":
        bias = rng.integers(-(2**20), 2**20, size=(n,), dtype=np.int32)
        if n >= 2:
            bias[0] = 2**20 - 1
            bias[1] = -(2**20 - 1)
        elif n == 1:
            bias[0] = 2**20 - 1
    else:
        raise ValueError(f"Unknown bias_class: {bias_class}")

    return w, bias


def generate_inputs(
    k: int,
    w: np.ndarray,
    input_class: str = "class1_random_int8",
    num_vectors: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Generate (V, K) INT8 input stimulus vectors according to input class.

    Input classes:
    - 'class1_random_int8': uniform random in [-128, 127]
    - 'class2_quant_range': uniform random in [-127, 127]
    - 'class3_extremes': 3 fixed vectors (all -128, all +127, all 0)
    - 'class4_one_hot': K one-hot vectors (identity matrix I_K)
    - 'class5_worst_case': 2*N sign-aligned vectors driving accumulator extremes
    - 'class6_alternating_sparse': alternating signs and 50% sparse vectors
    """
    rng = np.random.default_rng(seed)
    n = w.shape[0]

    if input_class == "class1_random_int8":
        return rng.integers(-128, 128, size=(num_vectors, k), dtype=np.int8)

    elif input_class == "class2_quant_range":
        return rng.integers(-127, 128, size=(num_vectors, k), dtype=np.int8)

    elif input_class == "class3_extremes":
        v_neg = np.full((1, k), -128, dtype=np.int8)
        v_pos = np.full((1, k), 127, dtype=np.int8)
        v_zero = np.zeros((1, k), dtype=np.int8)
        return np.vstack([v_neg, v_pos, v_zero])

    elif input_class == "class4_one_hot":
        # Each row is an elementary basis vector e_i of length K
        return np.eye(k, dtype=np.int8)

    elif input_class == "class5_worst_case":
        # For each row of W, produce:
        # 1. x_k = +127 if w_rk >= 0 else -128 (maximizes positive sum)
        # 2. x_k = -128 if w_rk >= 0 else +127 (maximizes negative sum)
        vectors = []
        for row_idx in range(n):
            w_row = w[row_idx]
            pos_vec = np.where(w_row >= 0, 127, -128).astype(np.int8)
            neg_vec = np.where(w_row >= 0, -128, 127).astype(np.int8)
            vectors.append(pos_vec)
            vectors.append(neg_vec)
        return np.vstack(vectors)

    elif input_class == "class6_alternating_sparse":
        # 1. Alternating +-64
        cols = np.arange(k)
        v_alt = np.where(cols % 2 == 0, 64, -64).astype(np.int8)
        # 2. Inverted alternating
        v_alt_inv = (-v_alt).astype(np.int8)
        # 3. Sparse random vectors
        sparse_vecs = rng.integers(-128, 128, size=(max(2, num_vectors - 2), k), dtype=np.int8)
        mask = rng.random(size=sparse_vecs.shape) < 0.5
        sparse_vecs[mask] = 0
        return np.vstack([v_alt[np.newaxis, :], v_alt_inv[np.newaxis, :], sparse_vecs])

    else:
        raise ValueError(f"Unknown input_class: {input_class}")


def get_l1_test_matrix(fast_mode: bool = True) -> list[TestMatrixConfig]:
    """Return test matrix configurations for Level L1.

    If fast_mode is True, returns a curated representative set covering all
    weight classes, all input classes, corner shapes, and bias modes.
    If fast_mode is False, returns the comprehensive matrix.
    """
    if fast_mode:
        return [
            # 1. 1x1 toy shape, tile 1x1, single nonzero, random int8
            TestMatrixConfig(1, 1, 1, 1, "single_nonzero", "class1_random_int8"),
            # 2. 3x5 non-multiple, tile 4x16, uniform random, one-hot inputs
            TestMatrixConfig(3, 5, 4, 16, "uniform_random", "class4_one_hot"),
            # 3. 4x16 multiple, tile 4x16, extremes_pos, quantizer range
            TestMatrixConfig(4, 16, 4, 16, "extremes_pos", "class2_quant_range"),
            # 4. 8x8 square multiple, tile 8x8, extremes_neg, worst case inputs
            TestMatrixConfig(8, 8, 8, 8, "extremes_neg", "class5_worst_case"),
            # 5. 8x16 rectangular multiple, tile 8x16, extremes_alt, constant extremes
            TestMatrixConfig(8, 16, 8, 16, "extremes_alt", "class3_extremes"),
            # 6. 9x17 non-multiple, tile 8x16, sparse_random, alternating/sparse inputs
            TestMatrixConfig(9, 17, 8, 16, "sparse_random", "class6_alternating_sparse"),
            # 7. 4x8 shape, tile 4x8, all_zero weights, random inputs
            TestMatrixConfig(4, 8, 4, 8, "all_zero", "class1_random_int8"),
            # 8. 4x16 shape, tile 4x16, near_gaussian, random inputs, synthetic bias
            TestMatrixConfig(
                4, 16, 4, 16, "near_gaussian", "class1_random_int8", bias_class="synthetic"
            ),
            # 9. 4x8 shape, tile 4x8, uniform random, quantizer range, extreme bias
            TestMatrixConfig(
                4, 8, 4, 8, "uniform_random", "class2_quant_range", bias_class="extreme"
            ),
            # 10. 9x33 large non-multiple, tile 8x16, uniform random, random inputs
            TestMatrixConfig(9, 33, 8, 16, "uniform_random", "class1_random_int8"),
        ]

    # Comprehensive matrix
    configs: list[TestMatrixConfig] = []
    n_dims = [1, 3, 4, 8, 9]
    k_dims = [1, 5, 16, 17, 33]
    tile_configs = [(1, 1), (4, 16), (8, 16), (8, 8)]

    # Pair each dimension with suitable tile sizes
    for n in n_dims:
        for k in k_dims:
            for tile_n, tile_k in tile_configs:
                # Filter out incompatible 1x1 tile for large matrices (too many tiles)
                if (tile_n == 1 and tile_k == 1) and (n > 4 or k > 8):
                    continue
                # Select cyclic weight and input class
                idx = (n * 31 + k * 17 + tile_n * 7 + tile_k) % len(WEIGHT_CLASSES)
                in_idx = (n * 13 + k * 23) % len(INPUT_CLASSES)
                bias = "synthetic" if (n + k) % 4 == 0 else "none"
                configs.append(
                    TestMatrixConfig(
                        n=n,
                        k=k,
                        tile_n=tile_n,
                        tile_k=tile_k,
                        weight_class=WEIGHT_CLASSES[idx],
                        input_class=INPUT_CLASSES[in_idx],
                        bias_class=bias,
                    )
                )
    return configs


def run_matrix_case(
    config: TestMatrixConfig,
    work_dir: Path,
    num_vectors: int = 5,
) -> tuple[SimResult, int]:
    """Execute end-to-end RTL generation and simulation for a test-matrix point."""
    work_dir.mkdir(parents=True, exist_ok=True)

    w, b = generate_weights(
        n=config.n,
        k=config.k,
        weight_class=config.weight_class,
        seed=config.seed,
        bias_class=config.bias_class,
    )

    x = generate_inputs(
        k=config.k,
        w=w,
        input_class=config.input_class,
        num_vectors=num_vectors,
        seed=config.seed,
    )
    v_actual = x.shape[0]

    y_exp = linear_reference_int64(x, w, b)

    op_name = f"mat_{config.name}"
    op = LinearOp(name=op_name, in_features=config.k, out_features=config.n, weight=w, bias=b)
    manifest, _ = tile_weight_matrix(w, tile_n=config.tile_n, tile_k=config.tile_k)

    # Pad inputs if K != k_padded
    if config.k != manifest.k_padded:
        x_stim, _ = pad_activations(x, manifest.tile_k)
    else:
        x_stim = x

    x_hex, y_hex = write_hex_vectors(x_stim, y_exp, work_dir)

    rtl_code = generate_full_module(op, manifest)
    dut_file = work_dir / f"{op_name}.v"
    dut_file.write_text(rtl_code, encoding="utf-8")

    tb_code = generate_wrapper_testbench(
        op=op,
        num_vectors=v_actual,
        x_hex_path=str(x_hex),
        y_hex_path=str(y_hex),
        module_name=f"llmhw_{op_name}",
        manifest=manifest,
    )
    tb_file = work_dir / f"tb_{op_name}.v"
    tb_file.write_text(tb_code, encoding="utf-8")

    report = run_simulation(
        dut_path=dut_file,
        tb_path=tb_file,
        expected_checked=v_actual,
        work_dir=work_dir,
    )
    return report, v_actual
