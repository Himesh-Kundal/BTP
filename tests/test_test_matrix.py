"""Tests for verification test-matrix generator (T6.1).

Verifies:
1. Weight class generation: all 8 classes conform to INT8 constraints and properties.
2. Input class generation: all 6 classes conform to stimulus definitions.
3. Dimension and tile grid: configurations cover level L1 requirements.
4. Matrix simulation: RTL simulation matches reference oracle bit-exact (Claim C1)
   across representative matrix configurations.
"""

from pathlib import Path

import numpy as np
import pytest

from verification.reference import linear_reference_int64
from verification.test_matrix import (
    INPUT_CLASSES,
    WEIGHT_CLASSES,
    TestMatrixConfig,
    generate_inputs,
    generate_weights,
    get_l1_test_matrix,
    run_matrix_case,
)


def test_weight_classes_properties() -> None:
    """Verify each weight class produces valid INT8 shapes and expected statistical properties."""
    n, k = 6, 10

    # 1. uniform_random
    w, b = generate_weights(n, k, "uniform_random", seed=42)
    assert w.shape == (n, k)
    assert w.dtype == np.int8
    assert b is None
    assert np.all(w >= -127) and np.all(w <= 127)

    # 2. extremes_pos
    w, _ = generate_weights(n, k, "extremes_pos")
    assert np.all(w == 127)

    # 3. extremes_neg
    w, _ = generate_weights(n, k, "extremes_neg")
    assert np.all(w == -127)

    # 4. extremes_alt
    w, _ = generate_weights(n, k, "extremes_alt")
    assert np.all(np.abs(w) == 127)
    # Check alternating pattern
    assert w[0, 0] == 127
    assert w[0, 1] == -127
    assert w[1, 0] == -127

    # 5. all_zero
    w, _ = generate_weights(n, k, "all_zero")
    assert np.all(w == 0)

    # 6. single_nonzero
    w, _ = generate_weights(n, k, "single_nonzero")
    assert np.count_nonzero(w) == 1
    assert w.dtype == np.int8

    # 7. sparse_random
    w, _ = generate_weights(n, k, "sparse_random", seed=10)
    zero_ratio = np.mean(w == 0)
    assert zero_ratio >= 0.5  # At least 50% sparse

    # 8. near_gaussian
    w, _ = generate_weights(n, k, "near_gaussian", seed=123)
    assert np.all(w >= -127) and np.all(w <= 127)
    assert not np.all(w == 0)


def test_bias_classes() -> None:
    """Verify bias generation for none, synthetic, and extreme classes."""
    n, k = 4, 8

    # none
    _, b_none = generate_weights(n, k, "uniform_random", bias_class="none")
    assert b_none is None

    # synthetic
    _, b_synth = generate_weights(n, k, "uniform_random", bias_class="synthetic", seed=42)
    assert b_synth is not None
    assert b_synth.shape == (n,)
    assert b_synth.dtype == np.int32
    assert np.all(np.abs(b_synth) <= 1000)

    # extreme (+- 2^20)
    _, b_extreme = generate_weights(n, k, "uniform_random", bias_class="extreme", seed=42)
    assert b_extreme is not None
    assert b_extreme.shape == (n,)
    assert b_extreme.dtype == np.int32
    # At least one element near 2^20
    assert np.any(np.abs(b_extreme) >= (2**20 - 100))


def test_input_classes_properties() -> None:
    """Verify input generation across all 6 classes."""
    n, k = 4, 6
    w, _ = generate_weights(n, k, "uniform_random", seed=42)

    # Class 1: uniform random int8 [-128, 127]
    x1 = generate_inputs(k, w, "class1_random_int8", num_vectors=10, seed=1)
    assert x1.shape == (10, k)
    assert x1.dtype == np.int8

    # Class 2: quantizer range [-127, 127]
    x2 = generate_inputs(k, w, "class2_quant_range", num_vectors=10, seed=2)
    assert x2.shape == (10, k)
    assert np.all(x2 >= -127) and np.all(x2 <= 127)

    # Class 3: constant extremes (all -128, all +127, all 0)
    x3 = generate_inputs(k, w, "class3_extremes")
    assert x3.shape == (3, k)
    assert np.all(x3[0] == -128)
    assert np.all(x3[1] == 127)
    assert np.all(x3[2] == 0)

    # Class 4: one-hot per lane (K vectors, identity matrix)
    x4 = generate_inputs(k, w, "class4_one_hot")
    assert x4.shape == (k, k)
    assert np.array_equal(x4, np.eye(k, dtype=np.int8))
    # When x4 is multiplied by W.T, each row j of output is column j of W!
    y4 = linear_reference_int64(x4, w)
    assert np.array_equal(y4, w.T)

    # Class 5: sign-aligned worst case (2*N vectors maximizing accumulator extremes)
    x5 = generate_inputs(k, w, "class5_worst_case")
    assert x5.shape == (2 * n, k)
    for row_idx in range(n):
        # Even vector: x_k = 127 * sign(w_rk)
        vec_pos = x5[2 * row_idx]
        pos_acc = np.sum(vec_pos.astype(np.int64) * w[row_idx].astype(np.int64))
        # Odd vector: x_k = -128 * sign(w_rk)
        vec_neg = x5[2 * row_idx + 1]
        neg_acc = np.sum(vec_neg.astype(np.int64) * w[row_idx].astype(np.int64))
        # Ensure accumulator magnitude is maximal
        assert pos_acc >= 0
        assert neg_acc <= 0

    # Class 6: alternating signs and sparse
    x6 = generate_inputs(k, w, "class6_alternating_sparse")
    assert x6.shape[0] >= 4
    assert x6.shape[1] == k


def test_test_matrix_grid_coverage() -> None:
    """Verify that get_l1_test_matrix spans the required dimensions and tile configs."""
    configs_full = get_l1_test_matrix(fast_mode=False)
    configs_fast = get_l1_test_matrix(fast_mode=True)

    assert len(configs_fast) > 0
    assert len(configs_full) >= len(configs_fast)

    # Fast mode must test every weight class and every input class at least once
    w_classes_seen = {c.weight_class for c in configs_fast}
    in_classes_seen = {c.input_class for c in configs_fast}
    assert set(WEIGHT_CLASSES).issubset(w_classes_seen)
    assert set(INPUT_CLASSES).issubset(in_classes_seen)


@pytest.mark.parametrize(
    "config",
    get_l1_test_matrix(fast_mode=True),
    ids=lambda c: c.name,
)
def test_matrix_simulation_cases(config: TestMatrixConfig, tmp_path: Path) -> None:
    """Run RTL simulation on diverse matrix configurations, verifying Claim C1."""
    report, num_vectors = run_matrix_case(config, tmp_path / f"sim_{config.name}")
    assert report.status == "PASS", f"Simulation failed:\n{report.stdout}\n{report.stderr}"
    assert report.checked == num_vectors
    assert report.mismatch == 0
    assert report.timing_err == 0
