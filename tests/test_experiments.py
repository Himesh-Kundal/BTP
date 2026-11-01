"""Unit tests for synthesis evaluation experiments engine (T7.3, T7.4)."""

from pathlib import Path

import numpy as np

from synth.experiments import (
    generate_all_zero_weights,
    generate_magnitude_matched_weights,
    generate_one_hot_weights,
    generate_shuffled_real_weights,
    generate_uniform_random_weights,
    load_real_quantized_slice,
    run_experiment_e1_e3,
    run_experiment_e2,
)


def test_random_weight_generators() -> None:
    """Verify properties, shapes, and determinism of synthetic weight generators."""
    tr, tc = 8, 16

    # Uniform random
    w_u1 = generate_uniform_random_weights(tr, tc, seed=42)
    w_u2 = generate_uniform_random_weights(tr, tc, seed=42)
    w_u3 = generate_uniform_random_weights(tr, tc, seed=43)
    assert w_u1.shape == (tr, tc)
    assert w_u1.dtype == np.int8
    np.testing.assert_array_equal(w_u1, w_u2)
    assert not np.array_equal(w_u1, w_u3)

    # All zero
    w_z = generate_all_zero_weights(tr, tc)
    assert w_z.shape == (tr, tc)
    assert np.all(w_z == 0)

    # One hot
    w_oh = generate_one_hot_weights(tr, tc, seed=42)
    assert w_oh.shape == (tr, tc)
    for r in range(tr):
        assert np.count_nonzero(w_oh[r]) == 1
        assert np.max(np.abs(w_oh[r])) == 1

    # Magnitude matched
    real_w = np.array([[-10, 20], [0, -5]], dtype=np.int8)
    w_mm = generate_magnitude_matched_weights(real_w, seed=42)
    assert w_mm.shape == real_w.shape
    assert sorted(np.abs(w_mm).ravel()) == sorted(np.abs(real_w).ravel())

    # Shuffled real
    w_sh = generate_shuffled_real_weights(real_w, seed=42)
    assert w_sh.shape == real_w.shape
    assert sorted(w_sh.ravel()) == sorted(real_w.ravel())


def test_real_weight_loader() -> None:
    """Verify loading and quantizing real model slices."""
    w = load_real_quantized_slice(layer=0, proj="q_proj", tr=4, tc=16)
    assert w.shape == (4, 16)
    assert w.dtype == np.int8
    assert np.any(w != 0)


def test_experiment_e1_e3_mini_run(tmp_path: Path) -> None:
    """Verify mini execution of E1 and E3 produces valid CSV outputs."""
    e1_csv = tmp_path / "E1_test.csv"
    e3_csv = tmp_path / "E3_test.csv"

    e1_records, e3_records = run_experiment_e1_e3(
        tile_grid=[(4, 16)],
        real_configs=[("q_proj", 0)],
        seeds=[42, 43],
        target="generic",
        timeout_seconds=60.0,
        out_e1_path=e1_csv,
        out_e3_path=e3_csv,
    )

    assert e1_csv.exists()
    assert e3_csv.exists()
    assert len(e1_records) > 0
    assert len(e3_records) > 0

    # Assert Design F, P, and P_bus are all present in E1
    design_types = {r["design_type"] for r in e1_records}
    assert {"F", "P", "P_bus"}.issubset(design_types)

    # Check cell scaling: F total cells should be significantly lower than P
    f_real = next(r for r in e1_records if r["design_type"] == "F" and "real" in r["weight_source"])
    p_rec = next(r for r in e1_records if r["design_type"] == "P")
    pbus_rec = next(r for r in e1_records if r["design_type"] == "P_bus")

    assert f_real["num_cells_total"] > 0
    assert p_rec["num_cells_total"] > f_real["num_cells_total"]
    assert pbus_rec["num_cells_total"] > f_real["num_cells_total"]
    assert p_rec["num_cells_seq"] > f_real["num_cells_seq"]


def test_experiment_e2_mini_run(tmp_path: Path) -> None:
    """Verify mini execution of E2 produces metrics across all 6 weight classes."""
    e2_csv = tmp_path / "E2_test.csv"

    records = run_experiment_e2(
        tile_size=(4, 16),
        seeds=[42, 43],
        target="generic",
        timeout_seconds=60.0,
        out_e2_path=e2_csv,
    )

    assert e2_csv.exists()
    classes = {r["weight_class"] for r in records}
    expected_classes = {
        "real",
        "all_zero",
        "uniform_random",
        "magnitude_matched",
        "shuffled_real",
        "one_hot",
    }
    assert classes == expected_classes

    # All-zero weights should synthesize to virtually zero combinational logic
    zero_rec = next(r for r in records if r["weight_class"] == "all_zero")
    real_rec = next(r for r in records if r["weight_class"] == "real")
    assert zero_rec["num_cells_total"] < real_rec["num_cells_total"]
