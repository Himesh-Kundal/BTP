"""Tests for full mutation testing hardening (M1-M9) (T6.5, docs/06 §A6).

Verifies:
1. Mutant generator produces all 9 mutant categories (M1-M9).
2. End-to-end mutation campaign executes against full registered wrapper DUT.
3. 100% kill rate (0 survivors) achieved across all 9 categories.
4. Kill matrix JSON recorded with full auditability.
"""

import json
from pathlib import Path

import numpy as np

from backend.module_generator import generate_full_module
from backend.testbench import generate_wrapper_testbench
from backend.tiling import tile_weight_matrix
from backend.vectors import write_hex_vectors
from ir.linear import LinearOp
from verification.mutate import generate_mutants, run_mutation_campaign
from verification.reference import linear_reference_int64


def test_generate_mutants_all_nine_classes() -> None:
    """Verify that generate_mutants produces mutants across all 9 classes (M1–M9)."""
    # Create a 2x4 module with non-trivial weights and multiple tiles
    w = np.array([[12, -7, 0, 15], [-3, 8, 22, -1]], dtype=np.int8)
    op = LinearOp(name="mut_synth", in_features=4, out_features=2, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=2)
    rtl = generate_full_module(op, manifest, latency=1)

    mutants = generate_mutants(rtl)
    assert len(mutants) >= 9

    types_generated = {m.mutant_type for m in mutants}
    expected_types = {"M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"}
    assert expected_types.issubset(types_generated), (
        f"Missing mutant types: {expected_types - types_generated}"
    )

    # Verify every mutant code is distinct from original
    for m in mutants:
        assert m.mutated_code != rtl, f"Mutant {m.id} did not modify RTL!"


def test_mutation_campaign_100_percent_kill_rate(tmp_path: Path) -> None:
    """Run full mutation campaign across M1-M9 on Design F module and assert 100% kill rate."""
    w = np.array([[15, -25, 35, -45], [10, -20, 30, -40]], dtype=np.int8)
    op = LinearOp(name="mut_campaign", in_features=4, out_features=2, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=2)

    # Construct comprehensive test stimulus:
    # 1. Random int8
    # 2. One-hot vectors (4 vectors) - catches M2, M4, M3
    # 3. Extremes (all -128, all +127) - catches M7, M6
    # 4. Sign-aligned worst-case - catches M6
    rng = np.random.default_rng(42)
    x_rand = rng.integers(-128, 128, size=(5, 4), dtype=np.int8)
    x_one_hot = np.eye(4, dtype=np.int8)
    x_ext = np.array(
        [[-128, -128, -128, -128], [127, 127, 127, 127], [-128, 127, -128, 127]], dtype=np.int8
    )
    # Sign aligned for row 0
    x_worst = np.array(
        [
            [127, -128, 127, -128],  # Max pos sum
            [-128, 127, -128, 127],  # Max neg sum
        ],
        dtype=np.int8,
    )

    x = np.vstack([x_rand, x_one_hot, x_ext, x_worst])
    num_vectors = len(x)

    y_exp = linear_reference_int64(x, w)

    work_dir = tmp_path / "campaign"
    work_dir.mkdir(parents=True, exist_ok=True)
    x_hex, y_hex = write_hex_vectors(x, y_exp, work_dir)

    rtl = generate_full_module(op, manifest, latency=1)
    dut_file = work_dir / f"{op.name}.v"
    dut_file.write_text(rtl, encoding="utf-8")

    tb = generate_wrapper_testbench(
        op=op,
        num_vectors=num_vectors,
        x_hex_path=str(x_hex),
        y_hex_path=str(y_hex),
        module_name=f"llmhw_{op.name}",
        manifest=manifest,
        latency=1,
    )
    tb_file = work_dir / f"tb_{op.name}.v"
    tb_file.write_text(tb, encoding="utf-8")

    output_json = tmp_path / "mutation_matrix.json"

    # Execute mutation campaign
    report = run_mutation_campaign(
        dut_path=dut_file,
        tb_path=tb_file,
        num_vectors=num_vectors,
        work_dir=work_dir / "runs",
        rtl_code=rtl,
        matrix_output_path=output_json,
    )

    assert report["total"] > 0
    assert report["survived"] == 0, (
        f"Survivors detected! {report['survived']} survived: {[m for m in report['mutants'] if not m['killed']]}"
    )
    assert report["killed"] == report["total"]
    assert report["kill_rate"] == 1.0

    # Verify all 9 types were tested and killed
    types_tested = {m["type"] for m in report["mutants"]}
    expected_types = {"M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"}
    assert expected_types.issubset(types_tested)

    for mut_type in expected_types:
        type_mutants = [m for m in report["mutants"] if m["type"] == mut_type]
        assert len(type_mutants) > 0, f"No mutants generated for {mut_type}"
        assert all(m["killed"] for m in type_mutants), f"Mutant {mut_type} survived!"

    # Verify JSON was written
    assert output_json.exists()
    with open(output_json) as f:
        data = json.load(f)
    assert data["kill_rate"] == 1.0
    assert data["survived"] == 0
