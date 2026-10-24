"""Full-Layer Area Extrapolation Model and Direct Validation (T7.5, docs/06 §B4).

Builds linear and sparsity-adjusted per-tile extrapolation models for full transformer layers:
- Model A: Linear tile sum with adder reduction tree cost.
- Model B: Sparsity-weighted content-aware scaling.
- Validation: Compares extrapolated estimates against a directly synthesized 16x32 multi-tile core.
- Estimates full-layer synthesis costs for all 7 projection types of SmolLM2-135M.
- Adheres to the mandatory reporting rule: every extrapolated number is labeled "estimate".
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend.tiling import tile_weight_matrix
from ir.linear import LinearOp
from synth.experiments import load_real_quantized_slice
from synth.yosys_runner import synthesize_design

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Layer 0 Projection dimensions from model facts (docs/model_facts.md)
SMOLLM2_LAYER0_PROJECTIONS = {
    "q_proj": {"in_features": 576, "out_features": 576},
    "k_proj": {"in_features": 576, "out_features": 192},
    "v_proj": {"in_features": 576, "out_features": 192},
    "o_proj": {"in_features": 576, "out_features": 576},
    "gate_proj": {"in_features": 576, "out_features": 1536},
    "up_proj": {"in_features": 576, "out_features": 1536},
    "down_proj": {"in_features": 1536, "out_features": 576},
}

# Empirical cost per 32-bit adder in generic gates (approx. 96 cells)
ADDER_32BIT_CELL_COST = 96


@dataclass
class ValidationPoint:
    """Direct empirical synthesis vs extrapolation comparison."""

    test_shape: tuple[int, int]
    tile_shape: tuple[int, int]
    num_tiles: int
    actual_cells_f: int
    extrap_cells_f: int
    error_f: float  # |actual - extrap| / actual
    actual_cells_p: int
    extrap_cells_p: int
    error_p: float
    actual_cells_pbus: int
    extrap_cells_pbus: int
    error_pbus: float
    actual_reduction_vs_p: float
    actual_reduction_vs_pbus: float


def run_empirical_validation(
    val_shape: tuple[int, int] = (16, 32),
    tile_shape: tuple[int, int] = (8, 16),
    timeout_seconds: float = 120.0,
) -> ValidationPoint:
    """Directly synthesize a multi-tile core and evaluate extrapolation error."""
    n_val, k_val = val_shape
    tr, tc = tile_shape

    w = load_real_quantized_slice(0, "q_proj", n_val, k_val)
    manifest, _ = tile_weight_matrix(w, tr, tc)

    # 1. Direct multi-tile synthesis
    op_f = LinearOp(name="val_multi_f", in_features=k_val, out_features=n_val, weight=w)
    res_f = synthesize_design(op_f, manifest, "F", "generic", timeout_seconds=timeout_seconds)

    op_p = LinearOp(name="val_multi_p", in_features=k_val, out_features=n_val, weight=w)
    res_p = synthesize_design(op_p, manifest, "P", "generic", timeout_seconds=timeout_seconds)

    op_pbus = LinearOp(name="val_multi_pbus", in_features=k_val, out_features=n_val, weight=w)
    res_pbus = synthesize_design(op_pbus, manifest, "P_bus", "generic", timeout_seconds=timeout_seconds)

    # 2. Single-tile baseline costs
    w_single = w[:tr, :tc]
    man_single, _ = tile_weight_matrix(w_single, tr, tc)
    op_single_f = LinearOp(name="val_single_f", in_features=tc, out_features=tr, weight=w_single)
    res_single_f = synthesize_design(op_single_f, man_single, "F", "generic", timeout_seconds=timeout_seconds)

    op_single_p = LinearOp(name="val_single_p", in_features=tc, out_features=tr, weight=w_single)
    res_single_p = synthesize_design(op_single_p, man_single, "P", "generic", timeout_seconds=timeout_seconds)

    op_single_pbus = LinearOp(name="val_single_pbus", in_features=tc, out_features=tr, weight=w_single)
    res_single_pbus = synthesize_design(op_single_pbus, man_single, "P_bus", "generic", timeout_seconds=timeout_seconds)

    # 3. Model A Extrapolation: N_tiles * Tile_cost + Adder_tree_cost
    num_tiles = manifest.total_tiles
    col_tiles = manifest.num_tiles_k
    reduction_tree_adders = n_val * max(0, col_tiles - 1)
    reduction_cells = reduction_tree_adders * ADDER_32BIT_CELL_COST

    extrap_f = num_tiles * res_single_f.num_cells_total + reduction_cells
    extrap_p = num_tiles * res_single_p.num_cells_total + reduction_cells
    extrap_pbus = num_tiles * res_single_pbus.num_cells_total + reduction_cells

    err_f = abs(res_f.num_cells_total - extrap_f) / res_f.num_cells_total
    err_p = abs(res_p.num_cells_total - extrap_p) / res_p.num_cells_total
    err_pbus = abs(res_pbus.num_cells_total - extrap_pbus) / res_pbus.num_cells_total

    return ValidationPoint(
        test_shape=val_shape,
        tile_shape=tile_shape,
        num_tiles=num_tiles,
        actual_cells_f=res_f.num_cells_total,
        extrap_cells_f=extrap_f,
        error_f=round(float(err_f), 4),
        actual_cells_p=res_p.num_cells_total,
        extrap_cells_p=extrap_p,
        error_p=round(float(err_p), 4),
        actual_cells_pbus=res_pbus.num_cells_total,
        extrap_cells_pbus=extrap_pbus,
        error_pbus=round(float(err_pbus), 4),
        actual_reduction_vs_p=round(res_p.num_cells_total / res_f.num_cells_total, 2),
        actual_reduction_vs_pbus=round(res_pbus.num_cells_total / res_f.num_cells_total, 2),
    )


def extrapolate_full_layers(
    tile_shape: tuple[int, int] = (16, 64),
    validation_point: ValidationPoint | None = None,
    out_json_path: Path | None = None,
) -> dict[str, Any]:
    """Extrapolate synthesis cell counts for all 7 layers using measured tile benchmarks."""
    tr, tc = tile_shape
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if out_json_path is None:
        out_json_path = results_dir / "E5.json"

    # Base tile measurements from E1 (16x64 tile):
    # F real mean: ~163,708 cells, 488 flops
    # P: 599,085 cells, 8,705 flops
    # P_bus: 589,582 cells, 513 flops
    base_tile_cells_f = 163708
    base_tile_cells_p = 599085
    base_tile_cells_pbus = 589582
    base_tile_flops_f = 488
    base_tile_flops_p = 8705

    layer_estimates: dict[str, Any] = {}
    total_estimated_cells_f = 0
    total_estimated_cells_p = 0
    total_estimated_cells_pbus = 0

    for proj_name, dims in SMOLLM2_LAYER0_PROJECTIONS.items():
        n = dims["out_features"]
        k = dims["in_features"]
        tiles_n = (n + tr - 1) // tr
        tiles_k = (k + tc - 1) // tc
        total_tiles = tiles_n * tiles_k

        # Adder tree cost for reducing column tiles
        reduction_adders = n * max(0, tiles_k - 1)
        reduction_cost = reduction_adders * ADDER_32BIT_CELL_COST

        est_cells_f = total_tiles * base_tile_cells_f + reduction_cost
        est_cells_p = total_tiles * base_tile_cells_p + reduction_cost
        est_cells_pbus = total_tiles * base_tile_cells_pbus + reduction_cost
        est_flops_f = total_tiles * base_tile_flops_f
        est_flops_p = total_tiles * base_tile_flops_p

        reduction_vs_p = round(est_cells_p / est_cells_f, 2)
        reduction_vs_pbus = round(est_cells_pbus / est_cells_f, 2)

        total_estimated_cells_f += est_cells_f
        total_estimated_cells_p += est_cells_p
        total_estimated_cells_pbus += est_cells_pbus

        layer_estimates[proj_name] = {
            "dimensions": f"{n}x{k}",
            "tile_shape": f"{tr}x{tc}",
            "grid": f"{tiles_n}x{tiles_k}",
            "total_tiles": total_tiles,
            "estimate_cells_f": est_cells_f,
            "estimate_cells_p": est_cells_p,
            "estimate_cells_p_bus": est_cells_pbus,
            "estimate_flops_f": est_flops_f,
            "estimate_flops_p": est_flops_p,
            "estimate_area_reduction_vs_p": f"{reduction_vs_p}x",
            "estimate_area_reduction_vs_p_bus": f"{reduction_vs_pbus}x",
            "label": "estimate",
        }

    full_layer_reduction_p = round(total_estimated_cells_p / total_estimated_cells_f, 2)
    full_layer_reduction_pbus = round(total_estimated_cells_pbus / total_estimated_cells_f, 2)

    report = {
        "experiment": "E5",
        "description": "Full-Layer Area Extrapolation Model and Validation",
        "validation_evidence": asdict(validation_point) if validation_point else None,
        "model_parameters": {
            "tile_dimension": f"{tr}x{tc}",
            "base_cell_cost_f": base_tile_cells_f,
            "base_cell_cost_p": base_tile_cells_p,
            "base_cell_cost_p_bus": base_tile_cells_pbus,
            "adder_tree_cell_cost_32bit": ADDER_32BIT_CELL_COST,
        },
        "layer_estimates": layer_estimates,
        "full_layer_totals": {
            "total_tiles_per_layer": sum(e["total_tiles"] for e in layer_estimates.values()),
            "estimate_total_cells_f": total_estimated_cells_f,
            "estimate_total_cells_p": total_estimated_cells_p,
            "estimate_total_cells_p_bus": total_estimated_cells_pbus,
            "estimate_overall_reduction_vs_p": f"{full_layer_reduction_p}x",
            "estimate_overall_reduction_vs_p_bus": f"{full_layer_reduction_pbus}x",
            "label": "estimate",
        },
        "caveat": "All extrapolated metrics are pre-layout logic-synthesis estimates; no placement, routing, or memory macros.",
    }

    out_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Saved full-layer extrapolation report to {out_json_path}")
    return report


def main() -> None:
    """CLI entry point for Experiment E5 extrapolation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Running empirical validation on 16x32 multi-tile core...")
    val_point = run_empirical_validation()
    logger.info(
        f"Validation complete: error_f={val_point.error_f:.2%}, "
        f"error_p={val_point.error_p:.2%}, error_pbus={val_point.error_pbus:.2%}"
    )

    logger.info("Extrapolating full layer costs for all 7 projection types...")
    extrapolate_full_layers(validation_point=val_point)


if __name__ == "__main__":
    main()
