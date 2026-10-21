"""Synthesis Evaluation Experiments Runner (T7.3, T7.4, docs/06 §B4).

Implements logic-synthesis experiments comparing Design F (fixed constant weights)
against Design P (storage-inclusive baseline) and Design P_bus (logic-only baseline):
- Experiment E1: Area scaling across tile dimensions (4x16, 8x16, 16x16, 16x64, 32x64).
- Experiment E3: Longest topological path (ltp) depth proxy.
- Experiment E2: Weight content & distribution effect (real, uniform random,
  magnitude-matched, shuffled-real, all-zero, one-hot).
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from backend.tiling import tile_weight_matrix
from compiler.quantization import quantize_weights
from ir.linear import LinearOp
from synth.yosys_runner import SynthesisResult, synthesize_design

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_GRID = [(4, 16), (8, 16), (16, 16), (16, 64), (32, 64)]
DEFAULT_REAL_CONFIGS = [("q_proj", 0), ("gate_proj", 0), ("down_proj", 0)]


def find_model_artifact_dir(layer: int, proj: str) -> Path:
    """Dynamically locate layer/proj artifacts under artifacts/ without hardcoded model strings."""
    art_root = REPO_ROOT / "artifacts"
    if not art_root.exists():
        raise FileNotFoundError(f"Artifact directory '{art_root}' not found.")

    # Search for any model subdirectory containing layer_<layer>/<proj>
    for model_dir in art_root.iterdir():
        if model_dir.is_dir() and not model_dir.name.startswith("."):
            target_dir = model_dir / f"layer_{layer}" / proj
            if target_dir.exists():
                return target_dir

    raise FileNotFoundError(f"Could not locate artifacts for layer {layer} projection {proj} in {art_root}")


def load_real_quantized_slice(layer: int, proj: str, tr: int, tc: int) -> np.ndarray:
    """Load float weights from artifact directory, quantize to INT8 (Rule N2), and return [tr, tc] slice."""
    art_dir = find_model_artifact_dir(layer, proj)
    weight_path = art_dir / "weight.npy"
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found at {weight_path}")

    w_float = np.load(weight_path)
    w_q, _ = quantize_weights(w_float)
    return np.ascontiguousarray(w_q[:tr, :tc], dtype=np.int8)


def generate_uniform_random_weights(tr: int, tc: int, seed: int) -> np.ndarray:
    """Generate uniform random INT8 weights in [-127, 127] with specified seed."""
    rng = np.random.default_rng(seed)
    return rng.integers(-127, 128, size=(tr, tc), dtype=np.int8)


def generate_magnitude_matched_weights(real_w: np.ndarray, seed: int) -> np.ndarray:
    """Randomize signs and spatial layout of weights while strictly preserving magnitude histogram."""
    rng = np.random.default_rng(seed)
    flat_magnitudes = np.abs(real_w).ravel()
    rng.shuffle(flat_magnitudes)
    random_signs = rng.choice([-1, 1], size=flat_magnitudes.shape)
    matched = (flat_magnitudes * random_signs).reshape(real_w.shape)
    return matched.astype(np.int8)


def generate_shuffled_real_weights(real_w: np.ndarray, seed: int) -> np.ndarray:
    """Randomly permute (n, k) positions of the exact real weight values."""
    rng = np.random.default_rng(seed)
    flat = real_w.ravel().copy()
    rng.shuffle(flat)
    return flat.reshape(real_w.shape).astype(np.int8)


def generate_all_zero_weights(tr: int, tc: int) -> np.ndarray:
    """Generate all-zero weights (extreme sparsity baseline)."""
    return np.zeros((tr, tc), dtype=np.int8)


def generate_one_hot_weights(tr: int, tc: int, seed: int) -> np.ndarray:
    """Generate one-hot weights (single nonzero coefficient +1 or -1 per row)."""
    rng = np.random.default_rng(seed)
    w = np.zeros((tr, tc), dtype=np.int8)
    for i in range(tr):
        col = rng.integers(0, tc)
        val = rng.choice([-1, 1])
        w[i, col] = val
    return w


def run_single_synthesis(
    op: LinearOp,
    design_type: str,
    target: str = "generic",
    timeout_seconds: float = 300.0,
) -> SynthesisResult:
    """Run synthesis for a single design configuration and enforce width checks."""
    weight_data = (
        op.weight.data
        if hasattr(op.weight, "data") and isinstance(op.weight.data, np.ndarray)
        else op.weight
    )
    manifest, _ = tile_weight_matrix(weight_data, op.out_features, op.in_features)
    res = synthesize_design(
        op=op,
        manifest=manifest,
        design_type=design_type,
        target=target,
        timeout_seconds=timeout_seconds,
    )
    return res


def _run_single_task(
    tr: int,
    tc: int,
    op: LinearOp,
    design_type: str,
    weight_source: str,
    target: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Worker task executing synthesis and capturing metrics or timeout."""
    try:
        res = run_single_synthesis(
            op=op,
            design_type=design_type,
            target=target,
            timeout_seconds=timeout_seconds,
        )
        return {
            "tile_rows": tr,
            "tile_cols": tc,
            "design_type": design_type,
            "weight_source": weight_source,
            "target": target,
            "result": res,
            "status": "OK" if res.returncode == 0 else "ERROR",
        }
    except (TimeoutError, RuntimeError, ValueError, OSError) as e:
        logger.warning(f"Synthesis failed/timed out for {design_type} {weight_source} ({tr}x{tc}): {e}")
        return {
            "tile_rows": tr,
            "tile_cols": tc,
            "design_type": design_type,
            "weight_source": weight_source,
            "target": target,
            "result": None,
            "status": "TIMEOUT",
            "error": str(e),
        }


def run_experiment_e1_e3(
    tile_grid: list[tuple[int, int]] | None = None,
    real_configs: list[tuple[str, int]] | None = None,
    seeds: list[int] | None = None,
    target: str = "generic",
    timeout_seconds: float = 300.0,
    max_workers: int = 8,
    out_e1_path: Path | None = None,
    out_e3_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute Experiments E1 (Area Scaling) and E3 (Topological Depth Proxy) concurrently."""
    if tile_grid is None:
        tile_grid = DEFAULT_GRID
    if real_configs is None:
        real_configs = DEFAULT_REAL_CONFIGS
    if seeds is None:
        seeds = DEFAULT_SEEDS

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if out_e1_path is None:
        out_e1_path = results_dir / "E1.csv"
    if out_e3_path is None:
        out_e3_path = results_dir / "E3.csv"

    from concurrent.futures import ThreadPoolExecutor

    e1_records: list[dict[str, Any]] = []
    e3_records: list[dict[str, Any]] = []

    for tr, tc in tile_grid:
        logger.info(f"--- Running E1/E3 on Tile ({tr}, {tc}) with {max_workers} workers ---")

        tasks = []
        op_dummy = LinearOp(
            name=f"tile_{tr}x{tc}",
            in_features=tc,
            out_features=tr,
            weight=np.zeros((tr, tc), dtype=np.int8),
        )

        # Baseline P and P_bus
        tasks.append((tr, tc, op_dummy, "P", "independent", target, timeout_seconds))
        tasks.append((tr, tc, op_dummy, "P_bus", "independent", target, timeout_seconds))

        # Real weights
        for proj, layer in real_configs:
            w_real = load_real_quantized_slice(layer, proj, tr, tc)
            op_real = LinearOp(
                name=f"f_l{layer}_{proj}_{tr}x{tc}",
                in_features=tc,
                out_features=tr,
                weight=w_real,
            )
            tasks.append((tr, tc, op_real, "F", f"real_L{layer}_{proj}", target, timeout_seconds))

        # Random seeds
        for s in seeds:
            w_rand = generate_uniform_random_weights(tr, tc, s)
            op_rand = LinearOp(
                name=f"f_rand_s{s}_{tr}x{tc}",
                in_features=tc,
                out_features=tr,
                weight=w_rand,
            )
            tasks.append((tr, tc, op_rand, "F", f"random_seed_{s}", target, timeout_seconds))

        # Execute tile tasks concurrently
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_run_single_task, *t) for t in tasks]
            for f in futures:
                results.append(f.result())

        # Map results by (design_type, weight_source)
        res_map = {(r["design_type"], r["weight_source"]): r for r in results}

        p_info = res_map.get(("P", "independent"))
        p_res = p_info["result"] if p_info and p_info["status"] == "OK" else None
        p_bus_info = res_map.get(("P_bus", "independent"))
        p_bus_res = p_bus_info["result"] if p_bus_info and p_bus_info["status"] == "OK" else None

        # 1. Programmable baseline P
        if p_res:
            e1_records.append({
                "tile_rows": tr,
                "tile_cols": tc,
                "design_type": "P",
                "weight_source": "independent",
                "target": target,
                "num_cells_total": p_res.num_cells_total,
                "num_cells_comb": p_res.num_cells_comb,
                "num_cells_seq": p_res.num_cells_seq,
                "num_lut4": p_res.num_lut4,
                "ltp_depth": p_res.ltp_depth,
                "wall_time_s": round(p_res.wall_time_seconds, 2),
                "sd_cells_total": 0.0,
                "sd_ltp_depth": 0.0,
                "status": "OK",
            })
            e3_records.append({
                "tile_rows": tr,
                "tile_cols": tc,
                "design_type": "P",
                "weight_source": "independent",
                "target": target,
                "ltp_depth": p_res.ltp_depth,
                "critical_path_ratio_vs_p": 1.0,
                "status": "OK",
            })
        else:
            e1_records.append({
                "tile_rows": tr,
                "tile_cols": tc,
                "design_type": "P",
                "weight_source": "independent",
                "target": target,
                "num_cells_total": -1,
                "num_cells_comb": -1,
                "num_cells_seq": -1,
                "num_lut4": -1,
                "ltp_depth": -1,
                "wall_time_s": timeout_seconds,
                "sd_cells_total": 0.0,
                "sd_ltp_depth": 0.0,
                "status": "TIMEOUT",
            })

        # 2. Programmable baseline P_bus
        if p_bus_res:
            e1_records.append({
                "tile_rows": tr,
                "tile_cols": tc,
                "design_type": "P_bus",
                "weight_source": "independent",
                "target": target,
                "num_cells_total": p_bus_res.num_cells_total,
                "num_cells_comb": p_bus_res.num_cells_comb,
                "num_cells_seq": p_bus_res.num_cells_seq,
                "num_lut4": p_bus_res.num_lut4,
                "ltp_depth": p_bus_res.ltp_depth,
                "wall_time_s": round(p_bus_res.wall_time_seconds, 2),
                "sd_cells_total": 0.0,
                "sd_ltp_depth": 0.0,
                "status": "OK",
            })
            e3_records.append({
                "tile_rows": tr,
                "tile_cols": tc,
                "design_type": "P_bus",
                "weight_source": "independent",
                "target": target,
                "ltp_depth": p_bus_res.ltp_depth,
                "critical_path_ratio_vs_p": round(p_bus_res.ltp_depth / p_res.ltp_depth, 3) if p_res and p_res.ltp_depth > 0 else -1.0,
                "status": "OK",
            })
        else:
            e1_records.append({
                "tile_rows": tr,
                "tile_cols": tc,
                "design_type": "P_bus",
                "weight_source": "independent",
                "target": target,
                "num_cells_total": -1,
                "num_cells_comb": -1,
                "num_cells_seq": -1,
                "num_lut4": -1,
                "ltp_depth": -1,
                "wall_time_s": timeout_seconds,
                "sd_cells_total": 0.0,
                "sd_ltp_depth": 0.0,
                "status": "TIMEOUT",
            })

        # 3. Real weights
        for proj, layer in real_configs:
            f_info = res_map.get(("F", f"real_L{layer}_{proj}"))
            f_res = f_info["result"] if f_info and f_info["status"] == "OK" else None
            if f_res:
                e1_records.append({
                    "tile_rows": tr,
                    "tile_cols": tc,
                    "design_type": "F",
                    "weight_source": f"real_L{layer}_{proj}",
                    "target": target,
                    "num_cells_total": f_res.num_cells_total,
                    "num_cells_comb": f_res.num_cells_comb,
                    "num_cells_seq": f_res.num_cells_seq,
                    "num_lut4": f_res.num_lut4,
                    "ltp_depth": f_res.ltp_depth,
                    "wall_time_s": round(f_res.wall_time_seconds, 2),
                    "sd_cells_total": 0.0,
                    "sd_ltp_depth": 0.0,
                    "status": "OK",
                })
                e3_records.append({
                    "tile_rows": tr,
                    "tile_cols": tc,
                    "design_type": "F",
                    "weight_source": f"real_L{layer}_{proj}",
                    "target": target,
                    "ltp_depth": f_res.ltp_depth,
                    "critical_path_ratio_vs_p": round(f_res.ltp_depth / p_res.ltp_depth, 3) if p_res and p_res.ltp_depth > 0 else -1.0,
                    "status": "OK",
                })
            else:
                e1_records.append({
                    "tile_rows": tr,
                    "tile_cols": tc,
                    "design_type": "F",
                    "weight_source": f"real_L{layer}_{proj}",
                    "target": target,
                    "num_cells_total": -1,
                    "num_cells_comb": -1,
                    "num_cells_seq": -1,
                    "num_lut4": -1,
                    "ltp_depth": -1,
                    "wall_time_s": timeout_seconds,
                    "sd_cells_total": 0.0,
                    "sd_ltp_depth": 0.0,
                    "status": "TIMEOUT",
                })

        # 4. Random seeds
        random_cells = []
        random_comb = []
        random_seq = []
        random_lut4 = []
        random_ltp = []
        random_times = []

        for s in seeds:
            f_rand_info = res_map.get(("F", f"random_seed_{s}"))
            f_rand_res = f_rand_info["result"] if f_rand_info and f_rand_info["status"] == "OK" else None
            if f_rand_res:
                random_cells.append(f_rand_res.num_cells_total)
                random_comb.append(f_rand_res.num_cells_comb)
                random_seq.append(f_rand_res.num_cells_seq)
                random_lut4.append(f_rand_res.num_lut4)
                random_ltp.append(f_rand_res.ltp_depth)
                random_times.append(f_rand_res.wall_time_seconds)

                e1_records.append({
                    "tile_rows": tr,
                    "tile_cols": tc,
                    "design_type": "F",
                    "weight_source": f"random_seed_{s}",
                    "target": target,
                    "num_cells_total": f_rand_res.num_cells_total,
                    "num_cells_comb": f_rand_res.num_cells_comb,
                    "num_cells_seq": f_rand_res.num_cells_seq,
                    "num_lut4": f_rand_res.num_lut4,
                    "ltp_depth": f_rand_res.ltp_depth,
                    "wall_time_s": round(f_rand_res.wall_time_seconds, 2),
                    "sd_cells_total": 0.0,
                    "sd_ltp_depth": 0.0,
                    "status": "OK",
                })

        if random_cells:
            mean_cells = float(np.mean(random_cells))
            sd_cells = float(np.std(random_cells))
            mean_comb = float(np.mean(random_comb))
            mean_seq = float(np.mean(random_seq))
            mean_lut4 = float(np.mean(random_lut4))
            mean_ltp = float(np.mean(random_ltp))
            sd_ltp = float(np.std(random_ltp))
            mean_time = float(np.mean(random_times))

            e1_records.append({
                "tile_rows": tr,
                "tile_cols": tc,
                "design_type": "F",
                "weight_source": f"random_uniform_mean_n{len(random_cells)}",
                "target": target,
                "num_cells_total": round(mean_cells, 1),
                "num_cells_comb": round(mean_comb, 1),
                "num_cells_seq": round(mean_seq, 1),
                "num_lut4": round(mean_lut4, 1),
                "ltp_depth": round(mean_ltp, 1),
                "wall_time_s": round(mean_time, 2),
                "sd_cells_total": round(sd_cells, 2),
                "sd_ltp_depth": round(sd_ltp, 2),
                "status": "OK",
            })
            e3_records.append({
                "tile_rows": tr,
                "tile_cols": tc,
                "design_type": "F",
                "weight_source": f"random_uniform_mean_n{len(random_cells)}",
                "target": target,
                "ltp_depth": round(mean_ltp, 1),
                "critical_path_ratio_vs_p": round(mean_ltp / p_res.ltp_depth, 3) if p_res and p_res.ltp_depth > 0 else -1.0,
                "status": "OK",
            })

    # Save to CSV
    save_records_to_csv(e1_records, out_e1_path)
    save_records_to_csv(e3_records, out_e3_path)

    return e1_records, e3_records


def run_experiment_e2(
    tile_size: tuple[int, int] = (8, 16),
    seeds: list[int] | None = None,
    target: str = "generic",
    timeout_seconds: float = 300.0,
    max_workers: int = 8,
    out_e2_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Execute Experiment E2 (Weight Content and Distribution Effect)."""
    if seeds is None:
        seeds = DEFAULT_SEEDS

    tr, tc = tile_size
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if out_e2_path is None:
        out_e2_path = results_dir / "E2.csv"

    from concurrent.futures import ThreadPoolExecutor

    # Base real weights: layer 0 q_proj
    real_w = load_real_quantized_slice(0, "q_proj", tr, tc)

    records: list[dict[str, Any]] = []

    # 1. Real weights
    op_real = LinearOp(name="f_e2_real", in_features=tc, out_features=tr, weight=real_w)
    res_real = run_single_synthesis(op_real, "F", target=target, timeout_seconds=timeout_seconds)
    records.append({
        "weight_class": "real",
        "description": "Layer 0 q_proj slice",
        "tile_rows": tr,
        "tile_cols": tc,
        "num_cells_total": res_real.num_cells_total,
        "num_cells_comb": res_real.num_cells_comb,
        "num_cells_seq": res_real.num_cells_seq,
        "num_lut4": res_real.num_lut4,
        "ltp_depth": res_real.ltp_depth,
        "sparsity_pct": round(float(np.mean(real_w == 0) * 100), 2),
        "wall_time_s": round(res_real.wall_time_seconds, 2),
        "sd_cells_total": 0.0,
    })

    # 2. All-zero weights (extreme sparsity)
    zero_w = generate_all_zero_weights(tr, tc)
    op_zero = LinearOp(name="f_e2_zero", in_features=tc, out_features=tr, weight=zero_w)
    res_zero = run_single_synthesis(op_zero, "F", target=target, timeout_seconds=timeout_seconds)
    records.append({
        "weight_class": "all_zero",
        "description": "All weights zero (100% sparse)",
        "tile_rows": tr,
        "tile_cols": tc,
        "num_cells_total": res_zero.num_cells_total,
        "num_cells_comb": res_zero.num_cells_comb,
        "num_cells_seq": res_zero.num_cells_seq,
        "num_lut4": res_zero.num_lut4,
        "ltp_depth": res_zero.ltp_depth,
        "sparsity_pct": 100.0,
        "wall_time_s": round(res_zero.wall_time_seconds, 2),
        "sd_cells_total": 0.0,
    })

    # Helper for multi-seed classes
    def _eval_class(weight_class: str, desc: str, gen_fn: Any) -> None:
        def _worker(s: int) -> tuple[int, SynthesisResult, float]:
            w = gen_fn(s)
            op = LinearOp(name=f"f_e2_{weight_class}_s{s}", in_features=tc, out_features=tr, weight=w)
            res = run_single_synthesis(op, "F", target=target, timeout_seconds=timeout_seconds)
            sp = float(np.mean(w == 0) * 100)
            return s, res, sp

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            worker_results = list(pool.map(_worker, seeds))

        cells = [r[1].num_cells_total for r in worker_results]
        combs = [r[1].num_cells_comb for r in worker_results]
        seqs = [r[1].num_cells_seq for r in worker_results]
        luts = [r[1].num_lut4 for r in worker_results]
        ltps = [r[1].num_ltp_depth if hasattr(r[1], "num_ltp_depth") else r[1].ltp_depth for r in worker_results]
        times = [r[1].wall_time_seconds for r in worker_results]
        sparsities = [r[2] for r in worker_results]

        records.append({
            "weight_class": weight_class,
            "description": desc,
            "tile_rows": tr,
            "tile_cols": tc,
            "num_cells_total": round(float(np.mean(cells)), 1),
            "num_cells_comb": round(float(np.mean(combs)), 1),
            "num_cells_seq": round(float(np.mean(seqs)), 1),
            "num_lut4": round(float(np.mean(luts)), 1),
            "ltp_depth": round(float(np.mean(ltps)), 1),
            "sparsity_pct": round(float(np.mean(sparsities)), 2),
            "wall_time_s": round(float(np.mean(times)), 2),
            "sd_cells_total": round(float(np.std(cells)), 2),
        })

    # 3. Uniform random INT8
    _eval_class(
        "uniform_random",
        "Uniform random INT8 [-127, 127]",
        lambda s: generate_uniform_random_weights(tr, tc, s),
    )

    # 4. Magnitude-matched random
    _eval_class(
        "magnitude_matched",
        "Preserved |w| histogram, randomized signs and spatial layout",
        lambda s: generate_magnitude_matched_weights(real_w, s),
    )

    # 5. Shuffled real weights
    _eval_class(
        "shuffled_real",
        "Exact real weights with permuted (n,k) positions",
        lambda s: generate_shuffled_real_weights(real_w, s),
    )

    # 6. One-hot weights
    _eval_class(
        "one_hot",
        "Single nonzero coefficient per row (+1 or -1)",
        lambda s: generate_one_hot_weights(tr, tc, s),
    )

    save_records_to_csv(records, out_e2_path)
    return records


def save_records_to_csv(records: list[dict[str, Any]], out_path: Path) -> None:
    """Save dictionary records to CSV with sorted consistent headers."""
    if not records:
        return
    fieldnames = list(records[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"Saved {len(records)} records to {out_path}")


def main() -> None:
    """CLI entry point for synthesis experiments."""
    parser = argparse.ArgumentParser(description="Run logic-synthesis experiments E1, E2, E3")
    parser.add_argument("--e1-e3", action="store_true", help="Execute Experiments E1 and E3")
    parser.add_argument("--e2", action="store_true", help="Execute Experiment E2")
    parser.add_argument("--all", action="store_true", help="Execute all experiments (E1, E2, E3)")
    parser.add_argument("--target", default="generic", choices=["generic", "ice40"])
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.all or args.e1_e3:
        run_experiment_e1_e3(target=args.target, timeout_seconds=args.timeout)
    if args.all or args.e2:
        run_experiment_e2(target=args.target, timeout_seconds=args.timeout)


if __name__ == "__main__":
    main()
