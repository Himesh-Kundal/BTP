"""Command-Line Interface (CLI) for LLMHW Hardware Compiler (T8.1).

Provides the unified entry point `llmhw`:
- `llmhw extract`: Extract model weights and capture calibration activations.
- `llmhw compile`: Compile dense linear layer to synthesizable Verilog (Design F / P / P_bus).
- `llmhw verify` : Verify RTL against independent NumPy INT64 reference oracle (Claim C1) + static weight audit.
- `llmhw synth`  : Run Yosys logic synthesis and extract cell/flop/depth metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


def find_artifact_dir(layer: int, proj: str, base_dir: Path | str = "artifacts") -> Path:
    """Locate artifact directory for a given layer index and projection type."""
    base = Path(base_dir).resolve()
    if not base.exists():
        raise FileNotFoundError(f"Artifacts base directory not found at: {base}")

    # 1. Direct path check
    direct = base / f"layer_{layer}" / proj
    if direct.is_dir() and (direct / "weight.npy").is_file():
        return direct

    # 2. Search subdirectories (e.g. SmolLM2-135M@93efa2f097d5)
    for model_dir in sorted(base.glob("*")):
        if not model_dir.is_dir():
            continue
        candidate = model_dir / f"layer_{layer}" / proj
        if candidate.is_dir() and (candidate / "weight.npy").is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not locate artifacts for layer {layer} '{proj}' under {base}"
    )


def cmd_extract(args: argparse.Namespace) -> int:
    """Execute 'llmhw extract': extract weights and capture activations."""
    print("================================================================================")
    print(" LLMHW EXTRACT — MODEL WEIGHT & ACTIVATION EXTRACTION")
    print("================================================================================")
    print(f"[*] Target Model   : {args.model}")
    print(f"[*] Revision       : {args.revision}")
    print(f"[*] Layers Target  : {args.layer}")
    print(f"[*] Projections    : {args.proj}")
    print(f"[*] Output Dir     : {args.out}")

    try:
        from frontend.activation_capture import capture_activations
        from frontend.artifact_dumper import dump_linear_artifact
        from frontend.linear_extractor import PROJECTION_TYPES, extract_linear_modules
        from frontend.model_loader import load_model_and_tokenizer
    except ImportError as e:
        print(f"[-] Frontend extraction requires torch/transformers dependencies: {e}", file=sys.stderr)
        return 1

    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)

    # Resolve layers to extract
    if args.layer == "all":
        target_layers = list(range(30))
    else:
        target_layers = [int(x.strip()) for x in args.layer.split(",") if x.strip()]

    # Resolve projections
    if args.proj == "all":
        target_projs = list(PROJECTION_TYPES)
    else:
        target_projs = [x.strip() for x in args.proj.split(",") if x.strip()]

    print(f"[*] Loading model {args.model}@{args.revision[:12]}...")
    model, tokenizer = load_model_and_tokenizer(model_id=args.model, revision=args.revision)

    print(f"[*] Extracting linear modules for layers {target_layers} and types {target_projs}...")
    extracted_modules = extract_linear_modules(model)

    filtered_modules = [
        m for m in extracted_modules
        if m.layer_idx in target_layers and m.proj_type in target_projs
    ]

    print(f"[*] Dumping weight artifacts for {len(filtered_modules)} modules...")
    dumped_paths = []
    for mod in filtered_modules:
        p = dump_linear_artifact(
            mod,
            base_dir=out_base,
            model_id=args.model,
            revision=args.revision,
        )
        dumped_paths.append(p)

    print(f"[+] Dumped {len(dumped_paths)} linear module weight artifacts.")

    if args.capture_calib:
        print(f"[*] Capturing activation samples (num_samples={args.num_samples})...")
        calib_paths = capture_activations(
            model=model,
            tokenizer=tokenizer,
            layers=target_layers,
            base_dir=out_base,
            model_id=args.model,
            revision=args.revision,
        )
        print(f"[+] Captured activations for {len(calib_paths)} projection targets.")

    print("\n[+] Extraction successfully completed.")
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    """Execute 'llmhw compile': compile linear layer to synthesizable Verilog."""
    from backend.baseline_p import generate_full_module_p, generate_full_module_p_bus
    from backend.module_generator import generate_full_module
    from backend.tiling import tile_weight_matrix
    from backend.verilog_codegen import sanitize_identifier
    from compiler.quantization import quantize_weights
    from ir.linear import LinearOp

    print("================================================================================")
    print(" LLMHW COMPILE — FIXED-WEIGHT & BASELINE VERILOG COMPILER")
    print("================================================================================")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load weights
    if args.weights is not None:
        weight_path = Path(args.weights).resolve()
        if not weight_path.is_file():
            print(f"[-] Weight file not found: {weight_path}", file=sys.stderr)
            return 1
        w_raw = np.load(weight_path)
        op_name = args.arch if args.arch else weight_path.stem
        print(f"[*] Loaded weights from: {weight_path}")
    else:
        try:
            art_dir = find_artifact_dir(layer=args.layer, proj=args.arch, base_dir=args.artifacts_dir)
            w_raw = np.load(art_dir / "weight.npy")
            op_name = f"L{args.layer}_{args.arch}"
            print(f"[*] Loaded artifacts from: {art_dir}")
        except FileNotFoundError as e:
            print(f"[-] {e}", file=sys.stderr)
            return 1

    # 2. Quantize if float
    if np.issubdtype(w_raw.dtype, np.floating):
        print("[*] Input weights are floating point. Applying Rule N2 symmetric INT8 quantization...")
        w_q, _ = quantize_weights(w_raw)
    else:
        w_q = w_raw.astype(np.int8)

    # 3. Optional slicing for demo/evaluation
    if args.slice_n is not None or args.slice_k is not None:
        sn = args.slice_n if args.slice_n is not None else w_q.shape[0]
        sk = args.slice_k if args.slice_k is not None else w_q.shape[1]
        w_q = w_q[:sn, :sk].copy()
        op_name = f"{op_name}_s{sn}x{sk}"
        print(f"[*] Sliced weight matrix to ({sn}, {sk})")

    n, k = w_q.shape
    print(f"[*] Matrix Shape   : N={n}, K={k}")
    print(f"[*] Tile Config    : Nt={args.tile_n}, Kt={args.tile_k}")
    print(f"[*] Architecture   : Design {args.design}")
    print(f"[*] Latency L      : {args.latency}")

    # 4. Tiling & Manifest
    manifest, _ = tile_weight_matrix(
        w_q,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        layer_name=op_name,
    )

    op = LinearOp(
        name=op_name,
        in_features=k,
        out_features=n,
        weight=w_q,
        is_quantized=True,
    )

    # 5. RTL Generation
    custom_name = args.module_name
    if args.design == "F":
        rtl_code = generate_full_module(
            op,
            manifest,
            tile_n=args.tile_n,
            tile_k=args.tile_k,
            latency=args.latency,
            module_name=custom_name,
        )
        raw_mod = f"llmhw_{op.name}" if not custom_name else custom_name
        top_name = sanitize_identifier(raw_mod)
    elif args.design == "P":
        rtl_code = generate_full_module_p(
            op,
            manifest,
        )
        top_name = sanitize_identifier(f"llmhw_{op.name}_p")
    elif args.design == "P_bus":
        rtl_code = generate_full_module_p_bus(
            op,
            manifest,
        )
        top_name = sanitize_identifier(f"llmhw_{op.name}_p_bus")
    else:
        print(f"[-] Unknown design variant: {args.design}", file=sys.stderr)
        return 1

    rtl_file = out_dir / f"{top_name}.v"
    rtl_file.write_text(rtl_code, encoding="utf-8")

    manifest_file = out_dir / f"{top_name}.manifest.json"
    manifest.save(manifest_file)

    num_lines = len(rtl_code.splitlines())
    file_size_kb = rtl_file.stat().st_size / 1024.0

    print("--------------------------------------------------------------------------------")
    print(f"[+] Top Module     : {top_name}")
    print(f"[+] Output Verilog : {rtl_file} ({num_lines} lines, {file_size_kb:.2f} KB)")
    print(f"[+] Manifest File  : {manifest_file}")
    print(f"[+] Manifest Hash  : {manifest.manifest_hash}")
    print("--------------------------------------------------------------------------------")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Execute 'llmhw verify': verify RTL against reference oracle and audit weights."""
    import tempfile

    from backend.testbench import generate_wrapper_testbench, generate_wrapper_testbench_p
    from backend.tiling import TileManifest, tile_weight_matrix
    from backend.vectors import write_hex_vectors, write_weight_hex_vectors
    from compiler.quantization import quantize_weights
    from ir.linear import LinearOp
    from verification.reference import linear_reference_int64
    from verification.simulator import run_simulation, run_verilator_simulation
    from verification.weight_audit import audit_rtl_source, audit_weights_match

    print("================================================================================")
    print(" LLMHW VERIFY — BIT-EXACT SIMULATION ORACLE (CLAIM C1) & WEIGHT AUDIT")
    print("================================================================================")

    # 1. Resolve RTL file
    if args.rtl is None:
        default_rtl = Path("build/rtl") / f"llmhw_L{args.layer}_{args.arch}.v"
        if default_rtl.is_file():
            rtl_path = default_rtl.resolve()
        else:
            print("[-] Error: --rtl file must be specified or present at default location.", file=sys.stderr)
            return 1
    else:
        rtl_path = Path(args.rtl).resolve()
        if not rtl_path.is_file():
            print(f"[-] RTL file not found: {rtl_path}", file=sys.stderr)
            return 1

    rtl_content = rtl_path.read_text(encoding="utf-8")
    top_name = rtl_path.stem

    # Detect design variant
    if top_name.endswith("_p"):
        design_type = "P"
    elif top_name.endswith("_p_bus"):
        design_type = "P_bus"
    else:
        design_type = "F"

    print(f"[*] DUT RTL File   : {rtl_path}")
    print(f"[*] Top Module     : {top_name} (Design {design_type})")
    print(f"[*] Simulator      : {args.sim}")
    print(f"[*] Test Vectors   : {args.num_vectors}")

    # 2. Resolve weights
    if args.weights is not None:
        w_path = Path(args.weights).resolve()
        w_raw = np.load(w_path)
    else:
        try:
            art_dir = find_artifact_dir(layer=args.layer, proj=args.arch, base_dir=args.artifacts_dir)
            w_raw = np.load(art_dir / "weight.npy")
        except FileNotFoundError as e:
            print(f"[-] {e}", file=sys.stderr)
            return 1

    if np.issubdtype(w_raw.dtype, np.floating):
        w_q, _ = quantize_weights(w_raw)
    else:
        w_q = w_raw.astype(np.int8)

    # Check if module name contains slice geometry, e.g. _s8x16
    m_slice = re.search(r"_s(\d+)x(\d+)", top_name)
    if m_slice:
        sn, sk = int(m_slice.group(1)), int(m_slice.group(2))
        w_q = w_q[:sn, :sk].copy()

    n, k = w_q.shape

    # 3. Resolve manifest
    if args.manifest is not None:
        man_path = Path(args.manifest).resolve()
        manifest = TileManifest.load(man_path)
    else:
        candidate_man = rtl_path.with_suffix(".manifest.json")
        if candidate_man.is_file():
            manifest = TileManifest.load(candidate_man)
        else:
            manifest, _ = tile_weight_matrix(w_q, tile_n=8, tile_k=16, layer_name=top_name)

    op = LinearOp(
        name=top_name,
        in_features=k,
        out_features=n,
        weight=w_q,
        is_quantized=True,
    )

    # 4. Static Weight Audit (Design F only)
    audit_results: dict[str, Any] = {"passed": True}
    if design_type == "F" and not args.no_audit:
        print("\n--- Running Static Interface & Weight Port Audit ---")
        port_audit = audit_rtl_source(
            rtl_content,
            expected_k=manifest.k_padded,
            expected_n=manifest.n_padded,
            strict=False,
        )
        const_audit = audit_weights_match(
            rtl_content,
            expected_weights=w_q,
            manifest=manifest,
            strict=False,
        )
        const_audit_clean = {k: v for k, v in const_audit.items() if k != "reconstructed_weights"}
        audit_passed = port_audit["passed"] and const_audit["passed"]
        audit_results = {
            "passed": audit_passed,
            "port_audit": port_audit,
            "const_audit": const_audit_clean,
        }
        if not audit_passed:
            print("[-] STATIC WEIGHT AUDIT FAILED:", file=sys.stderr)
            for v in port_audit.get("violations", []):
                print(f"    - {v}", file=sys.stderr)
            for v in const_audit.get("violations", []):
                print(f"    - {v}", file=sys.stderr)
            return 1
        print("[+] Interface Audit : PASS (0 weight ports, 0 weight signals, 0 memory arrays)")
        print(f"[+] Constant Audit  : PASS ({const_audit['total_weights_checked']} weights verified bit-exact)")

    # 5. Generate Stimulus & Reference Oracle
    print("\n--- Generating Stimulus and Golden Reference Oracle ---")
    rng = np.random.default_rng(seed=42)
    # Generate mix of random vectors, extreme values, and one-hot
    x_stim = rng.integers(-127, 128, size=(args.num_vectors, k), dtype=np.int8)
    if args.num_vectors >= 3:
        x_stim[0] = 0
        x_stim[1] = 127
        x_stim[2] = -127

    y_exp = linear_reference_int64(x_stim, w_q)
    print(f"[*] Oracle Output  : Shape {y_exp.shape}, INT64 evaluation")

    # 6. Run Simulation
    if args.work_dir is not None:
        work_dir = Path(args.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_obj = tempfile.TemporaryDirectory(prefix=f"sim_{top_name}_")
        work_dir = Path(temp_obj.name)

    write_hex_vectors(x_stim, y_exp, work_dir)
    if design_type == "P":
        write_weight_hex_vectors(w_q, work_dir)

    # Generate testbench
    if design_type == "P":
        tb_code = generate_wrapper_testbench_p(
            op=op,
            manifest=manifest,
            num_vectors=len(x_stim),
            module_name=top_name,
            latency=1,
        )
    else:
        tb_code = generate_wrapper_testbench(
            op=op,
            num_vectors=len(x_stim),
            module_name=top_name,
            latency=1,
            manifest=manifest,
        )

    tb_path = work_dir / "tb.v"
    tb_path.write_text(tb_code, encoding="utf-8")

    print(f"\n--- Running Hardware Simulation ({args.sim}) ---")
    if args.sim == "iverilog":
        sim_res = run_simulation(
            dut_path=rtl_path,
            tb_path=tb_path,
            expected_checked=len(x_stim),
            work_dir=work_dir,
            dump_vcd=args.dump_vcd,
        )
    elif args.sim == "verilator":
        sim_res = run_verilator_simulation(
            dut_path=rtl_path,
            tb_path=tb_path,
            expected_checked=len(x_stim),
            work_dir=work_dir,
        )
    else:
        print(f"[-] Unsupported simulator: {args.sim}", file=sys.stderr)
        return 1

    print(f"Status             : {sim_res.status}")
    print(f"Vectors Checked    : {sim_res.checked}")
    print(f"Mismatches         : {sim_res.mismatch}")
    print(f"Elapsed Time       : {sim_res.elapsed_seconds * 1000:.2f} ms")

    # 7. Write Report
    out_report = Path(args.out).resolve()
    out_report.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "top_module": top_name,
        "design_type": design_type,
        "simulator": args.sim,
        "status": sim_res.status,
        "checked": sim_res.checked,
        "mismatch": sim_res.mismatch,
        "elapsed_seconds": sim_res.elapsed_seconds,
        "weight_shape": [n, k],
        "audit": audit_results,
    }
    out_report.write_text(json.dumps(report_data, indent=2, default=str), encoding="utf-8")
    print(f"[+] Report Saved   : {out_report}")

    if sim_res.status != "PASS":
        print(f"[-] SIMULATION FAILED:\n{sim_res.stderr}", file=sys.stderr)
        return 1

    print("\n================================================================================")
    print(" VERIFICATION PASSED — BIT-EXACT MATCH WITH INDEPENDENT ORACLE (CLAIM C1)")
    print("================================================================================")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    """Execute 'llmhw synth': run Yosys logic synthesis and extract metrics."""
    from synth.yosys_runner import run_yosys_synthesis

    print("================================================================================")
    print(" LLMHW SYNTH — YOSYS LOGIC SYNTHESIS EVALUATION")
    print("================================================================================")

    rtl_path = Path(args.rtl).resolve()
    if not rtl_path.is_file():
        print(f"[-] RTL file not found: {rtl_path}", file=sys.stderr)
        return 1

    top = args.top if args.top else rtl_path.stem
    flatten = not args.no_flatten

    print(f"[*] Top Module     : {top}")
    print(f"[*] RTL File       : {rtl_path}")
    print(f"[*] Target         : {args.target}")
    print(f"[*] Flatten        : {flatten}")

    res = run_yosys_synthesis(
        top_module=top,
        verilog_paths=[rtl_path],
        target=args.target,
        flatten=flatten,
    )

    if res.returncode != 0:
        print(f"[-] Synthesis failed with return code {res.returncode}:\n{res.stderr}", file=sys.stderr)
        return 1

    print("\n--- Synthesis Results ---")
    print(f"Status             : SUCCESS (code {res.returncode})")
    print(f"Wall Time          : {res.wall_time_seconds:.2f} s")
    print(f"Total Cells        : {res.num_cells_total}")
    print(f"Combinational      : {res.num_cells_comb}")
    print(f"Sequential (DFF)   : {res.num_cells_seq}")
    if args.target == "ice40":
        print(f"LUT-4 Count        : {res.num_lut4}")
    print(f"LTP Depth          : {res.ltp_depth}")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
    print(f"[+] Metrics Saved  : {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser for the llmhw CLI."""
    parser = argparse.ArgumentParser(
        prog="llmhw",
        description="LLMHW — Model-Specific Dense Linear Layer Hardware Compiler",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. extract
    p_extract = subparsers.add_parser(
        "extract",
        help="Extract model weights and capture calibration activations",
    )
    p_extract.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M", help="Model ID or directory")
    p_extract.add_argument("--revision", type=str, default="93efa2f097d58c2a74874c7e644dbc9b0cee75a2", help="Pinned commit")
    p_extract.add_argument("--layer", type=str, default="0", help="Layer index (0..29) or 'all'")
    p_extract.add_argument("--proj", type=str, default="all", help="Projection type or 'all'")
    p_extract.add_argument("--out", type=Path, default=Path("artifacts"), help="Output artifact directory")
    p_extract.add_argument("--capture-calib", action="store_true", default=True, help="Capture calibration activations")
    p_extract.add_argument("--no-capture-calib", action="store_false", dest="capture_calib", help="Skip calibration capture")
    p_extract.add_argument("--num-samples", type=int, default=16, help="Number of calibration samples")

    # 2. compile
    p_compile = subparsers.add_parser(
        "compile",
        help="Compile dense linear layer to synthesizable Verilog (Design F / P / P_bus)",
    )
    p_compile.add_argument("--weights", type=Path, default=None, help="Path to .npy weight file")
    p_compile.add_argument("--layer", type=int, default=0, help="Layer index if loading from artifacts")
    p_compile.add_argument("--arch", type=str, default="q_proj", help="Projection type / layer name")
    p_compile.add_argument("--design", choices=["F", "P", "P_bus"], default="F", help="Architecture variant")
    p_compile.add_argument("--tile-n", type=int, default=8, help="Tile row size Nt")
    p_compile.add_argument("--tile-k", type=int, default=16, help="Tile col size Kt")
    p_compile.add_argument("--slice-n", type=int, default=None, help="Optional row slicing for fast evaluation")
    p_compile.add_argument("--slice-k", type=int, default=None, help="Optional col slicing for fast evaluation")
    p_compile.add_argument("--latency", type=int, default=1, help="Module pipeline latency L")
    p_compile.add_argument("--module-name", type=str, default=None, help="Custom top module name")
    p_compile.add_argument("--out", type=Path, default=Path("build/rtl"), help="Output directory for generated RTL")
    p_compile.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Artifacts root directory")

    # 3. verify
    p_verify = subparsers.add_parser(
        "verify",
        help="Verify RTL against reference oracle and run static weight audit",
    )
    p_verify.add_argument("--rtl", type=Path, default=None, help="Path to DUT Verilog file")
    p_verify.add_argument("--weights", type=Path, default=None, help="Path to .npy weight file")
    p_verify.add_argument("--layer", type=int, default=0, help="Layer index if loading from artifacts")
    p_verify.add_argument("--arch", type=str, default="q_proj", help="Projection type if loading from artifacts")
    p_verify.add_argument("--manifest", type=Path, default=None, help="Path to TileManifest JSON")
    p_verify.add_argument("--sim", choices=["iverilog", "verilator"], default="iverilog", help="Simulator engine")
    p_verify.add_argument("--num-vectors", type=int, default=20, help="Number of stimulus vectors")
    p_verify.add_argument("--dump-vcd", action="store_true", help="Dump simulation waveform VCD")
    p_verify.add_argument("--no-audit", action="store_true", help="Skip static weight audit")
    p_verify.add_argument("--out", type=Path, default=Path("results/cli_verify_report.json"), help="Output report JSON")
    p_verify.add_argument("--work-dir", type=Path, default=None, help="Working directory for simulation")
    p_verify.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Artifacts root directory")

    # 4. synth
    p_synth = subparsers.add_parser(
        "synth",
        help="Run Yosys logic synthesis and extract metrics",
    )
    p_synth.add_argument("--top", type=str, default=None, help="Top module name (auto-detected if None)")
    p_synth.add_argument("--rtl", type=Path, required=True, help="Path to synthesizable Verilog file")
    p_synth.add_argument("--target", choices=["generic", "ice40"], default="generic", help="Techmap target")
    p_synth.add_argument("--no-flatten", action="store_true", help="Keep module hierarchy unflattened")
    p_synth.add_argument("--out", type=Path, default=Path("results/cli_synth_report.json"), help="Output report JSON")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the llmhw CLI."""
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)
    if args.subcommand == "extract":
        return cmd_extract(args)
    if args.subcommand == "compile":
        return cmd_compile(args)
    if args.subcommand == "verify":
        return cmd_verify(args)
    if args.subcommand == "synth":
        return cmd_synth(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
