"""Command-Line Interface (CLI v2) for LLMHW Hardware Compiler (T8.1 & UG.1).

Provides the unified entry point `llmhw`:
Part I (Linear-Layer Compiler):
- `llmhw extract`: Extract model weights and capture calibration activations.
- `llmhw compile`: Compile dense linear layer to synthesizable Verilog (Design F / P / P_bus).
- `llmhw verify` : Verify RTL against independent NumPy INT64 reference oracle (Claim C1) + static weight audit.
- `llmhw synth`  : Run Yosys logic synthesis and extract cell/flop/depth metrics.

Part II (Full Decoder Model Accelerator Compiler):
- `llmhw compile-model`: Compile full autoregressive transformer decoder (tiny or SmolLM2-135M) to ROM-backed Verilog accelerator bundle (`top_accel.v`, ROM hex images, microprogram schedule, memory plan).
- `llmhw run`          : Execute autoregressive token generation on a compiled decoder bundle via `interpreter` (fast bit-exact software oracle) or `rtl` (Verilator hardware simulation).
- `llmhw verify --bundle <dir>`: Verify compiled v2 decoder bundle (structural & ROM weight audit v2 + RTL vs interpreter token-exact simulation).
- `llmhw synth --unit <name>`  : Synthesize Part II RTL library unit (`rtl/lib/<unit>.v`) in Yosys and report cells, DFFs, and logic depth.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
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


def cmd_compile_model(args: argparse.Namespace) -> int:
    """Execute 'llmhw compile-model': compile full decoder model to ROM-backed RTL accelerator bundle."""
    from backend.top_generator import emit_hardware_bundle
    from compiler.memory_planner import plan_memory
    from frontend.importer_v2 import (
        build_graph_v2_from_artifacts,
        calibrate_and_migrate_artifacts_v2,
        extract_and_dump_model_v2,
        load_model_artifacts_v2,
    )

    print("================================================================================")
    print(" LLMHW COMPILE-MODEL — FULL DECODER ROM-BACKED ACCELERATOR COMPILER (PART II)")
    print("================================================================================")
    t0 = time.perf_counter()

    repo_root = Path(__file__).resolve().parent
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_config = json.loads((repo_root / "numerics" / "config_frozen.json").read_text(encoding="utf-8"))

    # 1. Resolve or extract v2 model artifacts
    if args.artifacts_dir is not None:
        art_dir = Path(args.artifacts_dir).resolve()
        if not (art_dir / "manifest.json").is_file():
            print(f"[-] Artifacts manifest not found at: {art_dir / 'manifest.json'}", file=sys.stderr)
            return 1
    elif args.tiny:
        from frontend.tiny_model import generate_tiny_model
        art_dir = out_dir / "artifacts_v2"
        model = generate_tiny_model(seed=42)
        extract_and_dump_model_v2(
            model=model,
            config=model.config,
            target_dir=art_dir,
            model_id="tiny-decoder-v2",
            revision="tiny-seed-42",
        )
    else:
        default_art = (repo_root / "build" / "smollm2_artifacts_v2").resolve()
        if (default_art / "manifest.json").is_file() and not args.force_extract:
            art_dir = default_art
            print(f"[*] Reusing existing extracted v2 artifacts: {art_dir}")
        else:
            from eval.harness import get_tokenized_windows
            from frontend.model_loader import load_model_and_tokenizer
            art_dir = default_art
            print(f"[*] Loading {args.model}@{args.revision[:12]} for full decoder extraction...")
            model, _ = load_model_and_tokenizer(model_id=args.model, revision=args.revision)
            extract_and_dump_model_v2(
                model=model,
                config=model.config,
                target_dir=art_dir,
                model_id=args.model,
                revision=args.revision,
            )
            calib_windows = get_tokenized_windows(
                model_id=args.model,
                revision=args.revision,
                num_windows=4,
                window_length=64,
            )
            calibrate_and_migrate_artifacts_v2(
                model=model,
                calibration_windows=calib_windows,
                target_dir=art_dir,
                s_gate=0.10,
                swig_shift=6,
                score_shift=1,
            )

    # 2. Load artifacts & build GraphIR
    print(f"[*] Loading v2 artifacts and building GraphV2 from: {art_dir}")
    artifacts = load_model_artifacts_v2(art_dir)
    graph = build_graph_v2_from_artifacts(art_dir)
    cfg = dict(artifacts["config"])
    max_ctx = int(args.max_ctx) if args.max_ctx is not None else int(cfg.get("max_position_embeddings", 256))
    cfg["max_position_embeddings"] = max_ctx

    # 3. Emit hardware bundle & memory plan
    print(f"[*] Emitting ROM-backed hardware bundle into: {out_dir} (R={args.tile_r}, C={args.tile_c}, MAX_CTX={max_ctx})...")
    bundle_res = emit_hardware_bundle(
        graph=graph,
        weights_dict=artifacts["tensors"],
        config=cfg,
        frozen_config=frozen_config,
        out_dir=out_dir,
        tile_r=args.tile_r,
        tile_c=args.tile_c,
        max_ctx=max_ctx,
        calib_scales=artifacts.get("calib_scales"),
    )
    mem_plan = plan_memory(cfg, max_ctx=max_ctx)
    mem_plan_dict = mem_plan.to_dict()
    mem_plan_dict["model_config"] = cfg
    (out_dir / "memory_plan.json").write_text(json.dumps(mem_plan_dict, indent=2, sort_keys=True), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    top_v = out_dir / "top_accel.v"
    addr_map = bundle_res["address_map"]
    w_words = addr_map["weights_image"]["total_words"]
    p_words = addr_map["requant_image"]["total_words"]
    print("--------------------------------------------------------------------------------")
    print(f"[+] Bundle Dir     : {out_dir}")
    print(f"[+] Top Verilog    : {top_v} ({len(top_v.read_text(encoding='utf-8').splitlines())} lines)")
    print(f"[+] Weight Words   : {w_words:,} bytes")
    print(f"[+] Requant Words  : {p_words:,} entries")
    print(f"[+] Dynamic SRAM   : {mem_plan.total_dynamic_sram_bytes:,} bytes")
    print(f"[+] Compile Time   : {elapsed:.2f} s")
    print("--------------------------------------------------------------------------------")
    return 0


def _run_verilator_bundle_tokens(
    bundle_dir: Path,
    prompt_tokens: list[int],
    max_new_tokens: int,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Run Verilator simulation on a compiled v2 hardware bundle and return generated tokens & cycle stats."""
    from sim.tb_generator import generate_top_accel_tb_verilog

    bundle_dir = bundle_dir.resolve()
    if (bundle_dir / "memory_plan.json").is_file():
        mem_plan = json.loads((bundle_dir / "memory_plan.json").read_text(encoding="utf-8"))
        cfg = mem_plan.get("model_config")
    else:
        cfg = None

    if not cfg:
        sched = json.loads((bundle_dir / "schedule.json").read_text(encoding="utf-8"))
        cfg = sched["config"]

    cleanup_temp = False
    if work_dir is None:
        temp_obj = tempfile.TemporaryDirectory(prefix="llmhw_run_rtl_")
        work_path = Path(temp_obj.name)
        cleanup_temp = True
    else:
        work_path = Path(work_dir).resolve()
        work_path.mkdir(parents=True, exist_ok=True)

    try:
        tb_code = generate_top_accel_tb_verilog(
            prompt_tokens=prompt_tokens,
            num_generate=max_new_tokens,
            seq_id=0,
            trace_dir=None,
            expected_tokens=None,
            timeout_cycles=1_000_000_000,
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            vocab_size=cfg["vocab_size"],
            num_q_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
        )
        tb_file = work_path / "tb_run.v"
        tb_file.write_text(tb_code, encoding="utf-8")

        ver_bin = shutil.which("verilator")
        if not ver_bin:
            raise RuntimeError("verilator binary not found on PATH")

        repo_root = Path(__file__).resolve().parent
        rtl_lib = repo_root / "rtl" / "lib"
        all_rtl = sorted(rtl_lib.glob("*.v"))
        sources = [p for p in all_rtl if p.name != "top_accel.v"] + [bundle_dir / "top_accel.v", tb_file]

        sim_dir = work_path / "obj_dir"
        compile_cmd = [
            ver_bin,
            "--binary",
            "--timing",
            "-j",
            "4",
            "-CFLAGS",
            "-O3",
            "-Wall",
            "-Wno-BLKSEQ",
            "-Wno-WIDTHTRUNC",
            "-Wno-WIDTHEXPAND",
            "-Wno-UNUSEDSIGNAL",
            "-Wno-PINMISSING",
            "-Wno-UNUSEDPARAM",
            "-Wno-INITIALDLY",
            "-Wno-DECLFILENAME",
            "-Wno-CASEINCOMPLETE",
            "-Wno-SYNCASYNCNET",
            "-Wno-UNDRIVEN",
            f"-I{rtl_lib}",
            "--top-module",
            "tb_top_accel",
            "-Mdir",
            str(sim_dir),
        ] + [str(s) for s in sources]

        res_c = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
        if res_c.returncode != 0:
            raise RuntimeError(f"Verilator compile failed:\n{res_c.stderr}\n{res_c.stdout}")

        t0 = time.perf_counter()
        res_r = subprocess.run([str(sim_dir / "Vtb_top_accel")], capture_output=True, text=True, check=False)
        sim_sec = time.perf_counter() - t0
        if res_r.returncode != 0:
            raise RuntimeError(f"Verilator sim failed:\n{res_r.stderr}\n{res_r.stdout}")

        raw_out = res_r.stdout + res_r.stderr
        all_toks = [int(m.group(2)) for m in re.finditer(r"STEP\s+(\d+)\s+TOK_OUT=(\d+)", raw_out)]
        gen_toks = all_toks[len(prompt_tokens) - 1 : len(prompt_tokens) - 1 + max_new_tokens]
        m_cyc = re.search(r"TOTAL_CYCLES=(\d+)", raw_out)
        total_cycles = int(m_cyc.group(1)) if m_cyc else 0

        return {
            "all_tokens": all_toks,
            "generated_tokens": gen_toks,
            "total_cycles": total_cycles,
            "sim_sec": round(sim_sec, 3),
        }
    finally:
        if cleanup_temp:
            temp_obj.cleanup()


def cmd_run(args: argparse.Namespace) -> int:
    """Execute 'llmhw run': run autoregressive token generation on a compiled v2 hardware bundle."""
    from compiler.interpreter import CompiledArtifactInterpreter

    print("================================================================================")
    print(" LLMHW RUN — AUTOREGRESSIVE DECODER EXECUTION (INTERPRETER / VERILATOR RTL)")
    print("================================================================================")

    bundle_dir = Path(args.bundle).resolve()
    if not (bundle_dir / "top_accel.v").is_file():
        print(f"[-] Compiled hardware bundle not found at: {bundle_dir}", file=sys.stderr)
        return 1

    tokenizer = None
    if args.tokens is not None:
        prompt_ids = [int(x.strip()) for x in args.tokens.split(",") if x.strip()]
        prompt_repr = f"tokens={prompt_ids}"
    else:
        from transformers import AutoTokenizer

        from frontend.model_loader import DEFAULT_MODEL_ID, PINNED_REVISION
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID, revision=PINNED_REVISION)
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
        prompt_repr = repr(args.prompt)

    print(f"[*] Hardware Bundle: {bundle_dir}")
    print(f"[*] Backend Engine : {args.backend}")
    print(f"[*] Prompt         : {prompt_repr} (ids={prompt_ids})")
    print(f"[*] Max New Tokens : {args.max_new_tokens}")

    t0 = time.perf_counter()
    interp = CompiledArtifactInterpreter(bundle_dir)
    interp.reset()
    interp_all: list[int] = []
    total_steps = len(prompt_ids) + args.max_new_tokens
    for pos in range(total_steps):
        tok_in = prompt_ids[pos] if pos < len(prompt_ids) else interp_all[-1]
        _, tok_out = interp.decode_step(tok_in, pos=pos)
        interp_all.append(tok_out)
    interp_gen = interp_all[len(prompt_ids) - 1 : len(prompt_ids) - 1 + args.max_new_tokens]
    interp_sec = time.perf_counter() - t0

    rtl_res = None
    if args.backend in ("rtl", "both"):
        print("[*] Running Verilator RTL hardware simulation...")
        rtl_res = _run_verilator_bundle_tokens(
            bundle_dir=bundle_dir,
            prompt_tokens=prompt_ids,
            max_new_tokens=args.max_new_tokens,
        )
        gen_ids = rtl_res["generated_tokens"]
        if rtl_res["all_tokens"] != interp_all:
            print("[-] MISMATCH between RTL and CompiledArtifactInterpreter!", file=sys.stderr)
            return 1
        print(f"[+] RTL vs Interpreter Parity: 100% BIT-EXACT ({len(interp_all)}/{len(interp_all)} steps, {rtl_res['total_cycles']:,} cycles)")
    else:
        gen_ids = interp_gen

    decoded_text = tokenizer.decode(gen_ids) if tokenizer is not None else None
    print("--------------------------------------------------------------------------------")
    print(f"[+] Generated IDs  : {gen_ids}")
    if decoded_text is not None:
        print(f"[+] Generated Text : {decoded_text!r}")
    print(f"[+] Interpreter Sec: {interp_sec:.3f} s")
    if rtl_res is not None:
        print(f"[+] RTL Cycles     : {rtl_res['total_cycles']:,} ({rtl_res['sim_sec']:.2f} s)")
    print("--------------------------------------------------------------------------------")

    if args.out is not None:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bundle": str(bundle_dir),
            "backend": args.backend,
            "prompt_tokens": prompt_ids,
            "max_new_tokens": args.max_new_tokens,
            "generated_tokens": gen_ids,
            "decoded_text": decoded_text,
            "interpreter_sec": round(interp_sec, 4),
            "rtl": rtl_res,
            "status": "PASS",
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[+] Run Report     : {out_path}")

    return 0


def _verify_v2_bundle(args: argparse.Namespace) -> int:
    """Verify a Part II full-decoder hardware bundle (ROM/structural audits + RTL vs interpreter simulation)."""
    from compiler.interpreter import CompiledArtifactInterpreter
    from verification.structural_audit import (
        audit_no_expected_values_in_rtl,
        audit_top_accel_memory_structure,
    )
    from verification.weight_audit_v2 import audit_rom_hex_images, audit_rom_module_source

    bundle_dir = Path(args.bundle).resolve()
    top_v = bundle_dir / "top_accel.v"
    if not top_v.is_file():
        print(f"[-] Bundle top_accel.v not found at: {top_v}", file=sys.stderr)
        return 1

    print(f"[*] V2 Bundle Dir  : {bundle_dir}")
    top_src = top_v.read_text(encoding="utf-8")
    repo_root = Path(__file__).resolve().parent
    rom_src = (repo_root / "rtl" / "lib" / "rom_sync.v").read_text(encoding="utf-8")

    # 1. Structural & ROM Immutability Audits
    try:
        rom_audit = audit_rom_module_source(rom_src, module_name="rom_sync")
        exp_audit = audit_no_expected_values_in_rtl(top_src, module_name="top_accel")
        mem_audit = audit_top_accel_memory_structure(top_src)
        hex_audit = audit_rom_hex_images(bundle_dir / "address_map.json")
    except ValueError as e:
        print(f"[-] V2 STRUCTURAL / ROM AUDIT FAILED: {e}", file=sys.stderr)
        return 1

    print("[+] ROM Sync Audit : PASS (0 write ports, synchronous $readmemh ROM)")
    print("[+] No-Oracle Audit: PASS (0 expected values or golden traces in RTL)")
    print(f"[+] Memory Audit   : PASS ({len(mem_audit['instantiated_roms'])} immutable ROMs, {len(mem_audit['found_buffers'])} authorized SRAMs)")
    print(f"[+] ROM Hex SHA256 : PASS ({len(hex_audit['checked_roms'])} packed ROM images verified)")

    # 2. Run token-exact RTL vs Interpreter check
    prompt_ids = [int(x.strip()) for x in args.tokens.split(",") if x.strip()] if args.tokens else [1, 10, 25]
    max_new = args.max_new_tokens

    interp = CompiledArtifactInterpreter(bundle_dir)
    interp.reset()
    golden_all: list[int] = []
    for pos in range(len(prompt_ids) + max_new):
        tok_in = prompt_ids[pos] if pos < len(prompt_ids) else golden_all[-1]
        _, tok_out = interp.decode_step(tok_in, pos=pos)
        golden_all.append(tok_out)

    rtl_res = _run_verilator_bundle_tokens(
        bundle_dir=bundle_dir,
        prompt_tokens=prompt_ids,
        max_new_tokens=max_new,
        work_dir=args.work_dir,
    )
    tokens_exact = rtl_res["all_tokens"] == golden_all
    status = "PASS" if tokens_exact else "FAIL"

    print(f"[+] RTL Sim Status : {status} (tokens={rtl_res['all_tokens']}, golden={golden_all}, cycles={rtl_res['total_cycles']:,})")

    out_report = Path(args.out).resolve()
    out_report.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "bundle": str(bundle_dir),
        "status": status,
        "prompt_tokens": prompt_ids,
        "golden_tokens": golden_all,
        "rtl_tokens": rtl_res["all_tokens"],
        "tokens_exact": tokens_exact,
        "total_cycles": rtl_res["total_cycles"],
        "sim_sec": rtl_res["sim_sec"],
        "audit": {
            "passed": True,
            "rom_audit": rom_audit,
            "no_expected_values_audit": exp_audit,
            "memory_structure_audit": mem_audit,
        },
    }
    out_report.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"[+] Report Saved   : {out_report}")
    return 0 if tokens_exact else 1


def cmd_verify(args: argparse.Namespace) -> int:
    """Execute 'llmhw verify': verify RTL against reference oracle and audit weights."""
    from backend.testbench import generate_wrapper_testbench, generate_wrapper_testbench_p
    from backend.tiling import TileManifest, tile_weight_matrix
    from backend.vectors import write_hex_vectors, write_weight_hex_vectors
    from compiler.quantization import quantize_weights
    from ir.linear import LinearOp
    from verification.reference import linear_reference_int64
    from verification.simulator import run_simulation, run_verilator_simulation
    from verification.weight_audit import audit_rtl_source, audit_weights_match

    print("================================================================================")
    print(" LLMHW VERIFY — BIT-EXACT SIMULATION ORACLE (CLAIM C1 / C1') & WEIGHT AUDIT")
    print("================================================================================")

    if getattr(args, "bundle", None) is not None:
        return _verify_v2_bundle(args)

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

    repo_root = Path(__file__).resolve().parent
    rtl_lib = repo_root / "rtl" / "lib"

    if getattr(args, "unit", None) is not None:
        unit_name = args.unit.removesuffix(".v")
        rtl_path = (rtl_lib / f"{unit_name}.v").resolve()
        if not rtl_path.is_file():
            print(f"[-] Part II RTL unit not found: {rtl_path}", file=sys.stderr)
            return 1
        top = args.top if args.top else unit_name
        extra_files = [rtl_lib / "softmax.v"] if unit_name == "attention" else []
        verilog_paths = [rtl_path] + extra_files
    elif args.rtl is not None:
        rtl_path = Path(args.rtl).resolve()
        if not rtl_path.is_file():
            print(f"[-] RTL file not found: {rtl_path}", file=sys.stderr)
            return 1
        top = args.top if args.top else rtl_path.stem
        verilog_paths = [rtl_path]
    else:
        print("[-] Either --rtl <path> or --unit <unit_name> must be provided.", file=sys.stderr)
        return 1

    flatten = not args.no_flatten

    print(f"[*] Top Module     : {top}")
    print(f"[*] RTL File       : {rtl_path}")
    print(f"[*] Target         : {args.target}")
    print(f"[*] Flatten        : {flatten}")

    # Inject -Irtl/lib into a temporary wrapper copy if the file uses `include "numerics.vh"`
    src_text = rtl_path.read_text(encoding="utf-8")
    if '`include "numerics.vh"' in src_text:
        num_vh = (rtl_lib / "numerics.vh").resolve()
        with tempfile.TemporaryDirectory(prefix="llmhw_synth_") as tmpdir:
            patched = Path(tmpdir) / rtl_path.name
            patched.write_text(src_text.replace('`include "numerics.vh"', f'`include "{num_vh}"'), encoding="utf-8")
            v_paths = [patched] + verilog_paths[1:]
            res = run_yosys_synthesis(
                top_module=top,
                verilog_paths=v_paths,
                target=args.target,
                flatten=flatten,
            )
    else:
        res = run_yosys_synthesis(
            top_module=top,
            verilog_paths=verilog_paths,
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
    """Construct argument parser for the llmhw CLI (v1 + v2)."""
    parser = argparse.ArgumentParser(
        prog="llmhw",
        description="LLMHW — Model-Specific Transformer Decoder & Linear Hardware Compiler (Part I & Part II)",
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

    # 2. compile (Part I linear layer)
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

    # 3. compile-model (Part II full decoder)
    p_cmodel = subparsers.add_parser(
        "compile-model",
        help="Compile full transformer decoder (tiny or SmolLM2-135M) to ROM-backed Verilog accelerator bundle",
    )
    p_cmodel.add_argument("--tiny", action="store_true", help="Compile deterministic 2-layer tiny decoder fixture")
    p_cmodel.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M", help="HuggingFace model ID")
    p_cmodel.add_argument("--revision", type=str, default="93efa2f097d58c2a74874c7e644dbc9b0cee75a2", help="Pinned model revision")
    p_cmodel.add_argument("--artifacts-dir", type=Path, default=None, help="Pre-extracted v2 artifacts directory")
    p_cmodel.add_argument("--force-extract", action="store_true", help="Force re-extraction and SmoothQuant calibration")
    p_cmodel.add_argument("--tile-r", type=int, default=8, help="Linear engine row lanes R")
    p_cmodel.add_argument("--tile-c", type=int, default=16, help="Linear engine column lanes C")
    p_cmodel.add_argument("--max-ctx", type=int, default=None, help="Override MAX_CTX sequence length")
    p_cmodel.add_argument("--out", type=Path, default=Path("build/hw_bundle"), help="Output hardware bundle directory")

    # 4. run (Part II autoregressive generation)
    p_run = subparsers.add_parser(
        "run",
        help="Run autoregressive token generation on a compiled v2 hardware bundle (interpreter or Verilator RTL)",
    )
    p_run.add_argument("--bundle", type=Path, default=Path("build/smollm2_hw"), help="Path to compiled v2 hardware bundle")
    p_run.add_argument("--prompt", type=str, default="The capital of France is Paris. The capital of Germany is", help="Input text prompt")
    p_run.add_argument("--tokens", type=str, default=None, help="Comma-separated token IDs (bypasses tokenizer, e.g. '1,10,25')")
    p_run.add_argument("--max-new-tokens", type=int, default=4, help="Number of new tokens to decode")
    p_run.add_argument("--backend", choices=["interpreter", "rtl", "both"], default="interpreter", help="Execution backend")
    p_run.add_argument("--out", type=Path, default=None, help="Optional JSON path to save execution report")

    # 5. verify (Part I linear + Part II v2 bundle)
    p_verify = subparsers.add_parser(
        "verify",
        help="Verify RTL against reference oracle and run static weight/ROM audit (supports --rtl or --bundle)",
    )
    p_verify.add_argument("--rtl", type=Path, default=None, help="Path to DUT Verilog file (Part I linear)")
    p_verify.add_argument("--bundle", type=Path, default=None, help="Path to compiled v2 decoder bundle (Part II)")
    p_verify.add_argument("--tokens", type=str, default="1,10,25", help="Comma-separated prompt token IDs for --bundle verification")
    p_verify.add_argument("--max-new-tokens", type=int, default=2, help="Number of tokens to decode for --bundle verification")
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

    # 6. synth (Part I linear + Part II library units)
    p_synth = subparsers.add_parser(
        "synth",
        help="Run Yosys logic synthesis and extract metrics (supports --rtl or --unit)",
    )
    p_synth.add_argument("--top", type=str, default=None, help="Top module name (auto-detected if None)")
    p_synth.add_argument("--rtl", type=Path, default=None, help="Path to synthesizable Verilog file")
    p_synth.add_argument("--unit", type=str, default=None, help="Part II library unit name in rtl/lib/ (e.g. rmsnorm, rope, swiglu)")
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
    if args.subcommand == "compile-model":
        return cmd_compile_model(args)
    if args.subcommand == "run":
        return cmd_run(args)
    if args.subcommand == "verify":
        return cmd_verify(args)
    if args.subcommand == "synth":
        return cmd_synth(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
