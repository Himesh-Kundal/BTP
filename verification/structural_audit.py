"""Structural and Determinism Audits for Top-Level Accelerator (Task UD.6).

Enforces:
- Golden Rule 7: Immutable Weights (all weights/tables in ROMs with zero write ports)
- Golden Rule 9: Compiler Outputs, Not Oracle Outputs (no tokens, activations, or expected values in RTL)
- Golden Rule 11: Determinism (identical inputs yield byte-identical RTL and artifacts)
- Verification Contract docs/part2/06 §A7:
  1. Weight audit v2
  2. ROM interface test
  3. No-expected-values audit
  4. Determinism check
  5. Multi-tool lint (Icarus, Verilator, Yosys)

Strict Layering (Golden Rule 6):
Pure Python, standard library, and IR. No torch or transformers.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any

from verification.weight_audit_v2 import strip_verilog_comments


# Authorized dynamic memory buffers in top_accel
AUTHORIZED_DYNAMIC_BUFFERS = {
    "buf_res",
    "buf_act",
    "buf_q",
    "buf_k",
    "buf_v",
    "buf_attn_out",
    "buf_gate",
    "buf_up",
    "buf_swiglu",
    "buf_logits",
}

FORBIDDEN_ORACLE_PATTERNS = [
    r"\bexp(?:ected)?_tokens?\b",
    r"\bgolden_traces?\b",
    r"\boracle_tokens?\b",
    r"\bexp_mem\b",
    r"\bgolden_mem\b",
    r"\btb_exp\b",
]


def audit_no_expected_values_in_rtl(rtl_content: str, module_name: str = "top_accel") -> dict[str, Any]:
    """Verify generated RTL contains no oracle tokens, activations, or expected outputs (Rule 9).
    
    Args:
        rtl_content: Verilog source code.
        module_name: Name of module being audited.
        
    Returns:
        Audit report dict.
        
    Raises:
        ValueError: If forbidden oracle symbols or hardcoded output sequences are found.
    """
    clean = strip_verilog_comments(rtl_content)
    violations: list[str] = []

    for pat in FORBIDDEN_ORACLE_PATTERNS:
        matches = re.findall(pat, clean, re.IGNORECASE)
        if matches:
            violations.append(f"Forbidden oracle signal or memory pattern detected: '{matches[0]}'")

    # Check for hardcoded token sequence arrays (e.g. reg [15:0] tokens [0:...])
    tok_array_matches = re.findall(r"reg\s+(?:\[[^\]]+\]\s+)?(tokens?|golden|expected)\s*\[\s*\d+\s*:\s*\d+\s*\]", clean, re.IGNORECASE)
    if tok_array_matches:
        for m in tok_array_matches:
            violations.append(f"Forbidden hardcoded token array detected in RTL: '{m}'")

    # Check for $readmem loading expected files
    readmem_matches = re.findall(r"\$readmem[hb]\s*\(\s*\"([^\"]+)\"", clean)
    for f in readmem_matches:
        f_lower = f.lower()
        if any(term in f_lower for term in ["exp", "golden", "oracle", "expected", "trace"]):
            violations.append(f"Forbidden readmem of oracle file in RTL: '{f}'")

    if violations:
        msg = f"No-Expected-Values Audit failed for {module_name}:\n" + "\n".join(f"  - {v}" for v in violations)
        raise ValueError(msg)

    return {
        "status": "PASS",
        "module": module_name,
        "violations": [],
    }


def audit_top_accel_memory_structure(rtl_content: str) -> dict[str, Any]:
    """Verify top_accel memory architecture conforms to Rule 7 (Immutable Weights).
    
    1. All instantiated ROMs must be rom_sync with no write ports.
    2. All internal 2D memory arrays must belong to AUTHORIZED_DYNAMIC_BUFFERS.
    
    Args:
        rtl_content: Verilog code of top_accel.v.
        
    Returns:
        Audit report dict.
        
    Raises:
        ValueError: If unauthorized writable memories or ROM write ports are found.
    """
    clean = strip_verilog_comments(rtl_content)
    violations: list[str] = []

    # 1. Audit internal memory declarations
    mem_decls = re.finditer(
        r"reg\s+(?:signed\s+)?(?:\[[^\]]+\]\s+)?([a-zA-Z0-9_]+)\s*\[\s*0\s*:\s*([^\]]+)\s*\]\s*;",
        clean,
    )
    found_buffers = set()
    for m in mem_decls:
        buf_name = m.group(1)
        found_buffers.add(buf_name)
        if buf_name not in AUTHORIZED_DYNAMIC_BUFFERS:
            violations.append(f"Unauthorized writable memory array declared: '{buf_name}'")

    # 2. Audit rom_sync instantiations
    rom_inst_blocks = re.findall(r"rom_sync\s*#\s*\((.*?)\)\s*([a-zA-Z0-9_]+)\s*\((.*?)\);", clean, flags=re.DOTALL)
    instantiated_roms = []
    for params, inst_name, ports in rom_inst_blocks:
        instantiated_roms.append(inst_name)
        # Check ports for any write signals
        port_connections = re.findall(r"\.([a-zA-Z0-9_]+)\s*\(", ports)
        for p in port_connections:
            if any(term in p.lower() for term in ["we", "wr", "write", "din", "wdata"]):
                violations.append(f"Forbidden write port '{p}' connected to ROM instance '{inst_name}'")

    # Ensure required ROM instances exist
    required_roms = {
        "prog_rom_inst",
        "layer_bases_rom_inst",
        "rom_weights_inst",
        "rom_params_inst",
        "rom_rope_cos_inst",
        "rom_rope_sin_inst",
        "rom_rsqrt_inst",
        "rom_exp2_inst",
        "rom_silu_inst",
    }
    missing_roms = required_roms - set(instantiated_roms)
    if missing_roms:
        violations.append(f"Missing required ROM instances in top_accel: {sorted(missing_roms)}")

    if violations:
        msg = "Top Accelerator Memory Audit failed:\n" + "\n".join(f"  - {v}" for v in violations)
        raise ValueError(msg)

    return {
        "status": "PASS",
        "found_buffers": sorted(found_buffers),
        "instantiated_roms": instantiated_roms,
        "violations": [],
    }


def audit_generator_determinism(
    config: dict[str, Any],
    rom_files: dict[str, str],
) -> dict[str, Any]:
    """Verify that generating top accelerator RTL twice yields byte-identical code (Rule 11).
    
    Args:
        config: Model configuration.
        rom_files: ROM hex file mappings.
        
    Returns:
        Audit report dict.
        
    Raises:
        ValueError: If outputs differ by even 1 byte.
    """
    from backend.top_generator import generate_top_accel_verilog

    code1 = generate_top_accel_verilog(config, rom_files=rom_files)
    code2 = generate_top_accel_verilog(config, rom_files=rom_files)

    sha1 = hashlib.sha256(code1.encode("utf-8")).hexdigest()
    sha2 = hashlib.sha256(code2.encode("utf-8")).hexdigest()

    if sha1 != sha2 or code1 != code2:
        raise ValueError(f"Determinism violation: top_accel generation differed between runs (sha1={sha1}, sha2={sha2})")

    return {
        "status": "PASS",
        "sha256": sha1,
        "byte_length": len(code1.encode("utf-8")),
    }


def run_multi_tool_lint(
    rtl_files: list[Path | str] | None = None,
    top_module: str = "top_accel",
    inc_dir: Path | str = "rtl/lib",
) -> dict[str, Any]:
    """Run lint cleanliness checks across Icarus, Verilator, and Yosys.
    
    Args:
        rtl_files: List of Verilog source files. If None, uses canonical rtl/lib files.
        top_module: Name of top-level module.
        inc_dir: Directory containing include files (numerics.vh).
        
    Returns:
        Structured dict with returncodes and outputs from each tool.
        
    Raises:
        RuntimeError: If any linter fails with a nonzero exit code.
    """
    if rtl_files is None:
        lib_dir = Path("rtl/lib")
        rtl_files = [
            lib_dir / "top_accel.v",
            lib_dir / "rom_sync.v",
            lib_dir / "sequencer.v",
            lib_dir / "embed_gather.v",
            lib_dir / "rmsnorm.v",
            lib_dir / "rope.v",
            lib_dir / "attention.v",
            lib_dir / "residual_add.v",
            lib_dir / "swiglu.v",
            lib_dir / "argmax.v",
            lib_dir / "linear_engine.v",
            lib_dir / "kv_cache.v",
            lib_dir / "softmax.v",
        ]

    file_strs = [str(f) for f in rtl_files]

    # 1. Icarus Verilog Lint
    cmd_iverilog = ["iverilog", "-Wall", "-g2005", f"-I{inc_dir}", f"-s{top_module}"] + file_strs
    res_iv = subprocess.run(cmd_iverilog, capture_output=True, text=True)
    if res_iv.returncode != 0:
        raise RuntimeError(f"Icarus Verilog lint failed (rc={res_iv.returncode}):\n{res_iv.stderr}\n{res_iv.stdout}")

    # 2. Verilator Lint
    cmd_verilator = [
        "verilator",
        "--lint-only",
        "-Wall",
        "-Wno-DECLFILENAME",
        "-Wno-UNUSEDSIGNAL",
        "-Wno-UNDRIVEN",
        "-Wno-SYNCASYNCNET",
        "-Wno-WIDTHEXPAND",
        "-Wno-WIDTHTRUNC",
        "-Wno-BLKSEQ",
        f"-I{inc_dir}",
        f"--top-module",
        top_module,
    ] + file_strs
    res_ver = subprocess.run(cmd_verilator, capture_output=True, text=True)
    if res_ver.returncode != 0:
        raise RuntimeError(f"Verilator lint failed (rc={res_ver.returncode}):\n{res_ver.stderr}\n{res_ver.stdout}")

    # 3. Yosys Hierarchy & Proc Check
    yosys_script = (
        f"read_verilog -I{inc_dir} {' '.join(file_strs)}; "
        f"hierarchy -check -top {top_module}; "
        f"proc; "
        f"check -assert"
    )
    cmd_yosys = ["yosys", "-q", "-p", yosys_script]
    res_yo = subprocess.run(cmd_yosys, capture_output=True, text=True)
    if res_yo.returncode != 0:
        raise RuntimeError(f"Yosys check failed (rc={res_yo.returncode}):\n{res_yo.stderr}\n{res_yo.stdout}")

    return {
        "status": "PASS",
        "iverilog": {"returncode": res_iv.returncode, "stdout": res_iv.stdout, "stderr": res_iv.stderr},
        "verilator": {"returncode": res_ver.returncode, "stdout": res_ver.stdout, "stderr": res_ver.stderr},
        "yosys": {"returncode": res_yo.returncode, "stdout": res_yo.stdout, "stderr": res_yo.stderr},
    }
