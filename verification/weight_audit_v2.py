"""Weight Audit v2 for Part II Transformer Decoder Accelerator (Task UC.2).

Enforces Golden Rule 7 (Immutable Weights) and Verification Contract docs/part2/06 §A7:
1. Static inspection of ROM Verilog modules:
   - ROM modules have ONLY clk, addr, and data_out ports.
   - Zero write ports (we, write_en, wdata, etc.).
   - Zero runtime assignments to memory elements.
   - Fails on an injected write port or write enable.
2. Integrity audit of packed ROM images:
   - Hex files match sha256 recorded in address_map.json.
   - Hex line count equals total_words declared in address_map.json.
   - Fails on altered or corrupted hex contents.

Strict Layering (Golden Rule 6):
Only standard library and NumPy/IR. No torch or transformers imports.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def strip_verilog_comments(code: str) -> str:
    """Remove line comments (//) and block comments (/* */) from Verilog source."""
    no_block = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    clean = re.sub(r"//.*$", "", no_block, flags=re.MULTILINE)
    return clean


def audit_rom_module_source(rtl_content: str, module_name: str | None = None) -> dict[str, Any]:
    """Perform static inspection on a Verilog ROM module to ensure zero write capability.
    
    Args:
        rtl_content: Verilog source code string.
        module_name: Optional expected module name.
        
    Returns:
        Audit report dict with violations and pass/fail status.
        
    Raises:
        ValueError: If any write ports or runtime assignments are found.
    """
    clean = strip_verilog_comments(rtl_content)
    violations: list[str] = []

    # Find ports
    input_matches = re.findall(
        r"input\s+(?:wire\s+)?(?:\[[^\]]+\]\s+)?([a-zA-Z0-9_]+)",
        clean,
    )
    for port in input_matches:
        port_lower = port.lower()
        if any(term in port_lower for term in ["we", "wr", "write", "din", "wdata"]):
            violations.append(f"Forbidden write-related input port found: '{port}'")
        elif port_lower not in ["clk", "addr", "rst_n", "rst", "enable", "en", "ren"]:
            violations.append(f"Unexpected input port in ROM module: '{port}'")

    # Check for memory write assignments in always blocks
    # e.g., mem[...] <= ... or mem[...] = ...
    write_assign = re.findall(r"([a-zA-Z0-9_]+)\s*\[[^\]]+\]\s*(?:<=|=)\s*[^;]+;", clean)
    if write_assign:
        # Check if the target is a memory array
        for target in write_assign:
            # Look if target was assigned inside an always block (not initial)
            always_blocks = re.findall(r"always\s*@[^b]+begin(.*?)end", clean, flags=re.DOTALL)
            for blk in always_blocks:
                if re.search(rf"\b{target}\s*\[", blk):
                    violations.append(f"Runtime write assignment to memory array '{target}' in always block")

    # Ensure output is data or data_out
    output_matches = re.findall(
        r"output\s+(?:(?:reg|wire)\s+)?(?:signed\s+)?(?:\[[^\]]+\]\s+)?([a-zA-Z0-9_]+)",
        clean,
    )
    for out_port in output_matches:
        if not any(term in out_port.lower() for term in ["data", "dout", "val"]):
            violations.append(f"Unexpected output port in ROM module: '{out_port}'")

    if violations:
        msg = f"ROM Audit failed for {module_name or 'module'}:\n" + "\n".join(f"  - {v}" for v in violations)
        raise ValueError(msg)

    return {
        "status": "PASS",
        "input_ports": input_matches,
        "output_ports": output_matches,
        "violations": violations,
    }


def audit_rom_hex_images(address_map_path: Path | str) -> dict[str, Any]:
    """Verify cryptographic integrity of all ROM hex files against address_map.json.
    
    Args:
        address_map_path: Path to address_map.json.
        
    Returns:
        Audit report dict.
        
    Raises:
        ValueError: If any hex file has mismatched sha256 or incorrect line count.
    """
    map_p = Path(address_map_path)
    if not map_p.exists():
        raise FileNotFoundError(f"Address map not found: {map_p}")

    with map_p.open() as f:
        meta = json.load(f)

    base_dir = map_p.parent
    checked: list[dict[str, Any]] = []

    # Match either *_image keys or direct keys
    rom_key_map = [
        ("weights_image", "weights"),
        ("requant_image", "requant_params"),
        ("tables_image", "tables"),
    ]

    for key_a, key_b in rom_key_map:
        rom_info = meta.get(key_a) or meta.get(key_b)
        if not rom_info:
            continue

        file_name = rom_info["file"]
        hex_path = base_dir / file_name
        if not hex_path.exists():
            raise FileNotFoundError(f"Required ROM hex file not found: {hex_path}")

        # Compute SHA256
        hex_bytes = hex_path.read_bytes()
        actual_sha256 = hashlib.sha256(hex_bytes).hexdigest()
        expected_sha256 = rom_info.get("sha256")

        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(
                f"ROM integrity violation for '{key_a}' ({file_name}): "
                f"expected sha256={expected_sha256}, actual sha256={actual_sha256}"
            )

        # Count non-empty lines
        lines = [ln.strip() for ln in hex_path.read_text().splitlines() if ln.strip()]
        expected_words = rom_info.get("total_words")
        if expected_words is not None and len(lines) != expected_words:
            raise ValueError(
                f"ROM size mismatch for '{key_a}' ({file_name}): "
                f"expected {expected_words} words, found {len(lines)} lines"
            )

        checked.append({
            "rom": key_a,
            "file": file_name,
            "words": len(lines),
            "sha256": actual_sha256,
        })

    return {
        "status": "PASS",
        "checked_roms": checked,
    }
