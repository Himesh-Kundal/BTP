"""Interface conformance tests and weight port audit for Design F (T4.6).

Statically inspects generated Verilog RTL to prove:
1. Zero runtime weight ports (w_data, w_we, w_addr, w_flat) exist.
2. Zero internal weight signals (w_*, weight_*) exist.
3. Zero writable memory arrays (reg ... [...:...]) exist.
4. No non-synthesizable constructs (# delays, $display) exist in hardware modules.
5. Wrapper interface strictly conforms to docs/02 §6.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np


def strip_verilog_comments(code: str) -> str:
    """Remove line comments (//) and block comments (/* */) from Verilog source."""
    # Remove block comments
    no_block = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    # Remove line comments
    clean = re.sub(r"//.*$", "", no_block, flags=re.MULTILINE)
    return clean


def audit_rtl_source(
    rtl_content: str,
    expected_k: int | None = None,
    expected_n: int | None = None,
    is_wrapper: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    """Perform static audit on Design F Verilog source code.

    Args:
        rtl_content: String containing Verilog code.
        expected_k: Expected in_features K (or padded K').
        expected_n: Expected out_features N (or padded N').
        is_wrapper: If True, inspects wrapper port conventions.
        strict: If True, raises ValueError upon any audit violation.

    Returns:
        Audit report dictionary with details and pass/fail verdict.

    Raises:
        ValueError: If strict is True and any violations are detected.
    """
    clean_code = strip_verilog_comments(rtl_content)
    violations: list[str] = []

    # 1. Audit for weight input ports
    weight_ports: list[str] = []
    input_matches = re.finditer(
        r"input\s+(?:wire\s+)?(?:\[[^\]]+\]\s+)?([a-zA-Z0-9_]+)",
        clean_code,
    )
    for m in input_matches:
        port_name = m.group(1)
        if re.match(r"^(w_|weight_)", port_name, re.IGNORECASE):
            weight_ports.append(port_name)
            violations.append(f"Weight port found in Design F RTL: '{port_name}'")

    # 2. Audit for internal w_* signals
    internal_w_signals: list[str] = []
    sig_matches = re.finditer(
        r"(?:wire|reg)\s+(?:signed\s+)?(?:\[[^\]]+\]\s+)?([a-zA-Z0-9_]+)",
        clean_code,
    )
    for m in sig_matches:
        sig_name = m.group(1)
        if re.match(r"^(w_|weight_)", sig_name, re.IGNORECASE):
            internal_w_signals.append(sig_name)
            violations.append(f"Forbidden w_* signal found: '{sig_name}'")

    # 3. Audit for writable memory storage (e.g. reg [7:0] mem [0:15])
    storage_matches = re.findall(
        r"reg\s+(?:signed\s+)?(?:\[[^\]]+\]\s+)?([a-zA-Z0-9_]+)\s*\[\s*\d+\s*:\s*\d+\s*\]\s*;",
        clean_code,
    )
    if storage_matches:
        for s in storage_matches:
            violations.append(f"Writable memory storage array found in Design F RTL: '{s}'")

    # 4. Audit for forbidden delays (#) and $display
    if re.search(r"#[0-9]+", clean_code):
        violations.append("Delays (#) are forbidden in synthesizable Design F RTL")
    if re.search(r"\$display", clean_code):
        violations.append("$display statements are forbidden in synthesizable Design F RTL")

    # 5. Interface conformance check (port widths)
    if is_wrapper:
        # Check x_flat width if expected_k provided
        if expected_k is not None:
            expected_x_width = 8 * expected_k
            x_match = re.search(r"x_flat\b", clean_code)
            if not x_match:
                violations.append("Missing required 'x_flat' input port")
            else:
                x_width_match = re.search(
                    rf"input\s+(?:wire\s+)?\[\s*({expected_x_width - 1})\s*:\s*0\s*\]\s+x_flat",
                    clean_code,
                )
                if not x_width_match:
                    violations.append(
                        f"Port 'x_flat' width does not match expected [{expected_x_width - 1}:0] for K={expected_k}"
                    )

        # Check y_flat width if expected_n provided
        if expected_n is not None:
            expected_y_width = 32 * expected_n
            y_match = re.search(r"y_flat\b", clean_code)
            if not y_match:
                violations.append("Missing required 'y_flat' output port")
            else:
                y_width_match = re.search(
                    rf"output\s+(?:wire|reg)\s+\[\s*({expected_y_width - 1})\s*:\s*0\s*\]\s+y_flat",
                    clean_code,
                )
                if not y_width_match:
                    violations.append(
                        f"Port 'y_flat' width does not match expected [{expected_y_width - 1}:0] for N={expected_n}"
                    )

    passed = len(violations) == 0

    if strict and not passed:
        raise ValueError(f"Design F RTL audit failed with violations: {'; '.join(violations)}")

    return {
        "passed": passed,
        "violations": violations,
        "weight_ports_found": len(weight_ports),
        "w_signals_found": len(internal_w_signals),
        "storage_found": len(storage_matches),
    }


def audit_rtl_file(
    file_path: Path | str,
    expected_k: int | None = None,
    expected_n: int | None = None,
    is_wrapper: bool = True,
    strict: bool = True,
) -> bool:
    """Audit a Verilog file on disk.

    Args:
        file_path: Path to .v file.
        expected_k: Expected in_features K.
        expected_n: Expected out_features N.
        is_wrapper: If True, inspects wrapper port conventions.
        strict: If True, raises ValueError upon failure.

    Returns:
        True if audit passes.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"RTL file not found: {path}")

    content = path.read_text(encoding="utf-8")
    report = audit_rtl_source(
        content,
        expected_k=expected_k,
        expected_n=expected_n,
        is_wrapper=is_wrapper,
        strict=strict,
    )
    return report["passed"]


def parse_rtl_weights(
    rtl_content: str,
    expected_n: int,
    expected_k: int,
    manifest: Any | None = None,
) -> np.ndarray:
    """Parse constant weight expressions from Verilog source and reconstruct W_RTL.

    Supports both:
    1. Multi-tile full modules containing sub-modules `llmhw_<layer>_tile_i{i}_j{j}`.
    2. Single-module combinational cores containing `prod_{row}_{col}` lines.

    Returns:
        w_recon: np.ndarray of shape (expected_n, expected_k) with dtype int8.
    """
    clean_code = strip_verilog_comments(rtl_content)

    # 1. Check for tile core modules
    tile_pattern = re.compile(
        r"module\s+(?P<mod_name>llmhw_\w+_tile_i(?P<i>\d+)_j(?P<j>\d+))\s*\((?P<ports>.*?)\);(?P<body>.*?)endmodule",
        re.DOTALL,
    )
    tile_matches = list(tile_pattern.finditer(clean_code))

    if tile_matches:
        if manifest is not None:
            tile_n = manifest.tile_n
            tile_k = manifest.tile_k
            n_padded = manifest.n_padded
            k_padded = manifest.k_padded
        else:
            # Infer tile dimensions from ports and max indices
            ports = tile_matches[0].group("ports")
            in_match = re.search(r"input\s+wire\s+\[(\d+):0\]\s+x_flat", ports)
            out_match = re.search(r"output\s+wire\s+\[(\d+):0\]\s+y_flat", ports)
            tile_k = (int(in_match.group(1)) + 1) // 8 if in_match else expected_k
            tile_n = (int(out_match.group(1)) + 1) // 32 if out_match else expected_n
            max_i = max(int(m.group("i")) for m in tile_matches)
            max_j = max(int(m.group("j")) for m in tile_matches)
            n_padded = (max_i + 1) * tile_n
            k_padded = (max_j + 1) * tile_k

        w_padded = np.zeros((n_padded, k_padded), dtype=np.int8)

        prod_pattern = re.compile(
            r"wire\s+signed\s+\[15:0\]\s+p_(?P<row>\d+)_(?P<col>\d+)\s*=\s*(?P<rhs>[^;]+);"
        )

        for tm in tile_matches:
            ti = int(tm.group("i"))
            tj = int(tm.group("j"))
            body = tm.group("body")

            for pm in prod_pattern.finditer(body):
                row = int(pm.group("row"))
                col = int(pm.group("col"))
                rhs = pm.group("rhs").strip()

                mult_pos = re.search(r"\*\s*8'sd(\d+)", rhs)
                mult_neg = re.search(r"\*\s*-8'sd(\d+)", rhs)

                if mult_pos:
                    w_val = int(mult_pos.group(1))
                elif mult_neg:
                    w_val = -int(mult_neg.group(1))
                elif rhs.startswith("-"):
                    w_val = -1
                elif "x_" in rhs and "*" not in rhs:
                    w_val = 1
                elif rhs == "16'sd0":
                    w_val = 0
                else:
                    raise ValueError(f"Unrecognized product expression in tile {ti},{tj}: {rhs}")

                w_padded[ti * tile_n + row, tj * tile_k + col] = w_val

        return w_padded[:expected_n, :expected_k]

    # 2. Single-module combinational core
    w_recon = np.zeros((expected_n, expected_k), dtype=np.int8)
    prod_single = re.compile(
        r"wire\s+signed\s+\[15:0\]\s+prod_(?P<row>\d+)_(?P<col>\d+)\s*=\s*(?P<rhs>[^;]+);"
    )
    for pm in prod_single.finditer(clean_code):
        row = int(pm.group("row"))
        col = int(pm.group("col"))
        rhs = pm.group("rhs").strip()

        mult_pos = re.search(r"\*\s*8'sd(\d+)", rhs)
        mult_neg = re.search(r"\*\s*-8'sd(\d+)", rhs)

        if mult_pos:
            w_val = int(mult_pos.group(1))
        elif mult_neg:
            w_val = -int(mult_neg.group(1))
        elif rhs.startswith("-"):
            w_val = -1
        elif "x_" in rhs and "*" not in rhs:
            w_val = 1
        elif rhs == "16'sd0":
            w_val = 0
        else:
            raise ValueError(f"Unrecognized product expression: {rhs}")

        w_recon[row, col] = w_val

    return w_recon


def audit_weights_match(
    rtl_content: str,
    expected_weights: np.ndarray,
    manifest: Any | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Audit static weights parsed from RTL against expected quantized array and SHA256.

    Args:
        rtl_content: Verilog source code.
        expected_weights: Golden INT8 weights array of shape (N, K).
        manifest: Optional TileManifest containing tile hashes.
        strict: If True, raises ValueError upon any discrepancy.

    Returns:
        Audit report dictionary with match details.
    """
    w_exp = np.asarray(expected_weights, dtype=np.int8)
    n, k = w_exp.shape

    w_recon = parse_rtl_weights(rtl_content, expected_n=n, expected_k=k, manifest=manifest)

    mismatches = int(np.sum(w_recon != w_exp))
    passed = mismatches == 0
    violations: list[str] = []

    if mismatches > 0:
        violations.append(
            f"Weight audit mismatch: {mismatches} elements differ between RTL and expected array."
        )

    # Integrity hash check
    extracted_sha256 = hashlib.sha256(w_recon.tobytes()).hexdigest()
    expected_sha256 = hashlib.sha256(w_exp.tobytes()).hexdigest()

    if extracted_sha256 != expected_sha256:
        passed = False
        violations.append(
            f"Weight SHA256 mismatch: extracted {extracted_sha256} != expected {expected_sha256}"
        )

    if (
        manifest is not None
        and getattr(manifest, "manifest_hash", None)
        and manifest.manifest_hash
        and manifest.manifest_hash not in rtl_content
    ):
        passed = False
        violations.append(
            f"Manifest hash '{manifest.manifest_hash}' not found in RTL header comment"
        )

    if strict and not passed:
        raise ValueError("; ".join(violations))

    return {
        "passed": passed,
        "mismatches": mismatches,
        "total_weights_checked": n * k,
        "reconstructed_weights": w_recon,
        "extracted_sha256": extracted_sha256,
        "violations": violations,
    }
