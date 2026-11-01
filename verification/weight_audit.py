"""Interface conformance tests and weight port audit for Design F (T4.6).

Statically inspects generated Verilog RTL to prove:
1. Zero runtime weight ports (w_data, w_we, w_addr, w_flat) exist.
2. Zero internal weight signals (w_*, weight_*) exist.
3. Zero writable memory arrays (reg ... [...:...]) exist.
4. No non-synthesizable constructs (# delays, $display) exist in hardware modules.
5. Wrapper interface strictly conforms to docs/02 §6.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


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
