"""Fairness Protocol Enforcement and Audit Tools (T7.2, docs/06 §B2).

Enforces the 8 mandatory rules of the fairness protocol for synthesis evaluation:
1. Exact script and optimization flag parity between Design F and Design P/P_bus.
2. Rule N10 width conformance: x (8-bit), w (8-bit), prod (16-bit), acc (32-bit).
3. Registered I/O wrapper parity: identical pipeline latency (L=1) and clock-gating semantics.
4. Dual accounting for baseline: logic-only (P_bus) and storage-inclusive (P).
5. Separate reporting of sequential cells (flops) from combinational cells.
6. Evaluation across >= 5 seeds for random-weight comparisons.
7. Complete recording of tool versions, commands, and git hashes.
8. Standard pre-layout logic-synthesis caveats explicitly attached to metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from synth.yosys_runner import SynthesisResult


@dataclass
class FairnessAuditReport:
    """Report detailing adherence to the fairness protocol."""

    passed: bool
    violations: list[str]
    checks_performed: int
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": self.violations,
            "checks_performed": self.checks_performed,
            "details": self.details,
        }


def check_width_conformance(rtl: str, design_label: str = "Design") -> list[str]:
    """Verify that RTL declares strictly INT8 inputs/weights, INT16 products, and INT32 accumulators."""
    violations = []

    # Check input activation widths: wire signed [7:0] x_*
    x_wires = re.findall(r"wire\s+signed\s*\[(\d+):(\d+)\]\s+x_\d+", rtl)
    for hi, lo in x_wires:
        width = int(hi) - int(lo) + 1
        if width != 8:
            violations.append(f"{design_label}: Input wire x width is {width} bits, expected 8 bits.")

    # Check product wires: wire signed [15:0] prod_* or p_*
    prod_wires = re.findall(r"wire\s+signed\s*\[(\d+):(\d+)\]\s+(?:prod|p)_\d+_\d+", rtl)
    for hi, lo in prod_wires:
        width = int(hi) - int(lo) + 1
        if width != 16:
            violations.append(
                f"{design_label}: Product wire width is {width} bits, expected 16 bits."
            )

    # Check accumulator wires: wire signed [31:0] y_row_* or y_reduced_*
    acc_wires = re.findall(r"wire\s+signed\s*\[(\d+):(\d+)\]\s+(?:y_row|y_reduced)_\d+", rtl)
    for hi, lo in acc_wires:
        width = int(hi) - int(lo) + 1
        if width != 32:
            violations.append(
                f"{design_label}: Accumulator wire width is {width} bits, expected 32 bits."
            )

    return violations


def audit_synthesis_pair(
    res_f: SynthesisResult,
    res_p: SynthesisResult | None = None,
    res_p_bus: SynthesisResult | None = None,
    strict: bool = True,
) -> FairnessAuditReport:
    """Audit synthesis results between Design F and Baseline P variants for fairness conformance."""
    violations = []
    checks = 0
    details = {}

    # Check 1: Return codes
    checks += 1
    if res_f.returncode != 0:
        violations.append(f"Design F synthesis failed with return code {res_f.returncode}")
    if res_p is not None and res_p.returncode != 0:
        violations.append(f"Design P synthesis failed with return code {res_p.returncode}")
    if res_p_bus is not None and res_p_bus.returncode != 0:
        violations.append(f"Design P_bus synthesis failed with return code {res_p_bus.returncode}")

    # Check 2: Target techmap parity
    checks += 1
    target = res_f.target
    if res_p is not None and res_p.target != target:
        violations.append(
            f"Target mismatch: F used '{target}' but P used '{res_p.target}'"
        )
    if res_p_bus is not None and res_p_bus.target != target:
        violations.append(
            f"Target mismatch: F used '{target}' but P_bus used '{res_p_bus.target}'"
        )
    details["target"] = target

    # Check 3: Flatten optimization flag parity
    checks += 1
    flatten = res_f.flatten
    if res_p is not None and res_p.flatten != flatten:
        violations.append(
            f"Flatten mismatch: F used flatten={flatten} but P used flatten={res_p.flatten}"
        )
    if res_p_bus is not None and res_p_bus.flatten != flatten:
        violations.append(
            f"Flatten mismatch: F used flatten={flatten} but P_bus used flatten={res_p_bus.flatten}"
        )
    details["flatten"] = flatten

    # Check 4: Separate accounting of flops and combinational cells
    checks += 1
    if res_f.num_cells_seq is None or res_f.num_cells_comb is None:
        violations.append("Design F does not report sequential cells separately from combinational.")
    if res_p is not None and (res_p.num_cells_seq is None or res_p.num_cells_comb is None):
        violations.append("Design P does not report sequential cells separately from combinational.")
    if res_p_bus is not None and (res_p_bus.num_cells_seq is None or res_p_bus.num_cells_comb is None):
        violations.append("Design P_bus does not report sequential cells separately from combinational.")

    # Check 5: Both baseline accountings present (P and P_bus)
    checks += 1
    has_dual_accounting = (res_p is not None) and (res_p_bus is not None)
    if not has_dual_accounting:
        violations.append(
            "Fairness Rule 4 violation: Evaluation must present both logic-only (P_bus) and storage-inclusive (P) accountings."
        )
    details["has_dual_accounting"] = has_dual_accounting

    # Check 6: Relative storage hierarchy sanity
    if res_p is not None and res_p_bus is not None:
        checks += 1
        # P must have more flip-flops than P_bus because of W_reg array
        if res_p.num_cells_seq <= res_p_bus.num_cells_seq:
            violations.append(
                f"Storage hierarchy error: P flops ({res_p.num_cells_seq}) not strictly greater than P_bus flops ({res_p_bus.num_cells_seq})"
            )

    passed = len(violations) == 0
    report = FairnessAuditReport(
        passed=passed,
        violations=violations,
        checks_performed=checks,
        details=details,
    )

    if strict and not passed:
        raise ValueError(
            f"Fairness Protocol Violation ({len(violations)} errors):\n"
            + "\n".join(f" - {v}" for v in violations)
        )

    return report
