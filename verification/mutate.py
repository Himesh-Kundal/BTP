"""Mutation testing engine for generated RTL (M1-M3 for Phase 1).

Per spec docs/06 §A6:
- M1: Flip sign of one weight constant
- M2: Change one weight by +1
- M3: Drop one product term from an accumulation line
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verification.simulator import run_simulation


@dataclass
class Mutant:
    """Represents a mutated RTL instance."""

    id: str
    mutant_type: str  # e.g. "M1", "M2", "M3"
    description: str
    mutated_code: str


def generate_mutants(rtl_code: str) -> list[Mutant]:
    """Generate mutants M1, M2, and M3 from generated RTL code."""
    mutants: list[Mutant] = []
    lines = rtl_code.splitlines()

    # 1. Mutant M1: Flip sign of one weight constant
    # Matches: * 8'sd<val> or * -8'sd<val>
    m1_pattern = re.compile(r"(\*\s*)(?P<sign>-?)8'sd(?P<val>\d+)")
    for i, line in enumerate(lines):
        match = m1_pattern.search(line)
        if match:
            is_neg = match.group("sign") == "-"
            val = match.group("val")
            new_sign = "" if is_neg else "-"
            new_lit = f"*{match.group(1).replace('*', '')}{new_sign}8'sd{val}"
            mut_lines = list(lines)
            mut_lines[i] = m1_pattern.sub(new_lit, line, count=1)
            mutants.append(
                Mutant(
                    id=f"M1_line_{i + 1}",
                    mutant_type="M1",
                    description=f"Line {i + 1}: flipped weight sign to {new_sign}8'sd{val}",
                    mutated_code="\n".join(mut_lines),
                )
            )

    # 2. Mutant M2: Change one weight by +1
    for i, line in enumerate(lines):
        match = m1_pattern.search(line)
        if match:
            is_neg = match.group("sign") == "-"
            num = int(match.group("val"))
            actual_val = -num if is_neg else num
            mutated_val = actual_val + 1
            if mutated_val >= 0:
                new_lit = f"*{match.group(1).replace('*', '')}8'sd{mutated_val}"
            else:
                new_lit = f"*{match.group(1).replace('*', '')}-8'sd{abs(mutated_val)}"
            mut_lines = list(lines)
            mut_lines[i] = m1_pattern.sub(new_lit, line, count=1)
            mutants.append(
                Mutant(
                    id=f"M2_line_{i + 1}",
                    mutant_type="M2",
                    description=f"Line {i + 1}: changed weight from {actual_val} to {mutated_val}",
                    mutated_code="\n".join(mut_lines),
                )
            )

    # 3. Mutant M3: Drop one product term from an accumulation line
    # Matches: wire signed [31:0] y_\d+ = term1 + term2 + ...;
    accum_pattern = re.compile(r"wire signed \[31:0\] y_\d+\s*=\s*(?P<expr>[^;]+);")
    for i, line in enumerate(lines):
        match = accum_pattern.search(line)
        if match:
            expr = match.group("expr")
            terms = [t.strip() for t in expr.split("+")]
            if len(terms) > 1:
                for t_idx in range(len(terms)):
                    remaining = [t for idx, t in enumerate(terms) if idx != t_idx]
                    new_expr = " + ".join(remaining)
                    new_line = accum_pattern.sub(f"wire signed [31:0] y_0 = {new_expr};", line)
                    mut_lines = list(lines)
                    mut_lines[i] = new_line
                    mutants.append(
                        Mutant(
                            id=f"M3_line_{i + 1}_drop_{t_idx}",
                            mutant_type="M3",
                            description=f"Line {i + 1}: dropped term '{terms[t_idx]}'",
                            mutated_code="\n".join(mut_lines),
                        )
                    )

    return mutants


def run_mutation_testing(
    dut_path: str | Path,
    tb_path: str | Path,
    num_vectors: int,
    work_dir: str | Path,
    rtl_code: str,
) -> dict[str, Any]:
    """Execute all mutants against the testbench and calculate kill rate.

    A mutant is KILLED if the testbench returns status == 'FAIL' (mismatch > 0).
    A mutant SURVIVES if the testbench returns status == 'PASS'.
    """
    work_p = Path(work_dir)
    work_p.mkdir(parents=True, exist_ok=True)
    tb_p = Path(tb_path).resolve()

    mutants = generate_mutants(rtl_code)
    results = []
    killed = 0
    survived = 0

    for m in mutants:
        m_dir = work_p / m.id
        m_dir.mkdir(parents=True, exist_ok=True)
        m_dut = m_dir / "mutant.v"
        m_dut.write_text(m.mutated_code, encoding="utf-8")

        # Copy hex files if present in dut directory
        dut_dir = Path(dut_path).parent
        for hex_file in dut_dir.glob("*.hex"):
            target_hex = m_dir / hex_file.name
            if not target_hex.exists():
                target_hex.write_bytes(hex_file.read_bytes())

        res = run_simulation(
            dut_path=m_dut,
            tb_path=tb_p,
            expected_checked=num_vectors,
            work_dir=m_dir,
            dump_vcd=False,
        )

        is_killed = (res.status == "FAIL") or (res.mismatch > 0)
        if is_killed:
            killed += 1
        else:
            survived += 1

        results.append({
            "id": m.id,
            "type": m.mutant_type,
            "description": m.description,
            "killed": is_killed,
            "mismatch_count": res.mismatch,
        })

    total = len(mutants)
    kill_rate = (killed / total) if total > 0 else 0.0

    return {
        "total": total,
        "killed": killed,
        "survived": survived,
        "kill_rate": kill_rate,
        "mutants": results,
    }
