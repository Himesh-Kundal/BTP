"""Mutation testing engine for generated RTL (M1-M9 per docs/06 §A6).

Fault models:
- M1: Flip sign of one weight constant (caught by random vectors)
- M2: Change one weight by +1 (caught by one-hot vectors)
- M3: Drop one product term from an accumulation line (caught by one-hot/random)
- M4: Swap two input lanes (caught by one-hot vectors)
- M5: Reverse packing order of outputs (caught by asymmetric vectors)
- M6: Truncate accumulator to 16 bits (caught by sign-aligned worst-case vectors)
- M7: Make one multiplication unsigned (caught by negative inputs)
- M8: Off-by-one in padding / misroute last lane (caught by non-multiple shapes)
- M9: Add one cycle of latency / delay valid_out (caught by timing check)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from verification.simulator import run_simulation


@dataclass
class Mutant:
    """Represents a mutated RTL instance."""

    id: str
    mutant_type: str  # "M1" through "M9"
    description: str
    mutated_code: str


def generate_mutants(rtl_code: str) -> list[Mutant]:
    """Generate all mutant classes M1 through M9 from generated RTL code."""
    mutants: list[Mutant] = []
    lines = rtl_code.splitlines()

    # Pattern for constant signed literals: * 8'sd<val> or * -8'sd<val>
    m1_pattern = re.compile(r"(\*\s*)(?P<sign>-?)8'sd(?P<val>\d+)")

    # -------------------------------------------------------------------------
    # 1. Mutant M1: Flip sign of one weight constant
    # -------------------------------------------------------------------------
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
            break  # Representative mutant per type for campaign efficiency

    # -------------------------------------------------------------------------
    # 2. Mutant M2: Change one weight by +1
    # -------------------------------------------------------------------------
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
            break

    # -------------------------------------------------------------------------
    # 3. Mutant M3: Drop one product term from an accumulation line
    # -------------------------------------------------------------------------
    accum_pattern = re.compile(
        r"wire\s+signed\s+\[(?P<bits>\d+):0\]\s+(?P<sig>y_\w+|y_reduced_\w+|y_\d+)\s*=\s*(?P<expr>[^;]+);"
    )
    for i, line in enumerate(lines):
        match = accum_pattern.search(line)
        if match:
            expr = match.group("expr")
            terms = [t.strip() for t in expr.split("+")]
            if len(terms) > 1:
                # Drop first term
                new_expr = " + ".join(terms[1:])
                bits = match.group("bits")
                sig = match.group("sig")
                new_line = f"    wire signed [{bits}:0] {sig} = {new_expr};"
                mut_lines = list(lines)
                mut_lines[i] = new_line
                mutants.append(
                    Mutant(
                        id=f"M3_line_{i + 1}_drop_term",
                        mutant_type="M3",
                        description=f"Line {i + 1}: dropped term '{terms[0]}' from sum",
                        mutated_code="\n".join(mut_lines),
                    )
                )
                break

    # -------------------------------------------------------------------------
    # 4. Mutant M4: Swap two input lanes
    # -------------------------------------------------------------------------
    # Matches x_0 / x_1 or x_tile_0 / x_tile_1 slicing
    lane_pattern = re.compile(
        r"wire\s+(?:signed\s+)?\[(?P<bits>[^\]]+)\]\s+(?P<name>x_\d+|x_tile_\d+)\s*=\s*x_flat\[(?P<slice>[^\]]+)\];"
    )
    lane_indices = []
    for i, line in enumerate(lines):
        m = lane_pattern.search(line)
        if m:
            lane_indices.append((i, m))

    if len(lane_indices) >= 2:
        idx0, m0 = lane_indices[0]
        idx1, m1 = lane_indices[1]
        mut_lines = list(lines)
        # Swap slice ranges between lane 0 and lane 1
        line0_swapped = re.sub(r"x_flat\[[^\]]+\]", f"x_flat[{m1.group('slice')}]", lines[idx0])
        line1_swapped = re.sub(r"x_flat\[[^\]]+\]", f"x_flat[{m0.group('slice')}]", lines[idx1])
        mut_lines[idx0] = line0_swapped
        mut_lines[idx1] = line1_swapped
        mutants.append(
            Mutant(
                id=f"M4_swap_{m0.group('name')}_{m1.group('name')}",
                mutant_type="M4",
                description=f"Swapped input slice assignments between {m0.group('name')} and {m1.group('name')}",
                mutated_code="\n".join(mut_lines),
            )
        )

    # -------------------------------------------------------------------------
    # 5. Mutant M5: Reverse packing order of outputs
    # -------------------------------------------------------------------------
    assign_pattern = re.compile(
        r"assign\s+(?P<dst>y_padded|y_flat)\[(?P<range>[^\]]+)\]\s*=\s*(?P<src>[^;]+);"
    )
    pack_indices = []
    for i, line in enumerate(lines):
        m = assign_pattern.search(line)
        if m:
            pack_indices.append((i, m))

    if len(pack_indices) >= 2:
        idx0, m0 = pack_indices[0]
        idx1, m1 = pack_indices[1]
        mut_lines = list(lines)
        line0_swapped = re.sub(r"=\s*[^;]+;", f"= {m1.group('src')};", lines[idx0])
        line1_swapped = re.sub(r"=\s*[^;]+;", f"= {m0.group('src')};", lines[idx1])
        mut_lines[idx0] = line0_swapped
        mut_lines[idx1] = line1_swapped
        mutants.append(
            Mutant(
                id="M5_reverse_output_pack",
                mutant_type="M5",
                description=f"Swapped packing sources between line {idx0 + 1} and line {idx1 + 1}",
                mutated_code="\n".join(mut_lines),
            )
        )

    # -------------------------------------------------------------------------
    # 6. Mutant M6: Truncate accumulator to 16 bits (strip sign extension)
    # -------------------------------------------------------------------------
    for i, line in enumerate(lines):
        if "{{16{" in line:
            # Replace 16-bit sign extension with zero extension (16-bit truncation)
            mut_line = re.sub(r"\{\{16\{[^}]+\}\},\s*([^}]+)\}", r"{16'b0, \1}", line, count=1)
            mut_lines = list(lines)
            mut_lines[i] = mut_line
            mutants.append(
                Mutant(
                    id=f"M6_line_{i + 1}_trunc_acc",
                    mutant_type="M6",
                    description=f"Line {i + 1}: replaced sign extension with zero extension (16-bit truncation)",
                    mutated_code="\n".join(mut_lines),
                )
            )
            break
        elif "wire signed [31:0] y_reduced_" in line or "wire signed [31:0] y_row_" in line:
            mut_line = line.replace("wire signed [31:0]", "wire signed [15:0]")
            mut_lines = list(lines)
            mut_lines[i] = mut_line
            mutants.append(
                Mutant(
                    id=f"M6_line_{i + 1}_acc_16bit",
                    mutant_type="M6",
                    description=f"Line {i + 1}: truncated accumulator reduction wire to signed [15:0]",
                    mutated_code="\n".join(mut_lines),
                )
            )
            break

    # -------------------------------------------------------------------------
    # 7. Mutant M7: Make one multiplication unsigned
    # -------------------------------------------------------------------------
    for i, line in enumerate(lines):
        if "8'sd" in line and "*" in line:
            # Replace signed 8'sd with unsigned 8'd
            mut_line = line.replace("8'sd", "8'd")
            mut_lines = list(lines)
            mut_lines[i] = mut_line
            mutants.append(
                Mutant(
                    id=f"M7_line_{i + 1}_unsigned_mult",
                    mutant_type="M7",
                    description=f"Line {i + 1}: made multiplication literal unsigned (8'sd -> 8'd)",
                    mutated_code="\n".join(mut_lines),
                )
            )
            break

    # -------------------------------------------------------------------------
    # 8. Mutant M8: Off-by-one in padding / last lane misrouted
    # -------------------------------------------------------------------------
    if lane_indices:
        last_idx, last_m = lane_indices[-1]
        _first_idx, first_m = lane_indices[0]
        mut_lines = list(lines)
        # Misroute last lane slice to point to first lane slice
        mut_lines[last_idx] = re.sub(
            r"x_flat\[[^\]]+\]",
            f"x_flat[{first_m.group('slice')}]",
            lines[last_idx],
        )
        mutants.append(
            Mutant(
                id=f"M8_misroute_last_lane_{last_m.group('name')}",
                mutant_type="M8",
                description=f"Misrouted last lane {last_m.group('name')} slice to {first_m.group('name')}",
                mutated_code="\n".join(mut_lines),
            )
        )

    # -------------------------------------------------------------------------
    # 9. Mutant M9: Add one cycle of latency / delay valid_out
    # -------------------------------------------------------------------------
    for i, line in enumerate(lines):
        if "valid_out <= valid_in;" in line:
            mut_lines = list(lines)
            # Delay valid_out handshake by suppressing cycle-1 assertion
            mut_lines[i] = "            valid_out <= 1'b0; // M9 latency fault"
            mutants.append(
                Mutant(
                    id=f"M9_line_{i + 1}_latency_fault",
                    mutant_type="M9",
                    description=f"Line {i + 1}: delayed valid_out handshake violating L=1 latency contract",
                    mutated_code="\n".join(mut_lines),
                )
            )
            break

    return mutants


def run_mutation_testing(
    dut_path: str | Path,
    tb_path: str | Path,
    num_vectors: int,
    work_dir: str | Path,
    rtl_code: str,
) -> dict[str, Any]:
    """Execute all mutants against the testbench and calculate kill rate."""
    return run_mutation_campaign(
        dut_path=dut_path,
        tb_path=tb_path,
        num_vectors=num_vectors,
        work_dir=work_dir,
        rtl_code=rtl_code,
    )


def run_mutation_campaign(
    dut_path: str | Path,
    tb_path: str | Path,
    num_vectors: int,
    work_dir: str | Path,
    rtl_code: str,
    matrix_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute all M1-M9 mutants against testbench and save results/mutation_matrix.json."""
    work_p = Path(work_dir)
    work_p.mkdir(parents=True, exist_ok=True)
    tb_p = Path(tb_path).resolve()

    mutants = generate_mutants(rtl_code)
    results = []
    killed = 0
    survived = 0

    dut_dir = Path(dut_path).parent

    for m in mutants:
        m_dir = work_p / m.id
        m_dir.mkdir(parents=True, exist_ok=True)
        m_dut = m_dir / "mutant.v"
        m_dut.write_text(m.mutated_code, encoding="utf-8")

        # Copy hex vector files into mutant simulation directory
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

        is_killed = (res.status == "FAIL") or (res.mismatch > 0) or (res.timing_err > 0)
        if is_killed:
            killed += 1
        else:
            survived += 1

        results.append(
            {
                "id": m.id,
                "type": m.mutant_type,
                "description": m.description,
                "killed": is_killed,
                "status": res.status,
                "mismatch_count": res.mismatch,
                "timing_err_count": res.timing_err,
            }
        )

    total = len(mutants)
    kill_rate = (killed / total) if total > 0 else 0.0

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "killed": killed,
        "survived": survived,
        "kill_rate": kill_rate,
        "classes_tested": sorted({m.mutant_type for m in mutants}),
        "mutants": results,
    }

    if matrix_output_path is not None:
        out_p = Path(matrix_output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary
