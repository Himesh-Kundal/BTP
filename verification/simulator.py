"""Simulation runner and report generator (verification/simulator.py)."""

import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SimResult:
    """Structured result of an RTL simulation run."""

    status: str  # "PASS" or "FAIL"
    checked: int
    mismatch: int
    timing_err: int
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    report_path: str | None = None


def run_simulation(
    dut_path: str | Path,
    tb_path: str | Path,
    expected_checked: int,
    work_dir: str | Path,
    dump_vcd: bool = False,
    report_file: str | Path | None = None,
) -> SimResult:
    """Compile and simulate RTL using Icarus Verilog, enforcing the CHECKED rule.

    Per spec docs/02 §7 and docs/06 §A8:
    - Compiles DUT and TB with iverilog -g2005 -Wall.
    - Executes with vvp.
    - Enforces:
        1. TB_RESULT line parsed correctly.
        2. checked == expected_checked and expected_checked > 0.
        3. mismatch == 0.
        4. timing_err == 0.
        5. returncode == 0.
    - Writes report_file (JSON) if specified.
    """
    dut_p = Path(dut_path).resolve()
    tb_p = Path(tb_path).resolve()
    work_p = Path(work_dir).resolve()
    work_p.mkdir(parents=True, exist_ok=True)

    sim_bin = work_p / "sim.vvp"
    compile_cmd = [
        "iverilog",
        "-g2005",
        "-Wall",
        "-o",
        str(sim_bin),
        str(dut_p),
        str(tb_p),
    ]

    start_time = time.perf_counter()
    compile_res = subprocess.run(
        compile_cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(work_p),
    )

    if compile_res.returncode != 0:
        elapsed = time.perf_counter() - start_time
        res = SimResult(
            status="FAIL",
            checked=0,
            mismatch=0,
            timing_err=0,
            returncode=compile_res.returncode,
            stdout=compile_res.stdout,
            stderr=f"Compilation error:\n{compile_res.stderr}",
            elapsed_seconds=elapsed,
        )
        if report_file is not None:
            _write_report(res, Path(report_file))
        return res

    # Run simulation with vvp
    sim_cmd = ["vvp", str(sim_bin)]
    if dump_vcd:
        sim_cmd.append("+dump")

    sim_res = subprocess.run(
        sim_cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(work_p),
    )
    elapsed = time.perf_counter() - start_time

    stdout = sim_res.stdout
    stderr = sim_res.stderr

    # Parse TB_RESULT line
    # Format: TB_RESULT CHECKED=<n> MISMATCH=<m> TIMING_ERR=<t>
    match = re.search(
        r"TB_RESULT\s+CHECKED=(?P<checked>\d+)\s+MISMATCH=(?P<mismatch>\d+)\s+TIMING_ERR=(?P<timing>\d+)",
        stdout,
    )

    if not match:
        status = "FAIL"
        checked = 0
        mismatch = 1
        timing_err = 0
        err_msg = "Missing or malformed TB_RESULT summary line in testbench output."
        stderr = f"{stderr}\n{err_msg}" if stderr else err_msg
    else:
        checked = int(match.group("checked"))
        mismatch = int(match.group("mismatch"))
        timing_err = int(match.group("timing"))

        # Enforce pass rules
        if (
            sim_res.returncode == 0
            and mismatch == 0
            and timing_err == 0
            and checked == expected_checked
            and expected_checked > 0
        ):
            status = "PASS"
        else:
            status = "FAIL"

    result = SimResult(
        status=status,
        checked=checked,
        mismatch=mismatch,
        timing_err=timing_err,
        returncode=sim_res.returncode,
        stdout=stdout,
        stderr=stderr,
        elapsed_seconds=elapsed,
        report_path=str(report_file) if report_file else None,
    )

    if report_file is not None:
        _write_report(result, Path(report_file))

    return result


def _write_report(res: SimResult, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(asdict(res), f, indent=2)
