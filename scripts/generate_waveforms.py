"""Generate VCD waveforms for passing and failing testbench runs (T6.7).

Produces:
1. artifacts/waveforms/pass.vcd: Clean passing transaction (L=1 pipeline)
2. artifacts/waveforms/fail_m1_sign.vcd: Arithmetic mismatch (M1 sign-flip mutant)
3. artifacts/waveforms/fail_m9_timing.vcd: Protocol violation (M9 latency violation)
"""

import shutil
from pathlib import Path

import numpy as np

from backend.module_generator import generate_full_module
from backend.testbench import generate_wrapper_testbench
from backend.tiling import tile_weight_matrix
from backend.vectors import write_hex_vectors
from ir.linear import LinearOp
from verification.mutate import generate_mutants
from verification.reference import linear_reference_int64
from verification.simulator import run_simulation


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    art_dir = repo_root / "artifacts" / "waveforms"
    art_dir.mkdir(parents=True, exist_ok=True)

    w = np.array([[12, -7, 0, 15], [-3, 8, 22, -1]], dtype=np.int8)
    op = LinearOp(name="waveform_demo", in_features=4, out_features=2, weight=w)
    manifest, _ = tile_weight_matrix(w, tile_n=2, tile_k=2)

    x = np.array(
        [
            [10, 20, 30, 40],
            [127, -128, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    y_exp = linear_reference_int64(x, w)

    # 1. Passing run
    pass_dir = art_dir / "run_pass"
    pass_dir.mkdir(parents=True, exist_ok=True)
    x_hex, y_hex = write_hex_vectors(x, y_exp, pass_dir)
    rtl_clean = generate_full_module(op, manifest, latency=1)
    dut_clean = pass_dir / "dut.v"
    dut_clean.write_text(rtl_clean, encoding="utf-8")
    tb_clean = pass_dir / "tb.v"
    tb_clean.write_text(
        generate_wrapper_testbench(
            op=op,
            num_vectors=len(x),
            x_hex_path=str(x_hex),
            y_hex_path=str(y_hex),
            module_name="llmhw_waveform_demo",
            manifest=manifest,
            latency=1,
        ),
        encoding="utf-8",
    )
    rep_pass = run_simulation(dut_clean, tb_clean, expected_checked=len(x), work_dir=pass_dir, dump_vcd=True)
    print(f"Pass run status: {rep_pass.status} (checked={rep_pass.checked}, mismatch={rep_pass.mismatch})")
    vcd_pass = pass_dir / "llmhw_waveform_demo.vcd"
    if vcd_pass.exists():
        shutil.copy2(vcd_pass, art_dir / "pass.vcd")
        print(f"Copied {vcd_pass.name} -> artifacts/waveforms/pass.vcd ({vcd_pass.stat().st_size} bytes)")

    # 2. Failing run: M1 (Sign flip)
    m1_dir = art_dir / "run_m1_fail"
    m1_dir.mkdir(parents=True, exist_ok=True)
    write_hex_vectors(x, y_exp, m1_dir)
    mutants = generate_mutants(rtl_clean)
    m1_mut = next(m for m in mutants if m.mutant_type == "M1")
    dut_m1 = m1_dir / "dut.v"
    dut_m1.write_text(m1_mut.mutated_code, encoding="utf-8")
    tb_m1 = m1_dir / "tb.v"
    tb_m1.write_text(
        generate_wrapper_testbench(
            op=op,
            num_vectors=len(x),
            x_hex_path=str(m1_dir / "x.hex"),
            y_hex_path=str(m1_dir / "y_exp.hex"),
            module_name="llmhw_waveform_demo",
            manifest=manifest,
            latency=1,
        ),
        encoding="utf-8",
    )
    rep_m1 = run_simulation(dut_m1, tb_m1, expected_checked=len(x), work_dir=m1_dir, dump_vcd=True)
    print(f"M1 run status: {rep_m1.status} (checked={rep_m1.checked}, mismatch={rep_m1.mismatch})")
    vcd_m1 = m1_dir / "llmhw_waveform_demo.vcd"
    if vcd_m1.exists():
        shutil.copy2(vcd_m1, art_dir / "fail_m1_sign.vcd")
        print(f"Copied {vcd_m1.name} -> artifacts/waveforms/fail_m1_sign.vcd ({vcd_m1.stat().st_size} bytes)")

    # 3. Failing run: M9 (Latency violation)
    m9_dir = art_dir / "run_m9_fail"
    m9_dir.mkdir(parents=True, exist_ok=True)
    write_hex_vectors(x, y_exp, m9_dir)
    m9_mut = next(m for m in mutants if m.mutant_type == "M9")
    dut_m9 = m9_dir / "dut.v"
    dut_m9.write_text(m9_mut.mutated_code, encoding="utf-8")
    tb_m9 = m9_dir / "tb.v"
    tb_m9.write_text(
        generate_wrapper_testbench(
            op=op,
            num_vectors=len(x),
            x_hex_path=str(m9_dir / "x.hex"),
            y_hex_path=str(m9_dir / "y_exp.hex"),
            module_name="llmhw_waveform_demo",
            manifest=manifest,
            latency=1,
        ),
        encoding="utf-8",
    )
    rep_m9 = run_simulation(dut_m9, tb_m9, expected_checked=len(x), work_dir=m9_dir, dump_vcd=True)
    print(f"M9 run status: {rep_m9.status} (checked={rep_m9.checked}, timing_err={rep_m9.timing_err})")
    vcd_m9 = m9_dir / "llmhw_waveform_demo.vcd"
    if vcd_m9.exists():
        shutil.copy2(vcd_m9, art_dir / "fail_m9_timing.vcd")
        print(f"Copied {vcd_m9.name} -> artifacts/waveforms/fail_m9_timing.vcd ({vcd_m9.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
