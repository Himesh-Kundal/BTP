"""Unit tests for rtl/lib/numerics.vh and rtl/lib/requant.v (Task UC.1).

Exhaustively cross-tests Verilog-2001 numeric functions and requant module against
the reference oracle `numerics/ops.py`.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from numerics.ops import requant, rshift_rnd, sat


def test_rtl_rshift_rnd_and_sat_exhaustive_small_widths():
    """Verify rshift_rnd and sat functions in numerics.vh match numerics/ops.py exhaustively."""
    tb_code = """
`timescale 1ns/1ps
module tb_rshift_sat;
    `include "numerics.vh"

    integer i, s;
    reg signed [63:0] val;
    reg signed [63:0] rtl_res;
    reg signed [7:0]  rtl_sat8;

    initial begin
        // Loop over signed 10-bit range [-512, 511]
        for (i = -512; i <= 511; i = i + 1) begin
            val = i;
            // Test sat_int8
            rtl_sat8 = sat_int8(val);
            $display("SAT %0d %0d", i, rtl_sat8);

            // Test shifts from 0 to 8
            for (s = 0; s <= 8; s = s + 1) begin
                rtl_res = rshift_rnd64(val, s[5:0]);
                $display("SH %0d %0d %0d", i, s, rtl_res);
            end
        end
        $finish;
    end
endmodule
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        v_file = tmp_path / "tb.v"
        vvp_file = tmp_path / "tb.vvp"
        v_file.write_text(tb_code)

        compile_cmd = [
            "iverilog", "-g2005", "-Wall",
            "-I", "rtl/lib",
            "-o", str(vvp_file),
            str(v_file)
        ]
        comp = subprocess.run(compile_cmd, check=False, capture_output=True, text=True)
        assert comp.returncode == 0, f"Compilation failed: {comp.stderr}"

        run_cmd = ["vvp", str(vvp_file)]
        run = subprocess.run(run_cmd, check=False, capture_output=True, text=True)
        assert run.returncode == 0, f"Simulation failed: {run.stderr}"

        lines = run.stdout.strip().split("\n")
        sat_checked = 0
        sh_checked = 0

        for line in lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "SAT":
                val = int(parts[1])
                rtl_val = int(parts[2])
                expected = sat(val, -127, 127)
                assert rtl_val == expected, f"sat_int8 mismatch for val={val}: RTL={rtl_val}, expected={expected}"
                sat_checked += 1
            elif parts[0] == "SH":
                val = int(parts[1])
                s = int(parts[2])
                rtl_val = int(parts[3])
                expected = rshift_rnd(val, s)
                assert rtl_val == expected, f"rshift_rnd mismatch for val={val}, s={s}: RTL={rtl_val}, expected={expected}"
                sh_checked += 1

        assert sat_checked == 1024
        assert sh_checked == 1024 * 9


def test_rtl_requant_module():
    """Verify parameterized requant.v module against numerics/ops.py requant()."""
    test_cases = [
        # (acc, M, s)
        (0, 1000, 10),
        (100, 1024, 10),
        (-100, 1024, 10),
        (127, 1024, 10),
        (-127, 1024, 10),
        (50000, 2048, 12),
        (-50000, 2048, 12),
        (2147483647, 1, 24),   # Max INT32
        (-2147483648, 1, 24),  # Min INT32
        (1000000, 65536, 16),  # Saturation positive
        (-1000000, 65536, 16), # Saturation negative
        (15, 0, 5),            # Zero multiplier
        (7, 2, 0),             # Shift 0
        (-7, 2, 0),
        (12345, 54321, 20),
        (-12345, 54321, 20),
    ]

    # Add random test cases
    rng = np.random.RandomState(42)
    for _ in range(200):
        acc = int(rng.randint(-1_000_000, 1_000_000))
        M = int(rng.randint(1, 100_000))
        s = int(rng.randint(0, 28))
        test_cases.append((acc, M, s))

    def to_hex32(v: int) -> str:
        return f"32'sh{(v & 0xFFFFFFFF):08x}"

    tb_lines = [
        "`timescale 1ns/1ps",
        "module tb_requant;",
        "    reg clk;",
        "    reg rst_n;",
        "    reg valid_in;",
        "    reg signed [31:0] acc_in;",
        "    reg signed [31:0] M_in;",
        "    reg [5:0] shift_in;",
        "    wire valid_out;",
        "    wire signed [7:0] data_out;",
        "",
        "    requant #(32, 32, 6, 8, -8'sd127, 8'sd127, 1) dut (",
        "        .clk(clk),",
        "        .rst_n(rst_n),",
        "        .valid_in(valid_in),",
        "        .acc_in(acc_in),",
        "        .M_in(M_in),",
        "        .shift_in(shift_in),",
        "        .valid_out(valid_out),",
        "        .data_out(data_out)",
        "    );",
        "",
        "    always #5 clk = ~clk;",
        "",
        "    initial begin",
        "        clk = 0; rst_n = 0; valid_in = 0;",
        "        acc_in = 0; M_in = 0; shift_in = 0;",
        "        #20 rst_n = 1;",
        "        #10;",
    ]

    for idx, (acc, M, s) in enumerate(test_cases):
        tb_lines.append(f"        @(posedge clk); #1; valid_in = 1; acc_in = {to_hex32(acc)}; M_in = {to_hex32(M)}; shift_in = 6'd{s};")
        tb_lines.append("        @(posedge clk); #1; valid_in = 0;")
        tb_lines.append(f"        $display(\"RES %0d %0d\", {idx}, data_out);")

    tb_lines.append("        #20 $finish;")
    tb_lines.append("    end")
    tb_lines.append("endmodule")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        v_file = tmp_path / "tb.v"
        vvp_file = tmp_path / "tb.vvp"
        v_file.write_text("\n".join(tb_lines))

        compile_cmd = [
            "iverilog", "-g2005", "-Wall",
            "-I", "rtl/lib",
            "-o", str(vvp_file),
            str(v_file), "rtl/lib/requant.v"
        ]
        comp = subprocess.run(compile_cmd, check=False, capture_output=True, text=True)
        assert comp.returncode == 0, f"Compilation failed: {comp.stderr}"

        run_cmd = ["vvp", str(vvp_file)]
        run = subprocess.run(run_cmd, check=False, capture_output=True, text=True)
        assert run.returncode == 0, f"Simulation failed: {run.stderr}"

        lines = [ln for ln in run.stdout.strip().split("\n") if ln.startswith("RES")]
        assert len(lines) == len(test_cases)

        for line in lines:
            parts = line.split()
            idx = int(parts[1])
            acc, M, s = test_cases[idx]
            rtl_out = int(parts[2])
            expected = requant(acc, M, s, -127, 127)
            assert rtl_out == expected, f"Requant mismatch at idx={idx} (acc={acc}, M={M}, s={s}): RTL={rtl_out}, expected={expected}"
