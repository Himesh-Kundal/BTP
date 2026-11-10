"""Unit tests for parameterized SwiGLU unit (Task UC.6).

Verifies bit-for-bit equivalence against the reference oracle
`numerics/ops.py` and `compiler/interpreter.py`:
Exhaustive testing over the entire 8-bit SiLU input domain (256 values in [-128, 127])
across representative `up` values ([-127, -64, -1, 0, 1, 64, 127]).
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from numerics.ops import build_silu_lut, rshift_rnd, sat


def test_swiglu_exhaustive_domain():
    """Exhaustively verify swiglu.v across all 256 gate values and representative up values."""
    silu_lut = build_silu_lut(in_min=-128, in_max=127, in_scale=0.05, out_scale=0.05)
    assert len(silu_lut) == 256

    representative_ups = [-127, -64, -1, 0, 1, 64, 127]
    all_gate_vals = list(range(-128, 128))

    dim = len(all_gate_vals)  # 256

    silu_hex_lines = [f"{int(v) & 0xFF:02x}" for v in silu_lut]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "silu.hex").write_text("\n".join(silu_hex_lines) + "\n")

        gate_hex = [f"{int(g) & 0xFF:02x}" for g in all_gate_vals]
        (tmp_path / "gate.hex").write_text("\n".join(gate_hex) + "\n")

        for up_val in representative_ups:
            up_vec = np.full(dim, up_val, dtype=np.int64)
            gate_vec = np.array(all_gate_vals, dtype=np.int64)

            # Expected oracle calculation
            silu_out = np.array([silu_lut[int(g) + 128] for g in gate_vec], dtype=np.int64)
            prod = silu_out * up_vec
            expected = sat(rshift_rnd(prod, 7), -127, 127).astype(np.int8)

            up_hex = [f"{int(u) & 0xFF:02x}" for u in up_vec]
            (tmp_path / "up.hex").write_text("\n".join(up_hex) + "\n")

            tb_code = f"""
`timescale 1ns/1ps
module tb_swiglu;
    reg clk;
    reg rst_n;
    reg start;
    wire busy;
    wire done;

    wire [15:0] gate_addr;
    reg signed [7:0] gate_data;

    wire [15:0] up_addr;
    reg signed [7:0] up_data;

    wire [8:0] silu_lut_addr;
    reg signed [7:0] silu_lut_data;

    wire dst_we;
    wire [15:0] dst_addr;
    wire signed [7:0] dst_data;

    reg signed [7:0] gate_mem [0:{dim-1}];
    reg signed [7:0] up_mem [0:{dim-1}];
    reg signed [7:0] silu_mem [0:255];
    reg signed [7:0] dst_mem [0:{dim-1}];

    swiglu #(8, 8, 256, 8, -8'sd127, 8'sd127) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .dim(16'd{dim}),
        .M(32'sd1),
        .s(6'd7),
        .busy(busy),
        .done(done),
        .gate_addr(gate_addr),
        .gate_data(gate_data),
        .up_addr(up_addr),
        .up_data(up_data),
        .silu_lut_addr(silu_lut_addr),
        .silu_lut_data(silu_lut_data),
        .dst_we(dst_we),
        .dst_addr(dst_addr),
        .dst_data(dst_data)
    );

    always #5 clk = ~clk;

    always @(posedge clk) begin
        gate_data <= gate_mem[gate_addr];
        up_data   <= up_mem[up_addr];
        silu_lut_data <= silu_mem[silu_lut_addr[7:0]];
    end

    always @(posedge clk) begin
        if (dst_we) begin
            dst_mem[dst_addr] <= dst_data;
        end
    end

    integer i;
    initial begin
        $readmemh("{(tmp_path / 'gate.hex').as_posix()}", gate_mem);
        $readmemh("{(tmp_path / 'up.hex').as_posix()}", up_mem);
        $readmemh("{(tmp_path / 'silu.hex').as_posix()}", silu_mem);

        clk = 0; rst_n = 0; start = 0;
        #20 rst_n = 1;
        #10;
        @(posedge clk); #1;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        while (!done) @(posedge clk);
        #10;

        for (i = 0; i < {dim}; i = i + 1) begin
            $display("OUT %0d %0d", i, dst_mem[i]);
        end
        #20 $finish;
    end
endmodule
"""
            tb_file = tmp_path / "tb.v"
            vvp_file = tmp_path / "tb.vvp"
            tb_file.write_text(tb_code)

            comp = subprocess.run(
                ["iverilog", "-g2005", "-Wall", "-I", "rtl/lib", "-o", str(vvp_file), str(tb_file), "rtl/lib/swiglu.v"],
                check=False,
                capture_output=True,
                text=True,
            )
            assert comp.returncode == 0, f"iverilog failed: {comp.stderr}"

            sim = subprocess.run(["vvp", str(vvp_file)], check=False, capture_output=True, text=True)
            assert sim.returncode == 0, f"vvp failed: {sim.stderr}"

            lines = [ln for ln in sim.stdout.strip().splitlines() if ln.startswith("OUT")]
            assert len(lines) == dim, f"Expected {dim} output lines, got {len(lines)}"

            rtl_out = np.zeros(dim, dtype=np.int8)
            for ln in lines:
                parts = ln.split()
                idx = int(parts[1])
                val = int(parts[2])
                rtl_out[idx] = val

            np.testing.assert_array_equal(
                rtl_out,
                expected,
                err_msg=f"SwiGLU mismatch for up_val={up_val}",
            )
