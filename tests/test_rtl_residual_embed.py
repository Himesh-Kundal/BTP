"""Unit tests for residual_add and embed_gather units (Task UC.7).

Verifies bit-for-bit equivalence against the reference oracle
in `numerics/decoder.py` and `numerics/ops.py` including positive and negative saturation.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from numerics.ops import rshift_rnd, sat


def test_residual_add_with_saturation():
    """Verify residual_add.v conforms to saturating INT24 addition."""
    dim = 8
    # Test cases: normal, positive saturation, negative saturation, zeros
    a_vec = np.array([1000, 8388600, -8388600, 0, 5000000, -5000000, 100, -100], dtype=np.int64)
    b_vec = np.array([2000,     100,     -100, 0, 5000000, -5000000, -100,  100], dtype=np.int64)

    expected = sat(a_vec + b_vec, -8388608, 8388607)

    a_hex = [f"{int(v) & 0xFFFFFF:06x}" for v in a_vec]
    b_hex = [f"{int(v) & 0xFFFFFF:06x}" for v in b_vec]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "a.hex").write_text("\n".join(a_hex) + "\n")
        (tmp_path / "b.hex").write_text("\n".join(b_hex) + "\n")

        tb_code = f"""
`timescale 1ns/1ps
module tb_res_add;
    reg clk;
    reg rst_n;
    reg start;
    wire busy;
    wire done;

    wire [15:0] a_addr;
    reg signed [23:0] a_data;

    wire [15:0] b_addr;
    reg signed [23:0] b_data;

    wire dst_we;
    wire [15:0] dst_addr;
    wire signed [23:0] dst_data;

    reg signed [23:0] a_mem [0:{dim-1}];
    reg signed [23:0] b_mem [0:{dim-1}];
    reg signed [23:0] dst_mem [0:{dim-1}];

    residual_add #(24, -24'sd8388608, 24'sd8388607) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .dim(16'd{dim}),
        .busy(busy),
        .done(done),
        .a_addr(a_addr),
        .a_data(a_data),
        .b_addr(b_addr),
        .b_data(b_data),
        .dst_we(dst_we),
        .dst_addr(dst_addr),
        .dst_data(dst_data)
    );

    always #5 clk = ~clk;

    always @(posedge clk) begin
        a_data <= a_mem[a_addr];
        b_data <= b_mem[b_addr];
    end

    always @(posedge clk) begin
        if (dst_we) begin
            dst_mem[dst_addr] <= dst_data;
        end
    end

    integer i;
    initial begin
        $readmemh("{(tmp_path / 'a.hex').as_posix()}", a_mem);
        $readmemh("{(tmp_path / 'b.hex').as_posix()}", b_mem);

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
            ["iverilog", "-g2005", "-Wall", "-I", "rtl/lib", "-o", str(vvp_file), str(tb_file), "rtl/lib/residual_add.v"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert comp.returncode == 0, f"iverilog failed: {comp.stderr}"

        sim = subprocess.run(["vvp", str(vvp_file)], check=False, capture_output=True, text=True)
        assert sim.returncode == 0, f"vvp failed: {sim.stderr}"

        lines = [ln for ln in sim.stdout.strip().splitlines() if ln.startswith("OUT")]
        assert len(lines) == dim, f"Expected {dim} output lines, got {len(lines)}"

        rtl_out = np.zeros(dim, dtype=np.int64)
        for ln in lines:
            parts = ln.split()
            idx = int(parts[1])
            val = int(parts[2])
            rtl_out[idx] = val

        np.testing.assert_array_equal(rtl_out, expected)


def test_embed_gather():
    """Verify embed_gather.v conforms to embedding row gather and dyadic scaling."""
    vocab_size = 10
    hidden_size = 16
    total_elements = vocab_size * hidden_size

    rng = np.random.RandomState(42)
    embed_table = rng.randint(-127, 127, size=(vocab_size, hidden_size), dtype=np.int8)

    M = 4096
    s = 10

    test_tokens = [0, 3, 7, 9]

    embed_flat = embed_table.flatten()
    embed_hex = [f"{int(w) & 0xFF:02x}" for w in embed_flat]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "embed.hex").write_text("\n".join(embed_hex) + "\n")

        for tok_id in test_tokens:
            raw_w = embed_table[tok_id].astype(np.int64)
            expected = sat(rshift_rnd(raw_w * M, s), -8388608, 8388607)

            tb_code = f"""
`timescale 1ns/1ps
module tb_embed_gather;
    reg clk;
    reg rst_n;
    reg start;
    wire busy;
    wire done;

    wire [31:0] embed_addr;
    reg signed [7:0] embed_data;

    wire dst_we;
    wire [15:0] dst_addr;
    wire signed [23:0] dst_data;

    reg signed [7:0] embed_mem [0:{total_elements-1}];
    reg signed [23:0] dst_mem [0:{hidden_size-1}];

    embed_gather #(8, 24, -24'sd8388608, 24'sd8388607) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .token_id(16'd{tok_id}),
        .dim(16'd{hidden_size}),
        .M(32'sd{M}),
        .s(6'd{s}),
        .busy(busy),
        .done(done),
        .embed_addr(embed_addr),
        .embed_data(embed_data),
        .dst_we(dst_we),
        .dst_addr(dst_addr),
        .dst_data(dst_data)
    );

    always #5 clk = ~clk;

    always @(posedge clk) begin
        embed_data <= embed_mem[embed_addr];
    end

    always @(posedge clk) begin
        if (dst_we) begin
            dst_mem[dst_addr] <= dst_data;
        end
    end

    integer i;
    initial begin
        $readmemh("{(tmp_path / 'embed.hex').as_posix()}", embed_mem);

        clk = 0; rst_n = 0; start = 0;
        #20 rst_n = 1;
        #10;
        @(posedge clk); #1;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        while (!done) @(posedge clk);
        #10;

        for (i = 0; i < {hidden_size}; i = i + 1) begin
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
                ["iverilog", "-g2005", "-Wall", "-I", "rtl/lib", "-o", str(vvp_file), str(tb_file), "rtl/lib/embed_gather.v"],
                check=False,
                capture_output=True,
                text=True,
            )
            assert comp.returncode == 0, f"iverilog failed: {comp.stderr}"

            sim = subprocess.run(["vvp", str(vvp_file)], check=False, capture_output=True, text=True)
            assert sim.returncode == 0, f"vvp failed: {sim.stderr}"

            lines = [ln for ln in sim.stdout.strip().splitlines() if ln.startswith("OUT")]
            assert len(lines) == hidden_size

            rtl_out = np.zeros(hidden_size, dtype=np.int64)
            for ln in lines:
                parts = ln.split()
                idx = int(parts[1])
                val = int(parts[2])
                rtl_out[idx] = val

            np.testing.assert_array_equal(rtl_out, expected)
