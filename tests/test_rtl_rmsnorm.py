"""Unit tests for parameterized RMSNorm unit (Task UC.3).

Verifies bit-for-bit equivalence against the reference oracle
`numerics/decoder.py:integer_rmsnorm` across:
1. Random vectors
2. Adversarial cases: all zeros, single huge value, equal elements, tiny elements.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from numerics.ops import build_rsqrt_mantissa_lut, requant, rshift_rnd


def test_rmsnorm_bit_exact_against_oracle():
    """Verify rmsnorm.v against numerics/decoder.py integer_rmsnorm."""
    dim = 64
    rsqrt_lut = build_rsqrt_mantissa_lut(256, 16)

    # Reference integer RMSNorm function
    def ref_rmsnorm(x: np.ndarray, gamma: np.ndarray, M: int, s: int) -> np.ndarray:
        x_64 = x.astype(np.int64)
        ss = np.sum(x_64 * x_64)
        mean = ss // dim
        if mean <= 0:
            return np.zeros(dim, dtype=np.int8)
        b = int(mean).bit_length()
        sh_amt = ((b - 7) & ~1) if b > 8 else 0
        mantissa = int(mean >> sh_amt) & 0xFF
        rsqrt_val = rsqrt_lut[mantissa]
        exp_sh = sh_amt // 2
        scaled_x = rshift_rnd(x_64 * rsqrt_val, exp_sh) if exp_sh > 0 else (x_64 * rsqrt_val)
        out = requant(scaled_x * gamma, M, 19 + s)
        return out.astype(np.int8)

    test_vectors = [
        # Adversarial 1: All zeros
        np.zeros(dim, dtype=np.int64),
        # Adversarial 2: All equal elements
        np.full(dim, 42, dtype=np.int64),
        # Adversarial 3: Tiny values
        np.array([(-1 if i % 2 == 0 else 1) for i in range(dim)], dtype=np.int64),
        # Adversarial 4: One huge element
        np.array([50000 if i == 0 else 0 for i in range(dim)], dtype=np.int64),
        # Adversarial 5: Alternating positive and negative large values
        np.array([(-5000 if i % 2 == 0 else 5000) for i in range(dim)], dtype=np.int64),
    ]

    # Add 10 random vectors
    rng = np.random.RandomState(42)
    for _ in range(10):
        test_vectors.append(rng.randint(-10000, 10000, size=dim, dtype=np.int64))

    gamma = rng.randint(100, 2000, size=dim).astype(np.int64)
    M = 1024
    s = 10

    # Build rsqrt LUT hex file
    rsqrt_hex_lines = [f"{int(val) & 0xFFFF:04x}" for val in rsqrt_lut]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        lut_hex_file = tmp_path / "rsqrt.hex"
        lut_hex_file.write_text("\n".join(rsqrt_hex_lines) + "\n")

        # Test each vector
        for vec_idx, x_vec in enumerate(test_vectors):
            expected = ref_rmsnorm(x_vec, gamma, M, s)

            # Build Verilog testbench
            x_hex_lines = [f"{int(val) & 0xFFFFFF:06x}" for val in x_vec]
            gamma_hex_lines = [f"{int(val) & 0xFFFF:04x}" for val in gamma]

            (tmp_path / "x.hex").write_text("\n".join(x_hex_lines) + "\n")
            (tmp_path / "gamma.hex").write_text("\n".join(gamma_hex_lines) + "\n")

            tb_code = f"""
`timescale 1ns/1ps
module tb_rmsnorm;
    reg clk;
    reg rst_n;
    reg start;
    wire busy;
    wire done;

    wire [15:0] src_addr;
    reg signed [23:0] src_data;
    wire [15:0] gamma_addr;
    reg signed [15:0] gamma_data;
    wire [7:0] rsqrt_lut_addr;
    reg [15:0] rsqrt_lut_data;

    wire dst_we;
    wire [15:0] dst_addr;
    wire signed [7:0] dst_data;

    reg signed [23:0] src_mem [0:{dim-1}];
    reg signed [15:0] gamma_mem [0:{dim-1}];
    reg [15:0] rsqrt_mem [0:255];
    reg signed [7:0] dst_mem [0:{dim-1}];

    rmsnorm #(24, 48, 16, 16, 8, -8'sd127, 8'sd127) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .dim(16'd{dim}),
        .M(32'sd{M}),
        .s(6'd{s}),
        .busy(busy),
        .done(done),
        .src_addr(src_addr),
        .src_data(src_data),
        .gamma_addr(gamma_addr),
        .gamma_data(gamma_data),
        .rsqrt_lut_addr(rsqrt_lut_addr),
        .rsqrt_lut_data(rsqrt_lut_data),
        .dst_we(dst_we),
        .dst_addr(dst_addr),
        .dst_data(dst_data)
    );

    always #5 clk = ~clk;

    // Synchronous memory reads
    always @(posedge clk) begin
        src_data <= src_mem[src_addr];
        gamma_data <= gamma_mem[gamma_addr];
        rsqrt_lut_data <= rsqrt_mem[rsqrt_lut_addr];
    end

    // Capture destination writes
    always @(posedge clk) begin
        if (dst_we) begin
            dst_mem[dst_addr] <= dst_data;
        end
    end

    integer i;
    initial begin
        $readmemh("{(tmp_path / 'x.hex').as_posix()}", src_mem);
        $readmemh("{(tmp_path / 'gamma.hex').as_posix()}", gamma_mem);
        $readmemh("{(tmp_path / 'rsqrt.hex').as_posix()}", rsqrt_mem);

        clk = 0; rst_n = 0; start = 0;
        #20 rst_n = 1;
        #10;
        @(posedge clk); #1;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        // Wait for done
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
                ["iverilog", "-g2005", "-Wall", "-I", "rtl/lib", "-o", str(vvp_file), str(tb_file), "rtl/lib/rmsnorm.v"],
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
                err_msg=f"Mismatch at vector {vec_idx}:\nExpected: {expected[:10]}\nGot:      {rtl_out[:10]}",
            )
