"""Unit tests for parameterized RoPE unit (Task UC.4).

Verifies bit-for-bit equivalence against the reference oracle
`numerics/decoder.py:integer_rope` across multiple heads, dimensions, positions,
and boundary values.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from numerics.ops import build_rope_cos_sin_tables, rshift_rnd, sat


def test_rope_bit_exact_against_oracle():
    """Verify rope.v matches numerics/decoder.py integer_rope bit-for-bit."""
    num_heads = 4
    head_dim = 16
    max_ctx = 32
    half_dim = head_dim // 2

    cos_table, sin_table = build_rope_cos_sin_tables(
        max_ctx=max_ctx,
        head_dim=head_dim,
        theta=10000.0,
        out_bits=14,
    )

    def ref_rope(x: np.ndarray, pos: int) -> np.ndarray:
        # x is [num_heads, head_dim] INT8
        x1 = x[:, :half_dim].astype(np.int64)
        x2 = x[:, half_dim:].astype(np.int64)
        cos_m = cos_table[pos].astype(np.int64)
        sin_m = sin_table[pos].astype(np.int64)

        out1 = rshift_rnd(x1 * cos_m - x2 * sin_m, 14)
        out2 = rshift_rnd(x2 * cos_m + x1 * sin_m, 14)
        out = np.concatenate([out1, out2], axis=-1)
        return sat(out, -127, 127).astype(np.int8)

    # Test positions
    test_positions = [0, 1, 5, 15, 31]

    # Generate test vectors
    rng = np.random.RandomState(42)
    test_inputs = [
        # 1. Boundary: all zeros
        np.zeros((num_heads, head_dim), dtype=np.int8),
        # 2. Boundary: all max positive
        np.full((num_heads, head_dim), 127, dtype=np.int8),
        # 3. Boundary: all min negative
        np.full((num_heads, head_dim), -127, dtype=np.int8),
        # 4. Boundary: alternating +127, -127
        np.array([[127 if (h + i) % 2 == 0 else -127 for i in range(head_dim)] for h in range(num_heads)], dtype=np.int8),
        # 5. Random matrix
        rng.randint(-127, 127, size=(num_heads, head_dim), dtype=np.int8),
    ]

    total_words_tbl = max_ctx * half_dim
    cos_flat = cos_table.flatten()
    sin_flat = sin_table.flatten()

    cos_hex = [f"{int(c) & 0xFFFF:04x}" for c in cos_flat]
    sin_hex = [f"{int(s) & 0xFFFF:04x}" for s in sin_flat]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "cos.hex").write_text("\n".join(cos_hex) + "\n")
        (tmp_path / "sin.hex").write_text("\n".join(sin_hex) + "\n")

        for pos in test_positions:
            for vec_idx, x_mat in enumerate(test_inputs):
                expected = ref_rope(x_mat, pos)

                x_flat = x_mat.flatten()
                x_hex = [f"{int(v) & 0xFF:02x}" for v in x_flat]
                (tmp_path / "x.hex").write_text("\n".join(x_hex) + "\n")

                total_elem = num_heads * head_dim

                tb_code = f"""
`timescale 1ns/1ps
module tb_rope;
    reg clk;
    reg rst_n;
    reg start;
    wire busy;
    wire done;

    wire [15:0] src_addr;
    reg signed [7:0] src_data;

    wire [15:0] table_addr;
    reg signed [15:0] cos_data;
    reg signed [15:0] sin_data;

    wire dst_we;
    wire [15:0] dst_addr;
    wire signed [7:0] dst_data;

    reg signed [7:0] src_mem [0:{total_elem-1}];
    reg signed [15:0] cos_mem [0:{total_words_tbl-1}];
    reg signed [15:0] sin_mem [0:{total_words_tbl-1}];
    reg signed [7:0] dst_mem [0:{total_elem-1}];

    rope #(8, 16, 14, 8, -8'sd127, 8'sd127) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .num_heads(16'd{num_heads}),
        .head_dim(16'd{head_dim}),
        .pos(16'd{pos}),
        .busy(busy),
        .done(done),
        .src_addr(src_addr),
        .src_data(src_data),
        .table_addr(table_addr),
        .cos_data(cos_data),
        .sin_data(sin_data),
        .dst_we(dst_we),
        .dst_addr(dst_addr),
        .dst_data(dst_data)
    );

    always #5 clk = ~clk;

    always @(posedge clk) begin
        src_data <= src_mem[src_addr];
        cos_data <= cos_mem[table_addr];
        sin_data <= sin_mem[table_addr];
    end

    always @(posedge clk) begin
        if (dst_we) begin
            dst_mem[dst_addr] <= dst_data;
        end
    end

    integer i;
    initial begin
        $readmemh("{(tmp_path / 'x.hex').as_posix()}", src_mem);
        $readmemh("{(tmp_path / 'cos.hex').as_posix()}", cos_mem);
        $readmemh("{(tmp_path / 'sin.hex').as_posix()}", sin_mem);

        clk = 0; rst_n = 0; start = 0;
        #20 rst_n = 1;
        #10;
        @(posedge clk); #1;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        while (!done) @(posedge clk);
        #10;

        for (i = 0; i < {total_elem}; i = i + 1) begin
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
                    ["iverilog", "-g2005", "-Wall", "-I", "rtl/lib", "-o", str(vvp_file), str(tb_file), "rtl/lib/rope.v"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                assert comp.returncode == 0, f"iverilog failed: {comp.stderr}"

                sim = subprocess.run(["vvp", str(vvp_file)], check=False, capture_output=True, text=True)
                assert sim.returncode == 0, f"vvp failed: {sim.stderr}"

                lines = [ln for ln in sim.stdout.strip().splitlines() if ln.startswith("OUT")]
                assert len(lines) == total_elem, f"Expected {total_elem} output lines, got {len(lines)}"

                rtl_out = np.zeros(total_elem, dtype=np.int8)
                for ln in lines:
                    parts = ln.split()
                    idx = int(parts[1])
                    val = int(parts[2])
                    rtl_out[idx] = val

                rtl_mat = rtl_out.reshape(num_heads, head_dim)
                np.testing.assert_array_equal(
                    rtl_mat,
                    expected,
                    err_msg=f"RoPE mismatch at pos={pos}, vec_idx={vec_idx}",
                )
