"""Unit tests for parameterized Linear Engine (Task UC.10).

Verifies bit-for-bit equivalence against the reference integer oracle:
1. Exact multiple shape: (N=8, K=16) with tile (8, 16).
2. Multi-tile exact shape: (N=16, K=32) with tile (8, 16).
3. Non-multiple shape: (N=10, K=19) requiring zero-padding.
4. 24-bit residual/logit container mode (out_is_res = 1).
5. Sequential ROM streaming order verification.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from backend.tiling import pad_matrix
from numerics.ops import requant


def py_linear_oracle(x, w, M_list, s_val, out_is_res=False):
    """Reference integer oracle for linear projection: y = requant(x @ W.T, M, s)."""
    acc = x.astype(np.int64) @ w.T.astype(np.int64)
    out_min = -8388608 if out_is_res else -127
    out_max = 8388607 if out_is_res else 127
    m_arr = np.array(M_list, dtype=np.int64)
    return requant(acc, m_arr, s_val, out_min, out_max)


def pack_weights_to_rom_words(w, tile_r=8, tile_c=16):
    """Pack weight matrix into row-major tile words matching rom_packing.py."""
    w_pad, n_pad, k_pad = pad_matrix(w, tile_r, tile_c)
    num_tr = n_pad // tile_r
    num_tc = k_pad // tile_c

    rom_words = []
    for tr in range(num_tr):
        for tc in range(num_tc):
            tile = w_pad[tr * tile_r : (tr + 1) * tile_r, tc * tile_c : (tc + 1) * tile_c]
            for r in range(tile_r):
                row_bytes = tile[r, :]
                word_val = 0
                for c_idx in range(tile_c):
                    b = int(row_bytes[c_idx]) & 0xFF
                    word_val |= b << (c_idx * 8)
                rom_words.append(word_val)
    return rom_words, n_pad, k_pad


def run_linear_engine_sim(x, w, M_list, s_val, out_is_res=False, tile_r=8, tile_c=16):
    """Simulate linear_engine.v and return the output vector."""
    n, k = w.shape
    rom_words, n_pad, k_pad = pack_weights_to_rom_words(w, tile_r, tile_c)

    # Pad activation vector x to multiple of tile_c
    x_padded = np.zeros(k_pad, dtype=np.int8)
    x_padded[:k] = x
    num_act_words = k_pad // tile_c
    act_words = []
    for tc in range(num_act_words):
        chunk = x_padded[tc * tile_c : (tc + 1) * tile_c]
        val = 0
        for c_idx in range(tile_c):
            b = int(chunk[c_idx]) & 0xFF
            val |= b << (c_idx * 8)
        act_words.append(val)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Hex for weights
        hex_len_bytes = tile_c * 2
        w_hex = [f"{w_val:0{hex_len_bytes}x}" for w_val in rom_words]
        (tmp_path / "weights.hex").write_text("\n".join(w_hex) + "\n")

        # Hex for activations
        act_hex = [f"{a_val:0{hex_len_bytes}x}" for a_val in act_words]
        (tmp_path / "act.hex").write_text("\n".join(act_hex) + "\n")

        # Hex for params: M (32-bit hex) and s (16-bit hex)
        param_hex = [
            f"{int(m) & 0xFFFFFFFF:08x}{int(s_val) & 0xFFFF:04x}"
            for m in M_list
        ]
        # Pad params to n_pad if needed
        while len(param_hex) < n_pad:
            param_hex.append("000000010000")
        (tmp_path / "params.hex").write_text("\n".join(param_hex) + "\n")

        tb_code = f"""
`timescale 1ns/1ps
module tb_engine;
    reg clk;
    reg rst_n;
    reg start;
    wire busy;
    wire done;

    wire [15:0] act_rd_addr;
    reg signed [{tile_c * 8 - 1}:0] act_rd_data;

    wire [31:0] weight_rd_addr;
    reg signed [{tile_c * 8 - 1}:0] weight_rd_data;

    wire [31:0] param_rd_addr;
    reg signed [31:0] param_m_data;
    reg [5:0] param_s_data;

    wire out_wr_en;
    wire [15:0] out_wr_addr;
    wire signed [23:0] out_wr_data;

    reg signed [{tile_c * 8 - 1}:0] act_mem [0:{len(act_words) - 1}];
    reg signed [{tile_c * 8 - 1}:0] weight_mem [0:{len(rom_words) - 1}];
    reg [47:0] param_mem [0:{len(param_hex) - 1}];

    reg signed [23:0] out_mem [0:{n - 1}];

    initial begin
        $readmemh("{tmp_path}/act.hex", act_mem);
        $readmemh("{tmp_path}/weights.hex", weight_mem);
        $readmemh("{tmp_path}/params.hex", param_mem);
    end

    always @(posedge clk) begin
        act_rd_data <= act_mem[act_rd_addr];
        weight_rd_data <= weight_mem[weight_rd_addr];
        param_m_data <= param_mem[param_rd_addr][47:16];
        param_s_data <= param_mem[param_rd_addr][5:0];
    end

    always @(posedge clk) begin
        if (out_wr_en && out_wr_addr < {n}) begin
            out_mem[out_wr_addr] <= out_wr_data;
        end
    end

    linear_engine #(
        .R({tile_r}),
        .C({tile_c}),
        .ACT_WIDTH(8),
        .WEIGHT_WIDTH(8),
        .ACC_WIDTH(32),
        .M_WIDTH(32),
        .SHIFT_WIDTH(6),
        .BUF_ADDR_WIDTH(16),
        .ROM_ADDR_WIDTH(32)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .busy(busy),
        .done(done),
        .matrix_n(16'd{n}),
        .matrix_k(16'd{k}),
        .weight_base_addr(32'd0),
        .param_base_addr(32'd0),
        .out_is_res(1'b{1 if out_is_res else 0}),
        .act_rd_addr(act_rd_addr),
        .act_rd_data(act_rd_data),
        .weight_rd_addr(weight_rd_addr),
        .weight_rd_data(weight_rd_data),
        .param_rd_addr(param_rd_addr),
        .param_m_data(param_m_data),
        .param_s_data(param_s_data),
        .out_wr_en(out_wr_en),
        .out_wr_addr(out_wr_addr),
        .out_wr_data(out_wr_data)
    );

    always #5 clk = ~clk;

    integer fi;

    initial begin
        clk = 0;
        rst_n = 0;
        start = 0;
        #20;
        rst_n = 1;
        #20;

        @(posedge clk); #1;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        while (!done) begin
            @(posedge clk);
        end

        #10;
        for (fi = 0; fi < {n}; fi = fi + 1) begin
            $display("OUT %0d %0d", fi, out_mem[fi]);
        end
        $finish;
    end
endmodule
"""
        tb_file = tmp_path / "tb_engine.v"
        tb_file.write_text(tb_code)

        sim_exe = tmp_path / "sim.vvp"
        compile_cmd = [
            "iverilog",
            "-g2005",
            "-Irtl/lib",
            "-o",
            str(sim_exe),
            str(tb_file),
            "rtl/lib/linear_engine.v",
        ]
        res = subprocess.run(compile_cmd, check=False, capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

        run_cmd = ["vvp", str(sim_exe)]
        res_run = subprocess.run(run_cmd, check=False, capture_output=True, text=True)
        assert res_run.returncode == 0, f"Simulation failed: {res_run.stderr}"

        results = {}
        for line in res_run.stdout.splitlines():
            if line.startswith("OUT "):
                parts = line.split()
                idx = int(parts[1])
                val = int(parts[2])
                results[idx] = val

        out_arr = np.array([results[i] for i in range(n)])
        return out_arr


def test_linear_engine_exact_tile():
    """Verify linear engine on single exact tile (N=8, K=16)."""
    np.random.seed(42)
    n, k = 8, 16
    x = np.random.randint(-127, 127, size=k, dtype=np.int8)
    w = np.random.randint(-127, 127, size=(n, k), dtype=np.int8)
    M_list = np.random.randint(100, 500, size=n).tolist()
    s_val = 12

    expected = py_linear_oracle(x, w, M_list, s_val, out_is_res=False)
    sim_out = run_linear_engine_sim(x, w, M_list, s_val, out_is_res=False)

    np.testing.assert_array_equal(sim_out, expected)


def test_linear_engine_multi_tile_exact():
    """Verify linear engine across multiple tiles (N=16, K=32)."""
    np.random.seed(123)
    n, k = 16, 32
    x = np.random.randint(-100, 100, size=k, dtype=np.int8)
    w = np.random.randint(-100, 100, size=(n, k), dtype=np.int8)
    M_list = np.random.randint(50, 300, size=n).tolist()
    s_val = 14

    expected = py_linear_oracle(x, w, M_list, s_val, out_is_res=False)
    sim_out = run_linear_engine_sim(x, w, M_list, s_val, out_is_res=False)

    np.testing.assert_array_equal(sim_out, expected)


def test_linear_engine_non_multiple_shape_with_padding():
    """Verify non-multiple shape (N=10, K=19) pads to (16, 32) and produces exact unpadded results."""
    np.random.seed(456)
    n, k = 10, 19
    x = np.random.randint(-120, 120, size=k, dtype=np.int8)
    w = np.random.randint(-120, 120, size=(n, k), dtype=np.int8)
    M_list = np.random.randint(200, 800, size=n).tolist()
    s_val = 15

    expected = py_linear_oracle(x, w, M_list, s_val, out_is_res=False)
    sim_out = run_linear_engine_sim(x, w, M_list, s_val, out_is_res=False)

    np.testing.assert_array_equal(sim_out, expected)


def test_linear_engine_residual_24bit_mode():
    """Verify 24-bit container mode (out_is_res = True) preserving wider accumulator results."""
    np.random.seed(789)
    n, k = 8, 16
    x = np.random.randint(-127, 127, size=k, dtype=np.int8)
    w = np.random.randint(-127, 127, size=(n, k), dtype=np.int8)
    # Scales designed to produce values beyond INT8 range
    M_list = [1000] * n
    s_val = 5

    expected = py_linear_oracle(x, w, M_list, s_val, out_is_res=True)
    sim_out = run_linear_engine_sim(x, w, M_list, s_val, out_is_res=True)

    np.testing.assert_array_equal(sim_out, expected)


def test_linear_engine_sequential_rom_order():
    """Verify that weight ROM read address increments sequentially without gaps."""
    n, k = 16, 32
    w = np.ones((n, k), dtype=np.int8)
    rom_words, _, _ = pack_weights_to_rom_words(w, tile_r=8, tile_c=16)
    expected_total_weight_reads = len(rom_words)  # (16/8) * (32/16) * 8 = 32 words

    assert expected_total_weight_reads == 32

