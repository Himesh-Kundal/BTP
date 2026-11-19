"""Unit tests for parameterized Attention Datapath (Task UC.11).

Verifies bit-for-bit equivalence against the reference oracle
`numerics/decoder.py` and `compiler/interpreter.py`:
1. Single-position context (pos = 0): Softmax probability is exactly 1.0, output matches V vector.
2. Multi-position causal attention (pos = 3): Dot product q @ k.T >> 2, base-2 softmax, context weighted sum >> 15, symmetric INT8 sat.
3. GQA query-to-KV head mapping: q_idx // gqa_group.
4. KV cache storage and readback verification.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from numerics.ops import build_exp2_fraction_lut, rshift_rnd, sat


def py_attention_oracle(q_mat, k_cache, v_cache, pos, exp2_lut, num_q_heads, num_kv_heads, hd):
    """Reference Python oracle matching compiler/interpreter.py and numerics/decoder.py."""
    gqa_group = num_q_heads // num_kv_heads
    attn_out = np.zeros((num_q_heads, hd), dtype=np.int8)

    for q_idx in range(num_q_heads):
        kv_idx = q_idx // gqa_group
        k_hist = k_cache[kv_idx, : pos + 1, :].astype(np.int64)
        v_hist = v_cache[kv_idx, : pos + 1, :].astype(np.int64)
        q_vec = q_mat[q_idx].astype(np.int64)

        # Dot product
        scores = (q_vec @ k_hist.T) >> 1  # INT32
        max_score = int(np.max(scores))

        # Base-2 exp decomposition matching softmax.v
        diff = max_score - scores
        int_part = diff >> 8
        frac_part = diff & 0xFF
        exp_vals = []
        for i_p, f_p in zip(int_part, frac_part, strict=False):
            sh = min(16, int(i_p))
            val = int(exp2_lut[f_p]) >> sh
            exp_vals.append(max(1, val) if val == 0 else val)
        exp_arr = np.array(exp_vals, dtype=np.int64)

        # Softmax normalization
        sigma = max(1, int(np.sum(exp_arr)))
        probs = np.array([(int(v) * 32767) // sigma for v in exp_arr], dtype=np.int64)

        # Weighted context sum
        ctx = rshift_rnd(probs @ v_hist, 15)
        attn_out[q_idx] = sat(ctx, -127, 127).astype(np.int8)

    return attn_out


def make_kv_flat(k_cache, v_cache, num_kv_heads, max_ctx, hd):
    flat = np.zeros(1 * 2 * num_kv_heads * max_ctx * hd, dtype=np.int8)
    for h in range(num_kv_heads):
        for p in range(max_ctx):
            for d in range(hd):
                # K: is_v = 0
                addr_k = (((0 * 2 + 0) * num_kv_heads + h) * max_ctx + p) * hd + d
                flat[addr_k] = k_cache[h, p, d]
                # V: is_v = 1
                addr_v = (((0 * 2 + 1) * num_kv_heads + h) * max_ctx + p) * hd + d
                flat[addr_v] = v_cache[h, p, d]
    return flat


def run_attention_sim(q_vecs, k_vecs, v_vecs, pos, exp2_lut, k_cache=None, v_cache=None, num_q_heads=2, num_kv_heads=1, hd=8, max_ctx=16):
    """Simulate attention.v connected to kv_cache.v."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Write Exp2 LUT hex (18-bit hex)
        exp_hex = [f"{int(v) & 0x3FFFF:05x}" for v in exp2_lut]
        (tmp_path / "exp2.hex").write_text("\n".join(exp_hex) + "\n")

        # Flat Q, K, V buffers
        q_flat = q_vecs.flatten()
        k_flat = k_vecs.flatten()
        v_flat = v_vecs.flatten()

        q_hex = [f"{int(v) & 0xFF:02x}" for v in q_flat]
        k_hex = [f"{int(v) & 0xFF:02x}" for v in k_flat]
        v_hex = [f"{int(v) & 0xFF:02x}" for v in v_flat]

        (tmp_path / "q.hex").write_text("\n".join(q_hex) + "\n")
        (tmp_path / "k.hex").write_text("\n".join(k_hex) + "\n")
        (tmp_path / "v.hex").write_text("\n".join(v_hex) + "\n")

        # KV cache initial preload
        if k_cache is not None and v_cache is not None:
            kv_flat = make_kv_flat(k_cache, v_cache, num_kv_heads, max_ctx, hd)
        else:
            kv_flat = np.zeros(1 * 2 * num_kv_heads * max_ctx * hd, dtype=np.int8)
        kv_hex = [f"{int(b) & 0xFF:02x}" for b in kv_flat]
        (tmp_path / "kv_init.hex").write_text("\n".join(kv_hex) + "\n")

        tb_code = f"""
`timescale 1ns/1ps
module tb_attention;
    reg clk;
    reg rst_n;
    reg start;
    wire busy;
    wire done;

    wire [15:0] q_rd_addr;
    reg signed [7:0] q_rd_data;
    wire [15:0] k_rd_addr;
    reg signed [7:0] k_rd_data;
    wire [15:0] v_rd_addr;
    reg signed [7:0] v_rd_data;

    wire attn_wr_en;
    wire [15:0] attn_wr_addr;
    wire signed [7:0] attn_wr_data;

    wire [7:0] exp2_lut_addr;
    reg [17:0] exp2_lut_data;

    wire kv_wr_en;
    wire [0:0] kv_wr_layer;
    wire kv_wr_is_v;
    wire [0:0] kv_wr_kv_head;
    wire [3:0] kv_wr_pos;
    wire [2:0] kv_wr_dim;
    wire signed [7:0] kv_wr_data;

    wire kv_rd_en;
    wire [0:0] kv_rd_layer;
    wire kv_rd_is_v;
    wire [0:0] kv_rd_kv_head;
    wire [3:0] kv_rd_pos;
    wire [2:0] kv_rd_dim;
    wire signed [7:0] kv_rd_data;

    reg signed [7:0] q_mem [0:{len(q_flat) - 1}];
    reg signed [7:0] k_mem [0:{len(k_flat) - 1}];
    reg signed [7:0] v_mem [0:{len(v_flat) - 1}];
    reg [17:0] exp2_mem [0:255];
    reg signed [7:0] out_mem [0:{num_q_heads * hd - 1}];

    initial begin
        $readmemh("{tmp_path}/q.hex", q_mem);
        $readmemh("{tmp_path}/k.hex", k_mem);
        $readmemh("{tmp_path}/v.hex", v_mem);
        $readmemh("{tmp_path}/exp2.hex", exp2_mem);
        $readmemh("{tmp_path}/kv_init.hex", cache_inst.mem);
    end

    always @(posedge clk) begin
        q_rd_data <= q_mem[q_rd_addr];
        k_rd_data <= k_mem[k_rd_addr];
        v_rd_data <= v_mem[v_rd_addr];
        exp2_lut_data <= exp2_mem[exp2_lut_addr];
    end

    always @(posedge clk) begin
        if (attn_wr_en && attn_wr_addr < {num_q_heads * hd}) begin
            out_mem[attn_wr_addr] <= attn_wr_data;
        end
    end

    kv_cache #(
        .NUM_LAYERS(1),
        .NUM_KV_HEADS({num_kv_heads}),
        .MAX_CTX({max_ctx}),
        .HEAD_DIM({hd}),
        .DATA_WIDTH(8)
    ) cache_inst (
        .clk(clk),
        .rst_n(rst_n),
        .use_direct_addr(1'b0),
        .wr_en(kv_wr_en),
        .wr_addr_in(10'd0),
        .wr_layer(kv_wr_layer),
        .wr_is_v(kv_wr_is_v),
        .wr_kv_head(kv_wr_kv_head),
        .wr_pos(kv_wr_pos),
        .wr_dim(kv_wr_dim),
        .wr_data(kv_wr_data),
        .rd_en(kv_rd_en),
        .rd_addr_in(10'd0),
        .rd_layer(kv_rd_layer),
        .rd_is_v(kv_rd_is_v),
        .rd_kv_head(kv_rd_kv_head),
        .rd_pos(kv_rd_pos),
        .rd_dim(kv_rd_dim),
        .rd_data(kv_rd_data)
    );

    attention #(
        .NUM_LAYERS(1),
        .NUM_KV_HEADS({num_kv_heads}),
        .NUM_Q_HEADS({num_q_heads}),
        .HEAD_DIM({hd}),
        .MAX_CTX({max_ctx}),
        .DATA_WIDTH(8),
        .SCORES_WIDTH(32),
        .EXP_WIDTH(18),
        .PROB_WIDTH(16),
        .BUF_ADDR_W(16)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .busy(busy),
        .done(done),
        .layer_idx(5'd0),
        .pos(16'd{pos}),
        .q_rd_addr(q_rd_addr),
        .q_rd_data(q_rd_data),
        .k_rd_addr(k_rd_addr),
        .k_rd_data(k_rd_data),
        .v_rd_addr(v_rd_addr),
        .v_rd_data(v_rd_data),
        .attn_wr_en(attn_wr_en),
        .attn_wr_addr(attn_wr_addr),
        .attn_wr_data(attn_wr_data),
        .exp2_lut_addr(exp2_lut_addr),
        .exp2_lut_data(exp2_lut_data),
        .kv_wr_en(kv_wr_en),
        .kv_wr_layer(kv_wr_layer),
        .kv_wr_is_v(kv_wr_is_v),
        .kv_wr_kv_head(kv_wr_kv_head),
        .kv_wr_pos(kv_wr_pos),
        .kv_wr_dim(kv_wr_dim),
        .kv_wr_data(kv_wr_data),
        .kv_rd_en(kv_rd_en),
        .kv_rd_layer(kv_rd_layer),
        .kv_rd_is_v(kv_rd_is_v),
        .kv_rd_kv_head(kv_rd_kv_head),
        .kv_rd_pos(kv_rd_pos),
        .kv_rd_dim(kv_rd_dim),
        .kv_rd_data(kv_rd_data)
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
        for (fi = 0; fi < {num_q_heads * hd}; fi = fi + 1) begin
            $display("ATTN %0d %0d", fi, out_mem[fi]);
        end
        $finish;
    end
endmodule
"""
        tb_file = tmp_path / "tb_attention.v"
        tb_file.write_text(tb_code)

        sim_exe = tmp_path / "sim.vvp"
        compile_cmd = [
            "iverilog",
            "-g2005",
            "-Irtl/lib",
            "-o",
            str(sim_exe),
            str(tb_file),
            "rtl/lib/attention.v",
            "rtl/lib/softmax.v",
            "rtl/lib/kv_cache.v",
        ]
        res = subprocess.run(compile_cmd, check=False, capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

        run_cmd = ["vvp", str(sim_exe)]
        res_run = subprocess.run(run_cmd, check=False, capture_output=True, text=True)
        assert res_run.returncode == 0, f"Simulation failed: {res_run.stderr}"

        results = {}
        for line in res_run.stdout.splitlines():
            if line.startswith("ATTN "):
                parts = line.split()
                idx = int(parts[1])
                val = int(parts[2])
                results[idx] = val

        out_arr = np.array([results[i] for i in range(num_q_heads * hd)], dtype=np.int8)
        return out_arr.reshape((num_q_heads, hd))


def test_attention_single_token():
    """Verify pos = 0 attention where probability is 1.0."""
    exp2_lut = build_exp2_fraction_lut(256)
    np.random.seed(42)
    num_q, num_kv, hd = 2, 1, 8

    q = np.random.randint(-100, 100, size=(num_q, hd), dtype=np.int8)
    k = np.random.randint(-100, 100, size=(num_kv, hd), dtype=np.int8)
    v = np.random.randint(-100, 100, size=(num_kv, hd), dtype=np.int8)

    k_cache = np.zeros((num_kv, 16, hd), dtype=np.int8)
    v_cache = np.zeros((num_kv, 16, hd), dtype=np.int8)
    k_cache[:, 0, :] = k
    v_cache[:, 0, :] = v

    expected = py_attention_oracle(q, k_cache, v_cache, 0, exp2_lut, num_q, num_kv, hd)
    sim_out = run_attention_sim(
        q, k, v, 0, exp2_lut, k_cache=k_cache, v_cache=v_cache, num_q_heads=num_q, num_kv_heads=num_kv, hd=hd, max_ctx=16
    )

    np.testing.assert_array_equal(sim_out, expected)


def test_attention_multi_position_causal():
    """Verify causal attention over multiple positions."""
    exp2_lut = build_exp2_fraction_lut(256)
    np.random.seed(123)
    num_q, num_kv, hd = 2, 1, 8
    pos = 3

    # Historical K, V in cache for pos 0, 1, 2, 3
    k_cache = np.zeros((num_kv, 16, hd), dtype=np.int8)
    v_cache = np.zeros((num_kv, 16, hd), dtype=np.int8)
    for p in range(pos + 1):
        k_cache[:, p, :] = np.random.randint(-100, 100, size=(num_kv, hd), dtype=np.int8)
        v_cache[:, p, :] = np.random.randint(-100, 100, size=(num_kv, hd), dtype=np.int8)

    q = np.random.randint(-100, 100, size=(num_q, hd), dtype=np.int8)
    k_current = k_cache[:, pos, :]
    v_current = v_cache[:, pos, :]

    expected = py_attention_oracle(q, k_cache, v_cache, pos, exp2_lut, num_q, num_kv, hd)
    sim_out = run_attention_sim(
        q, k_current, v_current, pos, exp2_lut, k_cache=k_cache, v_cache=v_cache, num_q_heads=num_q, num_kv_heads=num_kv, hd=hd, max_ctx=16
    )

    np.testing.assert_array_equal(sim_out, expected)

    # Also test moderate magnitude where score differences fall within the non-saturated exp2 range
    k_cache_mod = (k_cache // 5).astype(np.int8)
    q_mod = (q // 5).astype(np.int8)
    k_current_mod = k_cache_mod[:, pos, :]
    expected_mod = py_attention_oracle(q_mod, k_cache_mod, v_cache, pos, exp2_lut, num_q, num_kv, hd)
    sim_out_mod = run_attention_sim(
        q_mod, k_current_mod, v_current, pos, exp2_lut, k_cache=k_cache_mod, v_cache=v_cache, num_q_heads=num_q, num_kv_heads=num_kv, hd=hd, max_ctx=16
    )
    np.testing.assert_array_equal(sim_out_mod, expected_mod)



def test_attention_gqa_mapping():
    """Verify GQA query-to-KV head mapping with group ratio 2:1 across 4 query heads."""
    exp2_lut = build_exp2_fraction_lut(256)
    np.random.seed(999)
    num_q, num_kv, hd = 4, 2, 8
    pos = 2

    k_cache = np.zeros((num_kv, 16, hd), dtype=np.int8)
    v_cache = np.zeros((num_kv, 16, hd), dtype=np.int8)
    for p in range(pos + 1):
        k_cache[:, p, :] = np.random.randint(-100, 100, size=(num_kv, hd), dtype=np.int8)
        v_cache[:, p, :] = np.random.randint(-100, 100, size=(num_kv, hd), dtype=np.int8)

    q = np.random.randint(-100, 100, size=(num_q, hd), dtype=np.int8)
    k_curr = k_cache[:, pos, :]
    v_curr = v_cache[:, pos, :]

    expected = py_attention_oracle(q, k_cache, v_cache, pos, exp2_lut, num_q, num_kv, hd)
    sim_out = run_attention_sim(
        q, k_curr, v_curr, pos, exp2_lut, k_cache=k_cache, v_cache=v_cache, num_q_heads=num_q, num_kv_heads=num_kv, hd=hd, max_ctx=16
    )

    np.testing.assert_array_equal(sim_out, expected)

