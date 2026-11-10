"""Unit tests for KV cache RAM and address generator (Task UC.9).

Verifies:
1. GQA query-to-KV head mapping (9:3 for SmolLM2, 2:1 for tiny model).
2. Dual-port synchronous RAM read/write ordering, data retention, and isolation.
3. Coordinate calculation against Python reference.
4. Causal history streaming (mode 1) up to position pos.
5. Direct address vs coordinate addressing.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np


def py_calc_addr(layer, is_v, kv_head, pos, dim, num_kv_heads=3, max_ctx=256, head_dim=64):
    """Reference address calculation for KV cache layout."""
    addr = layer
    addr = addr * 2 + is_v
    addr = addr * num_kv_heads + kv_head
    addr = addr * max_ctx + pos
    addr = addr * head_dim + dim
    return addr


def test_gqa_mapping_and_coord_calc():
    """Verify GQA head mapping and static coordinate translation in kv_addr_gen.v."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        tb_code = """
`timescale 1ns/1ps
module tb_addr_gen;
    reg clk;
    reg rst_n;

    // SmolLM2 config: 30 layers, 3 kv heads, 9 q heads, max_ctx=256, hd=64
    reg [3:0] q_head_smol;
    wire [1:0] kv_head_smol;

    reg [4:0] coord_layer;
    reg coord_is_v;
    reg [1:0] coord_kv_head;
    reg [7:0] coord_pos;
    reg [5:0] coord_dim;
    wire [21:0] coord_addr;

    kv_addr_gen #(
        .NUM_LAYERS(30),
        .NUM_KV_HEADS(3),
        .NUM_Q_HEADS(9),
        .MAX_CTX(256),
        .HEAD_DIM(64)
    ) dut_smol (
        .clk(clk),
        .rst_n(rst_n),
        .q_head_in(q_head_smol),
        .kv_head_out(kv_head_smol),
        .coord_layer(coord_layer),
        .coord_is_v(coord_is_v),
        .coord_kv_head(coord_kv_head),
        .coord_pos(coord_pos),
        .coord_dim(coord_dim),
        .coord_addr(coord_addr),
        .start(1'b0),
        .step(1'b0),
        .mode(1'b0),
        .stream_layer(5'd0),
        .stream_is_v(1'b0),
        .stream_kv_head(2'd0),
        .stream_pos(8'd0),
        .stream_max_pos(8'd0),
        .busy(),
        .done(),
        .out_valid(),
        .out_addr(),
        .out_pos(),
        .out_dim(),
        .last_dim(),
        .last_token()
    );

    // Tiny model config: 1 layer, 1 kv head, 2 q heads, max_ctx=256, hd=64
    reg [1:0] q_head_tiny;
    wire [0:0] kv_head_tiny;
    kv_addr_gen #(
        .NUM_LAYERS(1),
        .NUM_KV_HEADS(1),
        .NUM_Q_HEADS(2),
        .MAX_CTX(256),
        .HEAD_DIM(64)
    ) dut_tiny (
        .clk(clk),
        .rst_n(rst_n),
        .q_head_in(q_head_tiny),
        .kv_head_out(kv_head_tiny),
        .coord_layer(1'b0),
        .coord_is_v(1'b0),
        .coord_kv_head(1'b0),
        .coord_pos(8'd0),
        .coord_dim(6'd0),
        .coord_addr(),
        .start(1'b0),
        .step(1'b0),
        .mode(1'b0),
        .stream_layer(1'b0),
        .stream_is_v(1'b0),
        .stream_kv_head(1'b0),
        .stream_pos(8'd0),
        .stream_max_pos(8'd0),
        .busy(),
        .done(),
        .out_valid(),
        .out_addr(),
        .out_pos(),
        .out_dim(),
        .last_dim(),
        .last_token()
    );

    initial begin
        // SmolLM2 GQA: 9 q_heads -> 3 kv_heads (group size 3)
        q_head_smol = 0; #1; if (kv_head_smol !== 2'd0) $finish(1);
        q_head_smol = 1; #1; if (kv_head_smol !== 2'd0) $finish(1);
        q_head_smol = 2; #1; if (kv_head_smol !== 2'd0) $finish(1);
        q_head_smol = 3; #1; if (kv_head_smol !== 2'd1) $finish(1);
        q_head_smol = 4; #1; if (kv_head_smol !== 2'd1) $finish(1);
        q_head_smol = 5; #1; if (kv_head_smol !== 2'd1) $finish(1);
        q_head_smol = 6; #1; if (kv_head_smol !== 2'd2) $finish(1);
        q_head_smol = 7; #1; if (kv_head_smol !== 2'd2) $finish(1);
        q_head_smol = 8; #1; if (kv_head_smol !== 2'd2) $finish(1);

        // Tiny model GQA: 2 q_heads -> 1 kv_head (group size 2)
        q_head_tiny = 0; #1; if (kv_head_tiny !== 1'd0) $finish(1);
        q_head_tiny = 1; #1; if (kv_head_tiny !== 1'd0) $finish(1);

        // Coordinate checks
        coord_layer = 0; coord_is_v = 0; coord_kv_head = 0; coord_pos = 0; coord_dim = 0;
        #1; if (coord_addr !== 22'd0) begin $display("FAIL 0: %0d", coord_addr); $finish(1); end

        coord_layer = 0; coord_is_v = 1; coord_kv_head = 0; coord_pos = 0; coord_dim = 0;
        #1; if (coord_addr !== 22'd49152) begin $display("FAIL 49152: %0d", coord_addr); $finish(1); end

        coord_layer = 1; coord_is_v = 0; coord_kv_head = 0; coord_pos = 0; coord_dim = 0;
        #1; if (coord_addr !== 22'd98304) begin $display("FAIL 98304: %0d", coord_addr); $finish(1); end

        $display("GQA_COORD_PASS");
        $finish(0);
    end
endmodule
"""
        tb_file = tmp_path / "tb_addr_gen.v"
        tb_file.write_text(tb_code)

        sim_exe = tmp_path / "sim.vvp"
        compile_cmd = [
            "iverilog",
            "-g2005",
            "-Irtl/lib",
            "-o",
            str(sim_exe),
            str(tb_file),
            "rtl/lib/kv_addr_gen.v",
        ]
        res = subprocess.run(compile_cmd, check=False, capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

        run_cmd = ["vvp", str(sim_exe)]
        res_run = subprocess.run(run_cmd, check=False, capture_output=True, text=True)
        assert res_run.returncode == 0 and "GQA_COORD_PASS" in res_run.stdout


def test_kv_cache_read_write_ordering():
    """Verify write and read ordering, multi-layer persistence, and data retention."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        tb_code = """
`timescale 1ns/1ps
module tb_kv_rw;
    reg clk;
    reg rst_n;
    reg wr_en;
    reg [1:0] wr_layer;
    reg wr_is_v;
    reg [0:0] wr_kv_head;
    reg [7:0] wr_pos;
    reg [5:0] wr_dim;
    reg signed [7:0] wr_data;

    reg rd_en;
    reg [1:0] rd_layer;
    reg rd_is_v;
    reg [0:0] rd_kv_head;
    reg [7:0] rd_pos;
    reg [5:0] rd_dim;
    wire signed [7:0] rd_data;

    // Small instance: 4 layers, 1 kv head, max_ctx=16, hd=8
    kv_cache #(
        .NUM_LAYERS(4),
        .NUM_KV_HEADS(1),
        .MAX_CTX(16),
        .HEAD_DIM(8),
        .DATA_WIDTH(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .use_direct_addr(1'b0),
        .wr_en(wr_en),
        .wr_addr_in(10'd0),
        .wr_layer(wr_layer),
        .wr_is_v(wr_is_v),
        .wr_kv_head(wr_kv_head),
        .wr_pos(wr_pos),
        .wr_dim(wr_dim),
        .wr_data(wr_data),
        .rd_en(rd_en),
        .rd_addr_in(10'd0),
        .rd_layer(rd_layer),
        .rd_is_v(rd_is_v),
        .rd_kv_head(rd_kv_head),
        .rd_pos(rd_pos),
        .rd_dim(rd_dim),
        .rd_data(rd_data)
    );

    always #5 clk = ~clk;

    integer i, l, p, d;

    initial begin
        clk = 0;
        rst_n = 0;
        wr_en = 0;
        rd_en = 0;
        #20;
        rst_n = 1;
        #10;

        // Write patterned values across layers, positions, and dimensions
        for (l = 0; l < 4; l = l + 1) begin
            for (p = 0; p < 4; p = p + 1) begin
                for (d = 0; d < 8; d = d + 1) begin
                    @(posedge clk);
                    #1;
                    wr_en = 1;
                    wr_layer = l[1:0];
                    wr_is_v = 0; // Key
                    wr_kv_head = 0;
                    wr_pos = p[7:0];
                    wr_dim = d[5:0];
                    wr_data = (l * 30 + p * 8 + d) & 8'h7F;
                end
            end
        end

        // Write Value vectors for position 2 across layers
        for (l = 0; l < 4; l = l + 1) begin
            for (d = 0; d < 8; d = d + 1) begin
                @(posedge clk);
                #1;
                wr_en = 1;
                wr_layer = l[1:0];
                wr_is_v = 1; // Value
                wr_kv_head = 0;
                wr_pos = 8'd2;
                wr_dim = d[5:0];
                wr_data = -8'sd10 - d;
            end
        end

        @(posedge clk);
        #1;
        wr_en = 0;
        #20;

        // Verify Key vectors
        for (l = 0; l < 4; l = l + 1) begin
            for (p = 0; p < 4; p = p + 1) begin
                for (d = 0; d < 8; d = d + 1) begin
                    @(posedge clk);
                    #1;
                    rd_en = 1;
                    rd_layer = l[1:0];
                    rd_is_v = 0;
                    rd_kv_head = 0;
                    rd_pos = p[7:0];
                    rd_dim = d[5:0];

                    @(posedge clk); // Synchronous read cycle
                    #1;
                    if (rd_data !== ((l * 30 + p * 8 + d) & 8'h7F)) begin
                        $display("MISMATCH Key L%0d P%0d D%0d: got %0d expected %0d",
                                 l, p, d, rd_data, ((l * 30 + p * 8 + d) & 8'h7F));
                        $finish(1);
                    end
                end
            end
        end

        // Verify Value vectors
        for (l = 0; l < 4; l = l + 1) begin
            for (d = 0; d < 8; d = d + 1) begin
                @(posedge clk);
                #1;
                rd_en = 1;
                rd_layer = l[1:0];
                rd_is_v = 1;
                rd_kv_head = 0;
                rd_pos = 8'd2;
                rd_dim = d[5:0];

                @(posedge clk);
                #1;
                if (rd_data !== (-8'sd10 - d)) begin
                    $display("MISMATCH Value L%0d D%0d: got %0d expected %0d", l, d, rd_data, -8'sd10 - d);
                    $finish(1);
                end
            end
        end

        $display("KV_RW_PASS");
        $finish(0);
    end
endmodule
"""
        tb_file = tmp_path / "tb_kv_rw.v"
        tb_file.write_text(tb_code)

        sim_exe = tmp_path / "sim.vvp"
        compile_cmd = [
            "iverilog",
            "-g2005",
            "-Irtl/lib",
            "-o",
            str(sim_exe),
            str(tb_file),
            "rtl/lib/kv_cache.v",
        ]
        res = subprocess.run(compile_cmd, check=False, capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

        run_cmd = ["vvp", str(sim_exe)]
        res_run = subprocess.run(run_cmd, check=False, capture_output=True, text=True)
        assert res_run.returncode == 0 and "KV_RW_PASS" in res_run.stdout


def test_kv_streaming_causal_sequence():
    """Verify streaming generator connected to KV cache for causal sequence playback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        tb_code = """
`timescale 1ns/1ps
module tb_stream;
    reg clk;
    reg rst_n;

    // Direct address connection from addr_gen to cache
    wire [13:0] addr;
    wire [7:0] stream_pos;
    wire [5:0] stream_dim;
    wire last_dim;
    wire last_token;
    wire out_valid;
    wire busy;
    wire done;

    reg start;
    reg step;
    reg [7:0] max_pos;

    kv_addr_gen #(
        .NUM_LAYERS(2),
        .NUM_KV_HEADS(1),
        .NUM_Q_HEADS(2),
        .MAX_CTX(64),
        .HEAD_DIM(8)
    ) addr_gen (
        .clk(clk),
        .rst_n(rst_n),
        .q_head_in(2'd0),
        .kv_head_out(),
        .coord_layer(1'd0),
        .coord_is_v(1'b0),
        .coord_kv_head(1'd0),
        .coord_pos(8'd0),
        .coord_dim(6'd0),
        .coord_addr(),
        .start(start),
        .step(step),
        .mode(1'b1), // Mode 1: sequence
        .stream_layer(1'd1), // Layer 1
        .stream_is_v(1'b0), // Key
        .stream_kv_head(1'd0),
        .stream_pos(8'd0),
        .stream_max_pos(max_pos),
        .busy(busy),
        .done(done),
        .out_valid(out_valid),
        .out_addr(addr),
        .out_pos(stream_pos),
        .out_dim(stream_dim),
        .last_dim(last_dim),
        .last_token(last_token)
    );

    wire signed [7:0] rd_data;

    kv_cache #(
        .NUM_LAYERS(2),
        .NUM_KV_HEADS(1),
        .MAX_CTX(64),
        .HEAD_DIM(8),
        .DATA_WIDTH(8)
    ) cache (
        .clk(clk),
        .rst_n(rst_n),
        .use_direct_addr(1'b1),
        .wr_en(1'b0),
        .wr_addr_in(14'd0),
        .wr_layer(1'd0),
        .wr_is_v(1'b0),
        .wr_kv_head(1'd0),
        .wr_pos(8'd0),
        .wr_dim(6'd0),
        .wr_data(8'd0),
        .rd_en(out_valid),
        .rd_addr_in(addr),
        .rd_layer(1'd0),
        .rd_is_v(1'b0),
        .rd_kv_head(1'd0),
        .rd_pos(8'd0),
        .rd_dim(6'd0),
        .rd_data(rd_data)
    );

    always #5 clk = ~clk;

    integer word_count;
    integer expected_pos;
    integer expected_dim;

    initial begin
        clk = 0;
        rst_n = 0;
        start = 0;
        step = 1;
        max_pos = 8'd4; // 5 tokens (0..4), 8 dims each = 40 words
        word_count = 0;
        expected_pos = 0;
        expected_dim = 0;
        #20;
        rst_n = 1;
        #20;

        @(posedge clk);
        #1;
        start = 1;
        @(posedge clk);
        #1;
        start = 0;

        while (busy || out_valid) begin
            if (out_valid) begin
                if (stream_pos !== expected_pos[7:0] || stream_dim !== expected_dim[5:0]) begin
                    $display("STREAM ERROR at count %0d: pos=%0d (exp %0d), dim=%0d (exp %0d)",
                             word_count, stream_pos, expected_pos, stream_dim, expected_dim);
                    $finish(1);
                end
                word_count = word_count + 1;
                expected_dim = expected_dim + 1;
                if (expected_dim == 8) begin
                    expected_dim = 0;
                    expected_pos = expected_pos + 1;
                end
            end
            @(posedge clk);
            #1;
        end

        if (word_count !== 40) begin
            $display("ERROR: Expected 40 words, got %0d", word_count);
            $finish(1);
        end

        $display("STREAM_PASS words=%0d", word_count);
        $finish(0);
    end
endmodule
"""
        tb_file = tmp_path / "tb_stream.v"
        tb_file.write_text(tb_code)

        sim_exe = tmp_path / "sim.vvp"
        compile_cmd = [
            "iverilog",
            "-g2005",
            "-Irtl/lib",
            "-o",
            str(sim_exe),
            str(tb_file),
            "rtl/lib/kv_addr_gen.v",
            "rtl/lib/kv_cache.v",
        ]
        res = subprocess.run(compile_cmd, check=False, capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

        run_cmd = ["vvp", str(sim_exe)]
        res_run = subprocess.run(run_cmd, check=False, capture_output=True, text=True)
        assert res_run.returncode == 0 and "STREAM_PASS" in res_run.stdout


def test_kv_cache_direct_address_mode():
    """Verify kv_cache when use_direct_addr = 1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        tb_code = """
`timescale 1ns/1ps
module tb_direct;
    reg clk;
    reg rst_n;
    reg wr_en;
    reg [9:0] wr_addr;
    reg signed [7:0] wr_data;
    reg rd_en;
    reg [9:0] rd_addr;
    wire signed [7:0] rd_data;

    kv_cache #(
        .NUM_LAYERS(1),
        .NUM_KV_HEADS(1),
        .MAX_CTX(16),
        .HEAD_DIM(8),
        .DATA_WIDTH(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .use_direct_addr(1'b1),
        .wr_en(wr_en),
        .wr_addr_in(wr_addr),
        .wr_layer(1'd0),
        .wr_is_v(1'b0),
        .wr_kv_head(1'd0),
        .wr_pos(8'd0),
        .wr_dim(6'd0),
        .wr_data(wr_data),
        .rd_en(rd_en),
        .rd_addr_in(rd_addr),
        .rd_layer(1'd0),
        .rd_is_v(1'b0),
        .rd_kv_head(1'd0),
        .rd_pos(8'd0),
        .rd_dim(6'd0),
        .rd_data(rd_data)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rst_n = 0;
        wr_en = 0;
        rd_en = 0;
        #20;
        rst_n = 1;
        #10;

        @(posedge clk); #1;
        wr_en = 1; wr_addr = 10'd42; wr_data = 8'sd120;
        @(posedge clk); #1;
        wr_en = 1; wr_addr = 10'd99; wr_data = -8'sd88;
        @(posedge clk); #1;
        wr_en = 0;

        @(posedge clk); #1;
        rd_en = 1; rd_addr = 10'd42;
        @(posedge clk); #1;
        if (rd_data !== 8'sd120) $finish(1);

        rd_addr = 10'd99;
        @(posedge clk); #1;
        if (rd_data !== -8'sd88) $finish(1);

        $display("DIRECT_ADDR_PASS");
        $finish(0);
    end
endmodule
"""
        tb_file = tmp_path / "tb_direct.v"
        tb_file.write_text(tb_code)

        sim_exe = tmp_path / "sim.vvp"
        compile_cmd = [
            "iverilog",
            "-g2005",
            "-Irtl/lib",
            "-o",
            str(sim_exe),
            str(tb_file),
            "rtl/lib/kv_cache.v",
        ]
        res = subprocess.run(compile_cmd, check=False, capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

        run_cmd = ["vvp", str(sim_exe)]
        res_run = subprocess.run(run_cmd, check=False, capture_output=True, text=True)
        assert res_run.returncode == 0 and "DIRECT_ADDR_PASS" in res_run.stdout


def test_coord_calc_random_battery():
    """Verify coordinate translation against py_calc_addr across random coordinates."""
    np.random.seed(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cases = []
        for _ in range(50):
            l = int(np.random.randint(0, 30))
            v = int(np.random.randint(0, 2))
            h = int(np.random.randint(0, 3))
            p = int(np.random.randint(0, 256))
            d = int(np.random.randint(0, 64))
            exp_addr = py_calc_addr(l, v, h, p, d, num_kv_heads=3, max_ctx=256, head_dim=64)
            cases.append((l, v, h, p, d, exp_addr))

        checks_verilog = "\n".join(
            [
                f"coord_layer = 5'd{l}; coord_is_v = 1'b{v}; coord_kv_head = 2'd{h}; "
                f"coord_pos = 8'd{p}; coord_dim = 6'd{d}; #1; "
                f"if (coord_addr !== 22'd{exp}) begin $display(\"FAIL case {i}\"); $finish(1); end"
                for i, (l, v, h, p, d, exp) in enumerate(cases)
            ]
        )

        tb_code = f"""
`timescale 1ns/1ps
module tb_battery;
    reg clk;
    reg rst_n;
    reg [4:0] coord_layer;
    reg coord_is_v;
    reg [1:0] coord_kv_head;
    reg [7:0] coord_pos;
    reg [5:0] coord_dim;
    wire [21:0] coord_addr;

    kv_addr_gen #(
        .NUM_LAYERS(30),
        .NUM_KV_HEADS(3),
        .NUM_Q_HEADS(9),
        .MAX_CTX(256),
        .HEAD_DIM(64)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .q_head_in(4'd0),
        .kv_head_out(),
        .coord_layer(coord_layer),
        .coord_is_v(coord_is_v),
        .coord_kv_head(coord_kv_head),
        .coord_pos(coord_pos),
        .coord_dim(coord_dim),
        .coord_addr(coord_addr),
        .start(1'b0),
        .step(1'b0),
        .mode(1'b0),
        .stream_layer(5'd0),
        .stream_is_v(1'b0),
        .stream_kv_head(2'd0),
        .stream_pos(8'd0),
        .stream_max_pos(8'd0),
        .busy(),
        .done(),
        .out_valid(),
        .out_addr(),
        .out_pos(),
        .out_dim(),
        .last_dim(),
        .last_token()
    );

    initial begin
        {checks_verilog}
        $display("BATTERY_PASS");
        $finish(0);
    end
endmodule
"""
        tb_file = tmp_path / "tb_battery.v"
        tb_file.write_text(tb_code)

        sim_exe = tmp_path / "sim.vvp"
        compile_cmd = [
            "iverilog",
            "-g2005",
            "-Irtl/lib",
            "-o",
            str(sim_exe),
            str(tb_file),
            "rtl/lib/kv_addr_gen.v",
        ]
        res = subprocess.run(compile_cmd, check=False, capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

        run_cmd = ["vvp", str(sim_exe)]
        res_run = subprocess.run(run_cmd, check=False, capture_output=True, text=True)
        assert res_run.returncode == 0 and "BATTERY_PASS" in res_run.stdout

