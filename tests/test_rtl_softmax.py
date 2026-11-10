"""Unit tests for parameterized Softmax unit (Task UC.5).

Verifies bit-for-bit equivalence against the reference oracle
in `numerics/decoder.py` across:
1. Single-position context (seq_len = 1)
2. All equal scores
3. Extreme score spreads (e.g. huge differences)
4. Full context length (seq_len = 256 = MAX_CTX)
5. Random score vectors
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from numerics.ops import build_exp2_fraction_lut


def test_softmax_bit_exact_against_oracle():
    """Verify softmax.v against numerics/decoder.py integer softmax logic."""
    exp2_lut = build_exp2_fraction_lut(256, 16)

    def ref_softmax(scores: np.ndarray) -> np.ndarray:
        m = int(np.max(scores))
        delta = scores - m
        exp_vals = []
        for d in delta:
            int_part = abs(int(d)) >> 8
            frac_idx = abs(int(d)) & 0xFF
            frac_val = exp2_lut[frac_idx]
            exp_v = frac_val >> min(16, int_part)
            exp_vals.append(max(1, exp_v))
        exp_arr = np.array(exp_vals, dtype=np.int64)
        sigma = max(1, int(np.sum(exp_arr)))
        probs = (exp_arr * 32767) // sigma
        return probs.astype(np.int64)

    exp2_hex_lines = [f"{int(val) & 0x3FFFF:05x}" for val in exp2_lut]

    rng = np.random.RandomState(42)

    test_cases = [
        # 1. Single position context
        np.array([12345], dtype=np.int64),
        # 2. Equal scores
        np.array([500, 500, 500, 500], dtype=np.int64),
        # 3. Extreme spread
        np.array([100000, -100000, 0, 50000], dtype=np.int64),
        # 4. Short sequence
        rng.randint(-5000, 5000, size=8, dtype=np.int64),
        # 5. Medium sequence
        rng.randint(-10000, 10000, size=32, dtype=np.int64),
        # 6. Max context length (seq_len = 256)
        rng.randint(-20000, 20000, size=256, dtype=np.int64),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "exp2.hex").write_text("\n".join(exp2_hex_lines) + "\n")

        for case_idx, scores in enumerate(test_cases):
            expected = ref_softmax(scores)
            seq_len = len(scores)

            score_hex_lines = [f"{int(s) & 0xFFFFFFFF:08x}" for s in scores]
            (tmp_path / "scores.hex").write_text("\n".join(score_hex_lines) + "\n")

            tb_code = f"""
`timescale 1ns/1ps
module tb_softmax;
    reg clk;
    reg rst_n;
    reg start;
    wire busy;
    wire done;

    wire [15:0] score_addr;
    reg signed [31:0] score_data;

    wire [7:0] exp2_lut_addr;
    reg [17:0] exp2_lut_data;

    wire prob_we;
    wire [15:0] prob_addr;
    wire [15:0] prob_data;

    reg signed [31:0] score_mem [0:255];
    reg [17:0] exp2_mem [0:255];
    reg [15:0] prob_mem [0:255];

    softmax #(32, 18, 16, 256) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .seq_len(16'd{seq_len}),
        .busy(busy),
        .done(done),
        .score_addr(score_addr),
        .score_data(score_data),
        .exp2_lut_addr(exp2_lut_addr),
        .exp2_lut_data(exp2_lut_data),
        .prob_we(prob_we),
        .prob_addr(prob_addr),
        .prob_data(prob_data)
    );

    always #5 clk = ~clk;

    always @(posedge clk) begin
        score_data <= score_mem[score_addr];
        exp2_lut_data <= exp2_mem[exp2_lut_addr];
    end

    always @(posedge clk) begin
        if (prob_we) begin
            prob_mem[prob_addr] <= prob_data;
        end
    end

    integer i;
    initial begin
        $readmemh("{(tmp_path / 'scores.hex').as_posix()}", score_mem);
        $readmemh("{(tmp_path / 'exp2.hex').as_posix()}", exp2_mem);

        clk = 0; rst_n = 0; start = 0;
        #20 rst_n = 1;
        #10;
        @(posedge clk); #1;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        while (!done) @(posedge clk);
        #10;

        for (i = 0; i < {seq_len}; i = i + 1) begin
            $display("OUT %0d %0d", i, prob_mem[i]);
        end
        #20 $finish;
    end
endmodule
"""
            tb_file = tmp_path / "tb.v"
            vvp_file = tmp_path / "tb.vvp"
            tb_file.write_text(tb_code)

            comp = subprocess.run(
                ["iverilog", "-g2005", "-Wall", "-I", "rtl/lib", "-o", str(vvp_file), str(tb_file), "rtl/lib/softmax.v"],
                check=False,
                capture_output=True,
                text=True,
            )
            assert comp.returncode == 0, f"iverilog failed: {comp.stderr}"

            sim = subprocess.run(["vvp", str(vvp_file)], check=False, capture_output=True, text=True)
            assert sim.returncode == 0, f"vvp failed: {sim.stderr}"

            lines = [ln for ln in sim.stdout.strip().splitlines() if ln.startswith("OUT")]
            assert len(lines) == seq_len, f"Expected {seq_len} lines, got {len(lines)}"

            rtl_probs = np.zeros(seq_len, dtype=np.int64)
            for ln in lines:
                parts = ln.split()
                idx = int(parts[1])
                val = int(parts[2])
                rtl_probs[idx] = val

            np.testing.assert_array_equal(
                rtl_probs,
                expected,
                err_msg=f"Softmax mismatch in case {case_idx} (seq_len={seq_len})",
            )
