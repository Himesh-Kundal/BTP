"""Unit tests for parameterized Argmax unit with per-row scale normalization (Task UC.8).

Verifies bit-for-bit equivalence against the reference oracle
`numerics/ops.py` (N2-12, N2-13):
1. Normalizes raw accumulators by per-row dyadic scales: rshift_rnd(raw * M, s).
2. Resolves ties to the lowest token ID.
3. Scale inversion tests: lower raw accumulator with larger scale beats higher raw accumulator with smaller scale.
4. Edge cases: single token vocab, best at index 0, best at index V-1, all negative values.
5. Random vectors across vocabulary sizes.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from numerics.ops import rshift_rnd


def py_argmax_oracle(logits, M_list, s_list):
    """Reference oracle for streaming argmax with dyadic scale normalization and lowest-id tie breaking."""
    best_val = None
    best_id = 0
    for i, (log, m, s) in enumerate(zip(logits, M_list, s_list)):
        prod = int(log) * int(m)
        val = rshift_rnd(prod, int(s))
        if best_val is None or val > best_val:
            best_val = val
            best_id = i
    return best_id


def run_argmax_sim(logits, M_list, s_list):
    """Run Verilog simulation of argmax.v and return the best_token_id."""
    vocab_size = len(logits)
    assert len(M_list) == vocab_size
    assert len(s_list) == vocab_size

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Write memory hex files
        logit_hex = [f"{int(v) & 0xFFFFFFFF:08x}" for v in logits]
        (tmp_path / "logits.hex").write_text("\n".join(logit_hex) + "\n")

        M_hex = [f"{int(m) & 0xFFFFFFFF:08x}" for m in M_list]
        (tmp_path / "M.hex").write_text("\n".join(M_hex) + "\n")

        s_hex = [f"{int(s) & 0x3F:02x}" for s in s_list]
        (tmp_path / "s.hex").write_text("\n".join(s_hex) + "\n")

        tb_code = f"""
`timescale 1ns/1ps
module tb_argmax;
    reg clk;
    reg rst_n;
    reg start;
    wire busy;
    wire done;
    wire [15:0] best_token_id;

    wire [15:0] logit_addr;
    reg signed [31:0] logit_data;

    wire [15:0] scale_addr;
    reg signed [31:0] M_data;
    reg [5:0] s_data;

    reg signed [31:0] logit_mem [0:{vocab_size-1}];
    reg signed [31:0] M_mem [0:{vocab_size-1}];
    reg [5:0] s_mem [0:{vocab_size-1}];

    initial begin
        $readmemh("{tmp_path}/logits.hex", logit_mem);
        $readmemh("{tmp_path}/M.hex", M_mem);
        $readmemh("{tmp_path}/s.hex", s_mem);
    end

    // 1-cycle synchronous read memories
    always @(posedge clk) begin
        logit_data <= logit_mem[logit_addr];
        M_data     <= M_mem[scale_addr];
        s_data     <= s_mem[scale_addr];
    end

    argmax #(
        .ACC_WIDTH(32),
        .M_WIDTH(32),
        .SHIFT_WIDTH(6),
        .TOK_WIDTH(16)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .vocab_size(16'd{vocab_size}),
        .busy(busy),
        .done(done),
        .best_token_id(best_token_id),
        .logit_addr(logit_addr),
        .logit_data(logit_data),
        .scale_addr(scale_addr),
        .M_data(M_data),
        .s_data(s_data)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rst_n = 0;
        start = 0;
        #20;
        rst_n = 1;
        #20;

        @(posedge clk);
        #1;
        start = 1;
        @(posedge clk);
        #1;
        start = 0;

        while (!done) begin
            @(posedge clk);
        end

        $display("RESULT: %0d", best_token_id);
        $finish;
    end
endmodule
"""
        tb_file = tmp_path / "tb_argmax.v"
        tb_file.write_text(tb_code)

        sim_exe = tmp_path / "sim.vvp"
        cmd_compile = [
            "iverilog",
            "-g2005",
            "-Irtl/lib",
            "-o",
            str(sim_exe),
            str(tb_file),
            "rtl/lib/argmax.v",
        ]
        res = subprocess.run(cmd_compile, check=False, capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

        cmd_run = ["vvp", str(sim_exe)]
        res_run = subprocess.run(cmd_run, check=False, capture_output=True, text=True)
        assert res_run.returncode == 0, f"Simulation failed: {res_run.stderr}"

        for line in res_run.stdout.splitlines():
            if line.startswith("RESULT:"):
                return int(line.split(":")[1].strip())

        raise RuntimeError(f"Could not parse result from sim output:\n{res_run.stdout}")


def test_argmax_tie_breaking():
    """Verify Rule N2-13: Ties must resolve to the lowest token ID."""
    vocab_size = 32
    # All tokens have identical raw logits and scales
    logits = [1000] * vocab_size
    M_list = [1] * vocab_size
    s_list = [0] * vocab_size

    res = run_argmax_sim(logits, M_list, s_list)
    assert res == 0, f"Expected lowest token ID 0, got {res}"

    # Specific subsets tie for maximum: tokens 7, 15, 23 tie with scaled value 5000, others lower
    logits = [100] * vocab_size
    logits[7] = 5000
    logits[15] = 5000
    logits[23] = 5000
    res = run_argmax_sim(logits, M_list, s_list)
    assert res == 7, f"Expected lowest tied token ID 7, got {res}"


def test_argmax_scale_inversion():
    """Verify that a lower raw accumulator with a larger scale beats a higher raw accumulator with smaller scale."""
    vocab_size = 16
    logits = [100] * vocab_size
    M_list = [1] * vocab_size
    s_list = [0] * vocab_size

    # Token 2 has very high raw accumulator (100,000) but small scale (M=1, s=10 -> ~98)
    logits[2] = 100000
    M_list[2] = 1
    s_list[2] = 10  # 100000 >> 10 = 98

    # Token 5 has lower raw accumulator (500) but larger scale (M=4, s=0 -> 2000)
    logits[5] = 500
    M_list[5] = 4
    s_list[5] = 0

    expected = py_argmax_oracle(logits, M_list, s_list)
    assert expected == 5
    res = run_argmax_sim(logits, M_list, s_list)
    assert res == expected, f"Expected {expected}, got {res}"


def test_argmax_boundary_positions():
    """Verify argmax when winner is at first index (0) or last index (vocab_size - 1)."""
    vocab_size = 64
    # Winner at 0
    logits = [10] * vocab_size
    M_list = [1] * vocab_size
    s_list = [0] * vocab_size
    logits[0] = 9999
    assert run_argmax_sim(logits, M_list, s_list) == 0

    # Winner at vocab_size - 1
    logits[0] = 10
    logits[vocab_size - 1] = 9999
    assert run_argmax_sim(logits, M_list, s_list) == vocab_size - 1

    # vocab_size == 1
    assert run_argmax_sim([42], [2], [1]) == 0


def test_argmax_negative_values():
    """Verify argmax when all values are negative and scaled."""
    logits = [-500, -200, -1000, -50, -300, -400, -80, -900]
    M_list = [2, 1, 1, 4, 1, 1, 1, 1]
    s_list = [1, 0, 0, 1, 0, 0, 0, 0]
    # Scaled values:
    # -500*2 >> 1 = -500
    # -200*1 >> 0 = -200
    # -1000*1 >> 0 = -1000
    # -50*4 >> 1 = -100 (WINNER)
    # -300, -400, -80 (WINNER? wait, -80 > -100! Let's check oracle)
    expected = py_argmax_oracle(logits, M_list, s_list)
    res = run_argmax_sim(logits, M_list, s_list)
    assert res == expected, f"Expected {expected}, got {res}"


def test_argmax_random_cases():
    """Verify against python oracle over random logits and scales across multiple sizes."""
    np.random.seed(42)
    for v_size in [1, 4, 17, 128, 256]:
        logits = np.random.randint(-100000, 100000, size=v_size).tolist()
        M_list = np.random.randint(1, 1000, size=v_size).tolist()
        s_list = np.random.randint(0, 16, size=v_size).tolist()

        expected = py_argmax_oracle(logits, M_list, s_list)
        res = run_argmax_sim(logits, M_list, s_list)
        assert res == expected, f"Vocab size {v_size}: expected {expected}, got {res}"
