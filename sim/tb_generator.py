"""Testbench Generator & Runner for Top-Level Accelerator (Task UD.2).

Generates plain Verilog testbenches for `top_accel`:
- Drives step interface: clk, rst_n, seq_reset, step_valid, tok_in
- Feeds prompt tokens, steps autoregressive decode for N tokens
- Hierarchical trace dumping to build/<run>/trace/seq<id>/tok<idx>/L<layer>/<op>_<tensor>.hex
- Emits summary line:
  `TB2_RESULT TOKENS=<n> TOK_MISMATCH=<m> TRACE_CHECKED=<k> TRACE_MISMATCH=<j> TIMING_ERR=<t>`
- Verifies bit-exact and token-exact equivalence against golden traces and Python interpreter
- Validates negative controls (corrupted golden trace causes test failure)

Strict Layering (Golden Rule 6):
Only pure Python and NumPy are used. Never import torch or transformers.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from compiler.interpreter import CompiledArtifactInterpreter
from verification.trace_diff import DiffResult, compare_trace_directories


@dataclass
class Tb2SimulationResult:
    """Structured result from top-level accelerator testbench execution."""
    success: bool
    tokens: list[int]
    total_tokens: int
    tok_mismatch: int
    trace_checked: int
    trace_mismatch: int
    timing_err: int
    cycles: int
    raw_output: str
    diff_result: DiffResult | None = None


def int_to_twos_comp_hex(val: int, bits: int) -> str:
    """Format signed integer as two's complement hexadecimal with exact width."""
    mask = (1 << bits) - 1
    v = int(val) & mask
    hex_len = (bits + 3) // 4
    return f"{v:0{hex_len}x}"


def dump_interpreter_traces(
    interp: CompiledArtifactInterpreter,
    out_dir: str | Path,
    seq_id: int,
    tok_idx: int,
    num_layers: int = 2,
) -> int:
    """Dump Python interpreter intermediate traces in standard hierarchical layout."""
    base_dir = Path(out_dir) / f"seq{seq_id}" / f"tok{tok_idx}"
    files_written = 0

    # 1. Embed (Layer 0 container)
    embed_arr = interp.traces.get(f"embed_tokens_tok{interp.traces.get('argmax_tok', [0])[0]}")
    if embed_arr is None:
        # Fallback to any embed trace
        for k, v in interp.traces.items():
            if k.startswith("embed_tokens"):
                embed_arr = v
                break
    if embed_arr is not None:
        l0_dir = base_dir / "L0"
        l0_dir.mkdir(parents=True, exist_ok=True)
        lines = [int_to_twos_comp_hex(int(x), 24) for x in embed_arr.flatten()]
        (l0_dir / "embed_res.hex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        files_written += 1

    # 2. Per-layer tensors
    for l_idx in range(num_layers):
        l_dir = base_dir / f"L{l_idx}"
        l_dir.mkdir(parents=True, exist_ok=True)

        mapping = [
            (f"layer_attn_norm_L{l_idx}", "attn_norm_act.hex", 8),
            (f"layer_q_proj_L{l_idx}", "q_proj_q.hex", 8),
            (f"layer_k_proj_L{l_idx}", "k_proj_k.hex", 8),
            (f"layer_v_proj_L{l_idx}", "v_proj_v.hex", 8),
            (f"layer_q_rope_L{l_idx}", "q_rope_q.hex", 8),
            (f"layer_k_rope_L{l_idx}", "k_rope_k.hex", 8),
            (f"layer_attention_L{l_idx}", "attn_ctx.hex", 8),
            (f"layer_o_proj_L{l_idx}", "o_proj_act.hex", 8),
            (f"layer_res_add_attn_L{l_idx}", "res_attn_res.hex", 24),
            (f"layer_mlp_norm_L{l_idx}", "mlp_norm_act.hex", 8),
            (f"layer_gate_proj_L{l_idx}", "gate_proj_gate.hex", 8),
            (f"layer_up_proj_L{l_idx}", "up_proj_up.hex", 8),
            (f"layer_swiglu_L{l_idx}", "swiglu_swiglu.hex", 8),
            (f"layer_down_proj_L{l_idx}", "down_proj_act.hex", 8),
            (f"layer_res_add_mlp_L{l_idx}", "res_mlp_res.hex", 24),
        ]

        for trace_key, filename, bits in mapping:
            arr = interp.traces.get(trace_key)
            if arr is not None:
                lines = [int_to_twos_comp_hex(int(x), bits) for x in arr.flatten()]
                (l_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
                files_written += 1

    # 3. Final Norm and LM Head
    l0_dir = base_dir / "L0"
    l0_dir.mkdir(parents=True, exist_ok=True)

    fn_arr = interp.traces.get("final_norm_L0")
    if fn_arr is not None:
        lines = [int_to_twos_comp_hex(int(x), 8) for x in fn_arr.flatten()]
        (l0_dir / "final_norm_act.hex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        files_written += 1

    lh_arr = interp.traces.get("lm_head_L0")
    if lh_arr is not None:
        lines = [int_to_twos_comp_hex(int(x), 24) for x in lh_arr.flatten()]
        (l0_dir / "lm_head_logits.hex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        files_written += 1

    return files_written


def generate_top_accel_tb_verilog(
    prompt_tokens: list[int],
    num_generate: int = 1,
    seq_id: int = 0,
    trace_dir: str | Path | None = None,
    expected_tokens: list[int] | None = None,
    timeout_cycles: int = 10000000,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_q_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 16,
    vocab_size: int = 256,
) -> str:
    """Generate plain Verilog testbench tb_top_accel.v for end-to-end execution."""
    total_tokens = len(prompt_tokens) + num_generate
    prompt_len = len(prompt_tokens)
    dump_traces_enabled = 1 if trace_dir is not None else 0
    clean_trace_dir = Path(trace_dir).as_posix() if trace_dir is not None else ""

    exp_toks = expected_tokens if expected_tokens is not None else []

    tb_lines: list[str] = [
        "`timescale 1ns / 1ps",
        "",
        "/* verilator lint_off BLKSEQ */",
        "/* verilator lint_off WIDTHTRUNC */",
        "/* verilator lint_off INITIALDLY */",
        "/* verilator lint_off UNUSEDPARAM */",
        "module tb_top_accel;",
        "    reg clk;",
        "    reg rst_n;",
        "    reg seq_reset;",
        "    reg step_valid;",
        "    wire step_ready;",
        "    reg [15:0] tok_in;",
        "    wire out_valid;",
        "    wire [15:0] tok_out;",
        "",
        "    top_accel dut (",
        "        .clk(clk),",
        "        .rst_n(rst_n),",
        "        .seq_reset(seq_reset),",
        "        .step_valid(step_valid),",
        "        .step_ready(step_ready),",
        "        .tok_in(tok_in),",
        "        .out_valid(out_valid),",
        "        .tok_out(tok_out)",
        "    );",
        "",
        "    // Clock generation: 100MHz (10ns period)",
        "    always #5 clk = ~clk;",
        "",
        "    // Test control and accounting registers",
        f"    localparam TOTAL_STEPS = {total_tokens};",
        f"    localparam PROMPT_LEN   = {prompt_len};",
        f"    localparam SEQ_ID       = {seq_id};",
        f"    localparam TIMEOUT      = {timeout_cycles};",
        "",
        "    integer step_idx;",
        "    integer cycle_cnt;",
        "    integer tok_mismatch_cnt;",
        "    integer timing_err_cnt;",
        "    integer trace_checked_cnt;",
        "    integer trace_mismatch_cnt;",
        "    integer f, i;",
        "",
        f"    reg [15:0] prompt_mem [0:{max(1, prompt_len)-1}];",
        f"    reg [15:0] exp_mem [0:{max(1, len(exp_toks))-1}];",
        f"    reg [15:0] recorded_tokens [0:{total_tokens-1}];",
        "",
        "    initial begin",
    ]

    for idx, tok in enumerate(prompt_tokens):
        tb_lines.append(f"        prompt_mem[{idx}] = 16'd{tok};")

    for idx, tok in enumerate(exp_toks):
        tb_lines.append(f"        exp_mem[{idx}] = 16'd{tok};")

    tb_lines.extend([
        "        clk = 0;",
        "        rst_n = 0;",
        "        seq_reset = 0;",
        "        step_valid = 0;",
        "        tok_in = 16'd0;",
        "        cycle_cnt = 0;",
        "        tok_mismatch_cnt = 0;",
        "        timing_err_cnt = 0;",
        "        trace_checked_cnt = 0;",
        "        trace_mismatch_cnt = 0;",
        "    end",
        "",
        "    // Track cycle count",
        "    always @(posedge clk) begin",
        "        cycle_cnt <= cycle_cnt + 1;",
        "        if (cycle_cnt > TIMEOUT) begin",
        "            $display(\"ERROR: Simulation TIMEOUT exceeded %0d cycles!\", TIMEOUT);",
        "            timing_err_cnt = timing_err_cnt + 1;",
        "            $display(\"TB2_RESULT TOKENS=%0d TOK_MISMATCH=%0d TRACE_CHECKED=%0d TRACE_MISMATCH=%0d TIMING_ERR=%0d\",",
        "                     step_idx, tok_mismatch_cnt, trace_checked_cnt, trace_mismatch_cnt, timing_err_cnt);",
        "            $finish;",
        "        end",
        "    end",
        "",
    ])

    if dump_traces_enabled:
        tb_lines.extend([
            "    // Hierarchical Trace Dumping on Negedge clock when active_unit_done was high",
            "    reg sample_done;",
            "    reg [7:0] sample_pc;",
            "    reg [4:0] sample_layer;",
            "",
            "    always @(posedge clk) begin",
            "        if (rst_n && dut.seq_inst.state == 3'd5 && dut.seq_inst.active_unit_done) begin",
            "            sample_done  <= 1'b1;",
            "            sample_pc    <= dut.cur_pc;",
            "            sample_layer <= dut.cur_layer;",
            "        end else begin",
            "            sample_done  <= 1'b0;",
            "        end",
            "    end",
            "",
            "    always @(negedge clk) begin",
            "        if (sample_done) begin",
            "            case (sample_pc)",
            "                8'd0: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L0/embed_res.hex\", SEQ_ID, step_idx), \"w\");",
            f"                    for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%06x\\n\", dut.buf_res[i][23:0]);",
            "                    $fclose(f);",
            "                end",
            "                8'd1: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/attn_norm_act.hex\", SEQ_ID, step_idx, sample_layer), \"w\");",
            f"                    for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_act[i][7:0]);",
            "                    $fclose(f);",
            "                end",
            "                8'd2: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/q_proj_q.hex\", SEQ_ID, step_idx, sample_layer), \"w\");",
            f"                    for (i = 0; i < {num_q_heads * head_dim}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_q[i]);",
            "                    $fclose(f);",
            "                end",
            "                8'd3: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/k_proj_k.hex\", SEQ_ID, step_idx, sample_layer), \"w\");",
            f"                    for (i = 0; i < {num_kv_heads * head_dim}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_k[i]);",
            "                    $fclose(f);",
            "                end",
            "                8'd4: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/v_proj_v.hex\", SEQ_ID, step_idx, sample_layer), \"w\");",
            f"                    for (i = 0; i < {num_kv_heads * head_dim}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_v[i]);",
            "                    $fclose(f);",
            "                end",
            "                8'd5: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/q_rope_q.hex\", SEQ_ID, step_idx, sample_layer), \"w\");",
            f"                    for (i = 0; i < {num_q_heads * head_dim}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_q[i]);",
            "                    $fclose(f);",
            "                end",
            "                8'd6: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/k_rope_k.hex\", SEQ_ID, step_idx, sample_layer), \"w\");",
            f"                    for (i = 0; i < {num_kv_heads * head_dim}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_k[i]);",
            "                    $fclose(f);",
            "                end",
            "                8'd7: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/attn_ctx.hex\", SEQ_ID, step_idx, sample_layer), \"w\");",
            f"                    for (i = 0; i < {num_q_heads * head_dim}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_attn_out[i]);",
            "                    $fclose(f);",
            "                end",
            "                8'd8: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/o_proj_act.hex\", SEQ_ID, step_idx, sample_layer), \"w\");",
            f"                    for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_act[i][7:0]);",
            "                    $fclose(f);",
            "                end",
            "                8'd9: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/res_attn_res.hex\", SEQ_ID, step_idx, sample_layer), \"w\");",
            f"                    for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%06x\\n\", dut.buf_res[i][23:0]);",
            "                    $fclose(f);",
            "                end",
            "                8'd10: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/mlp_norm_act.hex\", SEQ_ID, step_idx, sample_layer), \"w\");",
            f"                    for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_act[i][7:0]);",
            "                    $fclose(f);",
            "                end",
            "                8'd11: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/gate_proj_gate.hex\", SEQ_ID, step_idx, sample_layer), \"w\");",
            f"                    for (i = 0; i < {intermediate_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_gate[i]);",
            "                    $fclose(f);",
            "                end",
            "                8'd12: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/up_proj_up.hex\", SEQ_ID, step_idx, sample_layer), \"w\");",
            f"                    for (i = 0; i < {intermediate_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_up[i]);",
            "                    $fclose(f);",
            "                end",
            "                8'd13: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/swiglu_swiglu.hex\", SEQ_ID, step_idx, sample_layer), \"w\");",
            f"                    for (i = 0; i < {intermediate_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_swiglu[i]);",
            "                    $fclose(f);",
            "                end",
            "                8'd14: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/down_proj_act.hex\", SEQ_ID, step_idx, sample_layer), \"w\");",
            f"                    for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_act[i][7:0]);",
            "                    $fclose(f);",
            "                end",
            "                8'd15: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/res_mlp_res.hex\", SEQ_ID, step_idx, sample_layer), \"w\");",
            f"                    for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%06x\\n\", dut.buf_res[i][23:0]);",
            "                    $fclose(f);",
            "                end",
            "                8'd16: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L0/final_norm_act.hex\", SEQ_ID, step_idx), \"w\");",
            f"                    for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_act[i][7:0]);",
            "                    $fclose(f);",
            "                end",
            "                8'd17: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L0/lm_head_logits.hex\", SEQ_ID, step_idx), \"w\");",
            f"                    for (i = 0; i < {vocab_size}; i = i + 1) $fwrite(f, \"%06x\\n\", dut.buf_logits[i][23:0]);",
            "                    $fclose(f);",
            "                end",
            "            endcase",
            "        end",
            "    end",
            "",
        ])

    tb_lines.extend([
        "    // Main Stimulus Loop",
        "    initial begin",
        "        #20;",
        "        rst_n = 1;",
        "        #20;",
        "",
        "        for (step_idx = 0; step_idx < TOTAL_STEPS; step_idx = step_idx + 1) begin",
        "            // Wait for step_ready",
        "            while (!step_ready) @(posedge clk);",
        "",
        "            @(posedge clk);",
        "            if (step_idx == 0) begin",
        "                seq_reset <= 1'b1;",
        "            end else begin",
        "                seq_reset <= 1'b0;",
        "            end",
        "            step_valid <= 1'b1;",
        "",
        "            if (step_idx < PROMPT_LEN) begin",
        "                tok_in <= prompt_mem[step_idx];",
        "            end else begin",
        "                tok_in <= recorded_tokens[step_idx - 1];",
        "            end",
        "",
        "            @(posedge clk);",
        "            seq_reset <= 1'b0;",
        "            step_valid <= 1'b0;",
        "",
        "            // Wait for step completion (out_valid)",
        "            while (!out_valid) @(posedge clk);",
        "",
        "            recorded_tokens[step_idx] = tok_out;",
        "            $display(\"STEP %0d TOK_OUT=%0d\", step_idx, tok_out);",
        "",
    ])

    if exp_toks:
        tb_lines.extend([
            f"            if (step_idx < {len(exp_toks)}) begin",
            "                if (tok_out !== exp_mem[step_idx]) begin",
            "                    $display(\"MISMATCH at step %0d: expected=%0d actual=%0d\", step_idx, exp_mem[step_idx], tok_out);",
            "                    tok_mismatch_cnt = tok_mismatch_cnt + 1;",
            "                end",
            "            end",
        ])

    tb_lines.extend([
        "        end",
        "",
        "        // Summary Result Line",
        "        $display(\"TOTAL_CYCLES=%0d\", cycle_cnt);",
        "        $display(\"TB2_RESULT TOKENS=%0d TOK_MISMATCH=%0d TRACE_CHECKED=%0d TRACE_MISMATCH=%0d TIMING_ERR=%0d\",",
        "                 TOTAL_STEPS, tok_mismatch_cnt, trace_checked_cnt, trace_mismatch_cnt, timing_err_cnt);",
        "        $finish;",
        "    end",
        "endmodule",
        "/* verilator lint_on BLKSEQ */",
        "/* verilator lint_on WIDTHTRUNC */",
    ])

    return "\n".join(tb_lines) + "\n"


def run_top_accel_simulation(
    bundle_dir: str | Path,
    prompt_tokens: list[int],
    num_generate: int = 1,
    seq_id: int = 0,
    trace_dir: str | Path | None = None,
    golden_trace_dir: str | Path | None = None,
    expected_tokens: list[int] | None = None,
    simulator: str = "iverilog",
    timeout_cycles: int = 10000000,
    model_config: dict | None = None,
) -> Tb2SimulationResult:
    """Compile and execute top_accel testbench with iverilog or Verilator."""
    b_path = Path(bundle_dir)
    cfg = model_config or {}
    num_layers = cfg.get("num_hidden_layers", 8)
    hidden_size = cfg.get("hidden_size", 64)
    intermediate_size = cfg.get("intermediate_size", 128)
    vocab_size = cfg.get("vocab_size", 256)
    num_q_heads = cfg.get("num_attention_heads", 4)
    num_kv_heads = cfg.get("num_key_value_heads", 2)
    head_dim = cfg.get("head_dim", 16)

    total_tokens = len(prompt_tokens) + num_generate
    t_path = Path(trace_dir) if trace_dir is not None else None
    if t_path is not None:
        for s in range(total_tokens):
            for l in range(num_layers):
                (t_path / f"seq{seq_id}" / f"tok{s}" / f"L{l}").mkdir(parents=True, exist_ok=True)

    # 1. Generate testbench
    tb_code = generate_top_accel_tb_verilog(
        prompt_tokens=prompt_tokens,
        num_generate=num_generate,
        seq_id=seq_id,
        trace_dir=t_path,
        expected_tokens=expected_tokens,
        timeout_cycles=timeout_cycles,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    tb_file = b_path / "tb_top_accel.v"
    tb_file.write_text(tb_code, encoding="utf-8")

    # 2. Gather RTL source files
    repo_root = Path(__file__).resolve().parent.parent
    rtl_lib = repo_root / "rtl" / "lib"
    all_rtl = sorted(rtl_lib.glob("*.v"))
    sources = [p for p in all_rtl if p.name != "top_accel.v"] + [b_path / "top_accel.v", tb_file]

    tokens_generated: list[int] = []
    total_tokens = len(prompt_tokens) + num_generate
    tok_mismatch = 0
    trace_checked = 0
    trace_mismatch = 0
    timing_err = 0
    raw_output = ""

    if simulator == "iverilog":
        iv_bin = shutil.which("iverilog")
        vvp_bin = shutil.which("vvp")
        if not iv_bin or not vvp_bin:
            raise RuntimeError("iverilog or vvp binary not found in PATH")

        vvp_out = b_path / "tb_top_accel.vvp"
        compile_cmd = [
            iv_bin,
            "-g2005",
            "-Wall",
            f"-I{rtl_lib}",
            "-o",
            str(vvp_out),
        ] + [str(s) for s in sources]

        res_c = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
        if res_c.returncode != 0:
            raise RuntimeError(f"iverilog compilation failed:\n{res_c.stderr}\n{res_c.stdout}")

        run_cmd = [vvp_bin, str(vvp_out)]
        res_r = subprocess.run(run_cmd, capture_output=True, text=True, check=False)
        raw_output = res_r.stdout + res_r.stderr

    elif simulator == "verilator":
        ver_bin = shutil.which("verilator")
        if not ver_bin:
            raise RuntimeError("verilator binary not found in PATH")

        sim_dir = b_path / "verilator_tb"
        sim_dir.mkdir(parents=True, exist_ok=True)
        compile_cmd = [
            ver_bin,
            "--binary",
            "--timing",
            "-Wall",
            "-Wno-BLKSEQ",
            "-Wno-WIDTHTRUNC",
            "-Wno-WIDTHEXPAND",
            "-Wno-UNUSEDSIGNAL",
            "-Wno-PINMISSING",
            "-Wno-UNUSEDPARAM",
            "-Wno-INITIALDLY",
            "-Wno-DECLFILENAME",
            "-Wno-CASEINCOMPLETE",
            "-Wno-SYNCASYNCNET",
            "-Wno-UNDRIVEN",
            f"-I{rtl_lib}",
            "--top-module",
            "tb_top_accel",
            "-Mdir",
            str(sim_dir),
        ] + [str(s) for s in sources]

        res_c = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
        if res_c.returncode != 0:
            raise RuntimeError(f"verilator compilation failed:\n{res_c.stderr}\n{res_c.stdout}")

        sim_bin = sim_dir / "Vtb_top_accel"
        res_r = subprocess.run([str(sim_bin)], capture_output=True, text=True, check=False)
        raw_output = res_r.stdout + res_r.stderr

    else:
        raise ValueError(f"Unknown simulator: {simulator}")

    # Extract tokens from STEP output
    for match in re.finditer(r"STEP\s+(\d+)\s+TOK_OUT=(\d+)", raw_output):
        tokens_generated.append(int(match.group(2)))

    # Parse TB2_RESULT line
    tb_match = re.search(
        r"TB2_RESULT\s+TOKENS=(\d+)\s+TOK_MISMATCH=(\d+)\s+TRACE_CHECKED=(\d+)\s+TRACE_MISMATCH=(\d+)\s+TIMING_ERR=(\d+)",
        raw_output,
    )
    if tb_match:
        total_tokens = int(tb_match.group(1))
        tok_mismatch = int(tb_match.group(2))
        trace_checked = int(tb_match.group(3))
        trace_mismatch = int(tb_match.group(4))
        timing_err = int(tb_match.group(5))

    diff_res = None
    if golden_trace_dir is not None and t_path is not None:
        diff_res = compare_trace_directories(Path(golden_trace_dir), t_path)
        trace_checked = diff_res.total_elements_checked
        if not diff_res.is_identical:
            trace_mismatch += 1

    success = (
        (len(tokens_generated) == total_tokens)
        and (tok_mismatch == 0)
        and (trace_mismatch == 0)
        and (timing_err == 0)
    )

    m_cyc = re.search(r"TOTAL_CYCLES=(\d+)", raw_output)
    total_cycles = int(m_cyc.group(1)) if m_cyc else 0

    return Tb2SimulationResult(
        success=success,
        tokens=tokens_generated,
        total_tokens=total_tokens,
        tok_mismatch=tok_mismatch,
        trace_checked=trace_checked,
        trace_mismatch=trace_mismatch,
        timing_err=timing_err,
        cycles=total_cycles,
        raw_output=raw_output,
        diff_result=diff_res,
    )


@dataclass
class Tb2MultiPromptResult:
    """Structured result from multi-prompt accelerator simulation."""
    success: bool
    tokens_by_prompt: list[list[int]]
    all_tokens: list[int]
    total_tokens: int
    tok_mismatch: int
    trace_checked: int
    trace_mismatch: int
    timing_err: int
    raw_output: str
    diff_result: DiffResult | None = None


def generate_multi_prompt_tb_verilog(
    prompts: list[list[int]],
    num_generates: list[int] | int = 1,
    trace_dir: str | Path | None = None,
    trace_seq_ids: list[int] | set[int] | None = None,
    expected_tokens: list[list[int]] | None = None,
    timeout_cycles: int = 50000000,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_q_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 16,
    vocab_size: int = 256,
) -> str:
    """Generate Verilog testbench running multiple prompts back-to-back with seq_reset."""
    num_prompts = len(prompts)
    if num_prompts == 0:
        raise ValueError("prompts list cannot be empty")

    if isinstance(num_generates, int):
        num_gen_list = [num_generates] * num_prompts
    else:
        num_gen_list = list(num_generates)
        if len(num_gen_list) != num_prompts:
            raise ValueError(f"num_generates length ({len(num_gen_list)}) != prompts length ({num_prompts})")

    prompt_lens = [len(p) for p in prompts]
    steps_per_prompt = [prompt_lens[i] + num_gen_list[i] for i in range(num_prompts)]
    total_steps = sum(steps_per_prompt)
    total_prompt_toks = sum(prompt_lens)

    # Calculate offsets
    prompt_offsets: list[int] = []
    p_off = 0
    for l in prompt_lens:
        prompt_offsets.append(p_off)
        p_off += l

    flat_prompt_tokens: list[int] = []
    for p in prompts:
        flat_prompt_tokens.extend(p)

    has_exp = expected_tokens is not None and len(expected_tokens) > 0
    exp_offsets: list[int] = []
    flat_exp_tokens: list[int] = []
    if has_exp:
        assert expected_tokens is not None
        e_off = 0
        for exp in expected_tokens:
            exp_offsets.append(e_off)
            flat_exp_tokens.extend(exp)
            e_off += len(exp)
    else:
        exp_offsets = [0] * num_prompts

    dump_traces_enabled = 1 if trace_dir is not None else 0
    clean_trace_dir = Path(trace_dir).as_posix() if trace_dir is not None else ""
    target_trace_seqs = set(trace_seq_ids) if trace_seq_ids is not None else set(range(num_prompts))

    tb_lines: list[str] = [
        "`timescale 1ns / 1ps",
        "",
        "/* verilator lint_off BLKSEQ */",
        "/* verilator lint_off WIDTHTRUNC */",
        "/* verilator lint_off INITIALDLY */",
        "/* verilator lint_off UNUSEDPARAM */",
        "module tb_top_accel;",
        "    reg clk;",
        "    reg rst_n;",
        "    reg seq_reset;",
        "    reg step_valid;",
        "    wire step_ready;",
        "    reg [15:0] tok_in;",
        "    wire out_valid;",
        "    wire [15:0] tok_out;",
        "",
        "    top_accel dut (",
        "        .clk(clk),",
        "        .rst_n(rst_n),",
        "        .seq_reset(seq_reset),",
        "        .step_valid(step_valid),",
        "        .step_ready(step_ready),",
        "        .tok_in(tok_in),",
        "        .out_valid(out_valid),",
        "        .tok_out(tok_out)",
        "    );",
        "",
        "    // Clock generation: 100MHz (10ns period)",
        "    always #5 clk = ~clk;",
        "",
        "    // Test control parameters and memories",
        f"    localparam NUM_PROMPTS       = {num_prompts};",
        f"    localparam TOTAL_STEPS       = {total_steps};",
        f"    localparam TOTAL_PROMPT_TOKS = {total_prompt_toks};",
        f"    localparam TOTAL_EXP_TOKS    = {max(1, len(flat_exp_tokens))};",
        f"    localparam TIMEOUT           = {timeout_cycles};",
        f"    localparam HAVE_EXP          = {1 if has_exp else 0};",
        "",
        f"    reg [15:0] prompt_mem [0:{max(1, total_prompt_toks)-1}];",
        f"    reg [15:0] exp_mem [0:{max(1, len(flat_exp_tokens))-1}];",
        f"    integer prompt_offsets [0:{num_prompts-1}];",
        f"    integer prompt_lens [0:{num_prompts-1}];",
        f"    integer num_gen_per_prompt [0:{num_prompts-1}];",
        f"    integer exp_offsets [0:{num_prompts-1}];",
        f"    integer dump_seq_mask [0:{num_prompts-1}];",
        "",
        "    integer cycle_cnt;",
        "    integer tok_mismatch_cnt;",
        "    integer timing_err_cnt;",
        "    integer trace_checked_cnt;",
        "    integer trace_mismatch_cnt;",
        "    integer cur_p_idx;",
        "    integer cur_step_idx;",
        "    integer cur_p_len;",
        "    integer cur_num_gen;",
        "    integer cur_p_off;",
        "    integer cur_exp_off;",
        "    reg [15:0] last_gen_token;",
        "    reg [15:0] active_seq_id;",
        "    reg [15:0] active_step_id;",
        "    integer f, i;",
        "",
        "    initial begin",
    ]

    for idx, tok in enumerate(flat_prompt_tokens):
        tb_lines.append(f"        prompt_mem[{idx}] = 16'd{tok};")

    for idx, tok in enumerate(flat_exp_tokens):
        tb_lines.append(f"        exp_mem[{idx}] = 16'd{tok};")

    for i in range(num_prompts):
        tb_lines.append(f"        prompt_offsets[{i}] = {prompt_offsets[i]};")
        tb_lines.append(f"        prompt_lens[{i}] = {prompt_lens[i]};")
        tb_lines.append(f"        num_gen_per_prompt[{i}] = {num_gen_list[i]};")
        tb_lines.append(f"        exp_offsets[{i}] = {exp_offsets[i]};")
        mask_val = 1 if i in target_trace_seqs else 0
        tb_lines.append(f"        dump_seq_mask[{i}] = {mask_val};")

    tb_lines.extend([
        "        clk = 0;",
        "        rst_n = 0;",
        "        seq_reset = 0;",
        "        step_valid = 0;",
        "        tok_in = 16'd0;",
        "        cycle_cnt = 0;",
        "        tok_mismatch_cnt = 0;",
        "        timing_err_cnt = 0;",
        "        trace_checked_cnt = 0;",
        "        trace_mismatch_cnt = 0;",
        "        active_seq_id = 0;",
        "        active_step_id = 0;",
        "    end",
        "",
        "    always @(posedge clk) begin",
        "        cycle_cnt <= cycle_cnt + 1;",
        "        if (cycle_cnt > TIMEOUT) begin",
        "            $display(\"ERROR: Simulation TIMEOUT exceeded %0d cycles!\", TIMEOUT);",
        "            timing_err_cnt = timing_err_cnt + 1;",
        "            $display(\"TB2_RESULT TOKENS=%0d TOK_MISMATCH=%0d TRACE_CHECKED=%0d TRACE_MISMATCH=%0d TIMING_ERR=%0d\",",
        "                     TOTAL_STEPS, tok_mismatch_cnt, trace_checked_cnt, trace_mismatch_cnt, timing_err_cnt);",
        "            $finish;",
        "        end",
        "    end",
        "",
    ])

    if dump_traces_enabled:
        tb_lines.extend([
            "    reg sample_done;",
            "    reg [7:0] sample_pc;",
            "    reg [4:0] sample_layer;",
            "",
            "    always @(posedge clk) begin",
            "        if (rst_n && dut.seq_inst.state == 3'd5 && dut.seq_inst.active_unit_done) begin",
            "            sample_done  <= 1'b1;",
            "            sample_pc    <= dut.cur_pc;",
            "            sample_layer <= dut.cur_layer;",
            "        end else begin",
            "            sample_done  <= 1'b0;",
            "        end",
            "    end",
            "",
            "    always @(negedge clk) begin",
            "        if (sample_done && dump_seq_mask[active_seq_id]) begin",
            "            case (sample_pc)",
            "                8'd0: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L0/embed_res.hex\", active_seq_id, active_step_id), \"w\");",
            f"                    for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%06x\\n\", dut.buf_res[i][23:0]);",
            "                    $fclose(f);",
            "                end",
            "                8'd1: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/attn_norm_act.hex\", active_seq_id, active_step_id, sample_layer), \"w\");",
            f"                    for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_act[i][7:0]);",
            "                    $fclose(f);",
            "                end",
            "                8'd2: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/q_proj_q.hex\", active_seq_id, active_step_id, sample_layer), \"w\");",
            f"                    for (i = 0; i < {num_q_heads * head_dim}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_q[i]);",
            "                    $fclose(f);",
            "                end",
            "                8'd3: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/k_proj_k.hex\", active_seq_id, active_step_id, sample_layer), \"w\");",
            f"                    for (i = 0; i < {num_kv_heads * head_dim}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_k[i]);",
            "                    $fclose(f);",
            "                end",
            "                8'd4: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/v_proj_v.hex\", active_seq_id, active_step_id, sample_layer), \"w\");",
            f"                    for (i = 0; i < {num_kv_heads * head_dim}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_v[i]);",
            "                    $fclose(f);",
            "                end",
            "                8'd5: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/q_rope_q.hex\", active_seq_id, active_step_id, sample_layer), \"w\");",
            f"                    for (i = 0; i < {num_q_heads * head_dim}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_q[i]);",
            "                    $fclose(f);",
            "                end",
            "                8'd6: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/k_rope_k.hex\", active_seq_id, active_step_id, sample_layer), \"w\");",
            f"                    for (i = 0; i < {num_kv_heads * head_dim}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_k[i]);",
            "                    $fclose(f);",
            "                end",
            "                8'd7: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/attn_ctx.hex\", active_seq_id, active_step_id, sample_layer), \"w\");",
            f"                    for (i = 0; i < {num_q_heads * head_dim}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_attn_out[i]);",
            "                    $fclose(f);",
            "                end",
            "                8'd8: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/o_proj_act.hex\", active_seq_id, active_step_id, sample_layer), \"w\");",
            f"                    for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_act[i][7:0]);",
            "                    $fclose(f);",
            "                end",
            "                8'd9: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/res_attn_res.hex\", active_seq_id, active_step_id, sample_layer), \"w\");",
            f"                    for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%06x\\n\", dut.buf_res[i][23:0]);",
            "                    $fclose(f);",
            "                end",
            "                8'd10: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/mlp_norm_act.hex\", active_seq_id, active_step_id, sample_layer), \"w\");",
            f"                    for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_act[i][7:0]);",
            "                    $fclose(f);",
            "                end",
            "                8'd11: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/gate_proj_gate.hex\", active_seq_id, active_step_id, sample_layer), \"w\");",
            f"                    for (i = 0; i < {intermediate_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_gate[i]);",
            "                    $fclose(f);",
            "                end",
            "                8'd12: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/up_proj_up.hex\", active_seq_id, active_step_id, sample_layer), \"w\");",
            f"                    for (i = 0; i < {intermediate_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_up[i]);",
            "                    $fclose(f);",
            "                end",
            "                8'd13: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/swiglu_swiglu.hex\", active_seq_id, active_step_id, sample_layer), \"w\");",
            f"                    for (i = 0; i < {intermediate_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_swiglu[i]);",
            "                    $fclose(f);",
            "                end",
            "                8'd14: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/down_proj_act.hex\", active_seq_id, active_step_id, sample_layer), \"w\");",
            f"                    for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_act[i][7:0]);",
            "                    $fclose(f);",
            "                end",
            "                8'd15: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/res_mlp_res.hex\", active_seq_id, active_step_id, sample_layer), \"w\");",
            f"                    for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%06x\\n\", dut.buf_res[i][23:0]);",
            "                    $fclose(f);",
            "                end",
            "                8'd16: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L0/final_norm_act.hex\", active_seq_id, active_step_id), \"w\");",
            f"                    for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_act[i][7:0]);",
            "                    $fclose(f);",
            "                end",
            "                8'd17: begin",
            f"                    f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L0/lm_head_logits.hex\", active_seq_id, active_step_id), \"w\");",
            f"                    for (i = 0; i < {vocab_size}; i = i + 1) $fwrite(f, \"%06x\\n\", dut.buf_logits[i][23:0]);",
            "                    $fclose(f);",
            "                end",
            "            endcase",
            "        end",
            "    end",
            "",
        ])

    tb_lines.extend([
        "    // Main Multi-Prompt Stimulus Loop",
        "    initial begin",
        "        #20;",
        "        rst_n = 1;",
        "        #20;",
        "",
        "        for (cur_p_idx = 0; cur_p_idx < NUM_PROMPTS; cur_p_idx = cur_p_idx + 1) begin",
        "            active_seq_id = cur_p_idx;",
        "            cur_p_len = prompt_lens[cur_p_idx];",
        "            cur_num_gen = num_gen_per_prompt[cur_p_idx];",
        "            cur_p_off = prompt_offsets[cur_p_idx];",
        "            cur_exp_off = exp_offsets[cur_p_idx];",
        "            last_gen_token = 16'd0;",
        "",
        "            for (cur_step_idx = 0; cur_step_idx < (cur_p_len + cur_num_gen); cur_step_idx = cur_step_idx + 1) begin",
        "                active_step_id = cur_step_idx;",
        "                while (!step_ready) @(posedge clk);",
        "",
        "                @(posedge clk);",
        "                if (cur_step_idx == 0) begin",
        "                    seq_reset <= 1'b1;",
        "                end else begin",
        "                    seq_reset <= 1'b0;",
        "                end",
        "                step_valid <= 1'b1;",
        "",
        "                if (cur_step_idx < cur_p_len) begin",
        "                    tok_in <= prompt_mem[cur_p_off + cur_step_idx];",
        "                end else begin",
        "                    tok_in <= last_gen_token;",
        "                end",
        "",
        "                @(posedge clk);",
        "                seq_reset <= 1'b0;",
        "                step_valid <= 1'b0;",
        "",
        "                while (!out_valid) @(posedge clk);",
        "                last_gen_token = tok_out;",
        "                $display(\"SEQ %0d STEP %0d TOK_OUT=%0d\", cur_p_idx, cur_step_idx, tok_out);",
        "",
        "                if (HAVE_EXP != 0) begin",
        "                    if (tok_out !== exp_mem[cur_exp_off + cur_step_idx]) begin",
        "                        $display(\"MISMATCH seq %0d step %0d: expected=%0d actual=%0d\",",
        "                                 cur_p_idx, cur_step_idx, exp_mem[cur_exp_off + cur_step_idx], tok_out);",
        "                        tok_mismatch_cnt = tok_mismatch_cnt + 1;",
        "                    end",
        "                end",
        "            end",
        "        end",
        "",
        "        $display(\"TB2_RESULT TOKENS=%0d TOK_MISMATCH=%0d TRACE_CHECKED=%0d TRACE_MISMATCH=%0d TIMING_ERR=%0d\",",
        "                 TOTAL_STEPS, tok_mismatch_cnt, trace_checked_cnt, trace_mismatch_cnt, timing_err_cnt);",
        "        $finish;",
        "    end",
        "endmodule",
        "/* verilator lint_on BLKSEQ */",
        "/* verilator lint_on WIDTHTRUNC */",
        "/* verilator lint_on INITIALDLY */",
        "/* verilator lint_on UNUSEDPARAM */",
    ])

    return "\n".join(tb_lines) + "\n"


def run_multi_prompt_simulation(
    bundle_dir: str | Path,
    prompts: list[list[int]],
    num_generates: list[int] | int = 1,
    trace_dir: str | Path | None = None,
    trace_seq_ids: list[int] | set[int] | None = None,
    golden_trace_dir: str | Path | None = None,
    expected_tokens: list[list[int]] | None = None,
    simulator: str = "iverilog",
    timeout_cycles: int = 50000000,
) -> Tb2MultiPromptResult:
    """Compile and execute multi-prompt top_accel testbench with iverilog or Verilator."""
    b_path = Path(bundle_dir)
    num_prompts = len(prompts)
    if isinstance(num_generates, int):
        num_gen_list = [num_generates] * num_prompts
    else:
        num_gen_list = list(num_generates)

    t_path = Path(trace_dir) if trace_dir is not None else None
    target_trace_seqs = set(trace_seq_ids) if trace_seq_ids is not None else set(range(num_prompts))
    if t_path is not None:
        for p_idx in target_trace_seqs:
            tot_s = len(prompts[p_idx]) + num_gen_list[p_idx]
            for s in range(tot_s):
                for l in range(8):
                    (t_path / f"seq{p_idx}" / f"tok{s}" / f"L{l}").mkdir(parents=True, exist_ok=True)

    # 1. Generate testbench
    tb_code = generate_multi_prompt_tb_verilog(
        prompts=prompts,
        num_generates=num_gen_list,
        trace_dir=t_path,
        trace_seq_ids=trace_seq_ids,
        expected_tokens=expected_tokens,
        timeout_cycles=timeout_cycles,
    )
    tb_file = b_path / "tb_multi_top_accel.v"
    tb_file.write_text(tb_code, encoding="utf-8")

    # 2. Gather RTL sources
    repo_root = Path(__file__).resolve().parent.parent
    rtl_lib = repo_root / "rtl" / "lib"
    all_rtl = sorted(rtl_lib.glob("*.v"))
    sources = [p for p in all_rtl if p.name != "top_accel.v"] + [b_path / "top_accel.v", tb_file]

    tokens_by_prompt: list[list[int]] = [[] for _ in range(num_prompts)]
    all_tokens: list[int] = []
    total_tokens = sum(len(prompts[i]) + num_gen_list[i] for i in range(num_prompts))
    tok_mismatch = 0
    trace_checked = 0
    trace_mismatch = 0
    timing_err = 0
    raw_output = ""

    if simulator == "iverilog":
        iv_bin = shutil.which("iverilog")
        vvp_bin = shutil.which("vvp")
        if not iv_bin or not vvp_bin:
            raise RuntimeError("iverilog or vvp binary not found in PATH")

        vvp_out = b_path / "tb_multi_top_accel.vvp"
        compile_cmd = [
            iv_bin,
            "-g2005",
            "-Wall",
            f"-I{rtl_lib}",
            "-o",
            str(vvp_out),
        ] + [str(s) for s in sources]

        res_c = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
        if res_c.returncode != 0:
            raise RuntimeError(f"iverilog compilation failed:\n{res_c.stderr}\n{res_c.stdout}")

        run_cmd = [vvp_bin, str(vvp_out)]
        res_r = subprocess.run(run_cmd, capture_output=True, text=True, check=False)
        raw_output = res_r.stdout + res_r.stderr

    elif simulator == "verilator":
        ver_bin = shutil.which("verilator")
        if not ver_bin:
            raise RuntimeError("verilator binary not found in PATH")

        sim_dir = b_path / "verilator_multi_tb"
        sim_dir.mkdir(parents=True, exist_ok=True)
        compile_cmd = [
            ver_bin,
            "--binary",
            "--timing",
            "-Wall",
            "-Wno-BLKSEQ",
            "-Wno-WIDTHTRUNC",
            "-Wno-WIDTHEXPAND",
            "-Wno-UNUSEDSIGNAL",
            "-Wno-PINMISSING",
            "-Wno-UNUSEDPARAM",
            "-Wno-INITIALDLY",
            "-Wno-DECLFILENAME",
            "-Wno-CASEINCOMPLETE",
            "-Wno-SYNCASYNCNET",
            "-Wno-UNDRIVEN",
            f"-I{rtl_lib}",
            "--top-module",
            "tb_top_accel",
            "-Mdir",
            str(sim_dir),
        ] + [str(s) for s in sources]

        res_c = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
        if res_c.returncode != 0:
            raise RuntimeError(f"verilator compilation failed:\n{res_c.stderr}\n{res_c.stdout}")

        sim_bin = sim_dir / "Vtb_top_accel"
        res_r = subprocess.run([str(sim_bin)], capture_output=True, text=True, check=False)
        raw_output = res_r.stdout + res_r.stderr

    else:
        raise ValueError(f"Unknown simulator: {simulator}")

    for match in re.finditer(r"SEQ\s+(\d+)\s+STEP\s+(\d+)\s+TOK_OUT=(\d+)", raw_output):
        s_id = int(match.group(1))
        tok = int(match.group(3))
        if s_id < num_prompts:
            tokens_by_prompt[s_id].append(tok)
        all_tokens.append(tok)

    tb_match = re.search(
        r"TB2_RESULT\s+TOKENS=(\d+)\s+TOK_MISMATCH=(\d+)\s+TRACE_CHECKED=(\d+)\s+TRACE_MISMATCH=(\d+)\s+TIMING_ERR=(\d+)",
        raw_output,
    )
    if tb_match:
        total_tokens = int(tb_match.group(1))
        tok_mismatch = int(tb_match.group(2))
        trace_checked = int(tb_match.group(3))
        trace_mismatch = int(tb_match.group(4))
        timing_err = int(tb_match.group(5))

    diff_res = None
    if golden_trace_dir is not None and t_path is not None:
        diff_res = compare_trace_directories(Path(golden_trace_dir), t_path)
        trace_checked = diff_res.total_elements_checked
        if not diff_res.is_identical:
            trace_mismatch += 1

    success = (
        (len(all_tokens) == total_tokens)
        and (tok_mismatch == 0)
        and (trace_mismatch == 0)
        and (timing_err == 0)
    )

    return Tb2MultiPromptResult(
        success=success,
        tokens_by_prompt=tokens_by_prompt,
        all_tokens=all_tokens,
        total_tokens=total_tokens,
        tok_mismatch=tok_mismatch,
        trace_checked=trace_checked,
        trace_mismatch=trace_mismatch,
        timing_err=timing_err,
        raw_output=raw_output,
        diff_result=diff_res,
    )

