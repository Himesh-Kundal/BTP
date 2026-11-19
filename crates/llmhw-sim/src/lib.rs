//! Testbench Generator & Simulator Runner (Icarus Verilog & Verilator) (Phase R4).
//!
//! Strict Layering (Golden Rule 6): No ML framework dependencies.

use llmhw_compiler::{int_to_twos_comp_hex, CompiledArtifactInterpreter};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tb2SimulationResult {
    pub success: bool,
    pub tokens: Vec<usize>,
    pub total_tokens: usize,
    pub tok_mismatch: usize,
    pub trace_checked: usize,
    pub trace_mismatch: usize,
    pub timing_err: usize,
    pub cycles: u64,
    pub sim_sec: f64,
    pub raw_output: String,
}

/// Dump `CompiledArtifactInterpreter` intermediate traces in standard hierarchical layout.
pub fn dump_interpreter_traces(
    interp: &CompiledArtifactInterpreter,
    out_dir: &Path,
    seq_id: usize,
    tok_idx: usize,
    num_layers: usize,
) -> anyhow::Result<usize> {
    let base_dir = out_dir.join(format!("seq{seq_id}")).join(format!("tok{tok_idx}"));
    let mut files_written = 0usize;

    // 1. Embed (Layer 0 container)
    if let Some(embed_arr) = interp.traces.get("embed_tokens_current") {
        let l0_dir = base_dir.join("L0");
        fs::create_dir_all(&l0_dir)?;
        let lines: Vec<String> = embed_arr
            .iter()
            .map(|&x| int_to_twos_comp_hex(x, 24))
            .collect();
        fs::write(l0_dir.join("embed_res.hex"), format!("{}\n", lines.join("\n")))?;
        files_written += 1;
    }

    // 2. Per-layer tensors
    for l_idx in 0..num_layers {
        let l_dir = base_dir.join(format!("L{l_idx}"));
        fs::create_dir_all(&l_dir)?;

        let mapping = [
            (format!("layer_attn_norm_L{l_idx}"), "attn_norm_act.hex", 8u32),
            (format!("layer_q_proj_L{l_idx}"), "q_proj_q.hex", 8),
            (format!("layer_k_proj_L{l_idx}"), "k_proj_k.hex", 8),
            (format!("layer_v_proj_L{l_idx}"), "v_proj_v.hex", 8),
            (format!("layer_q_rope_L{l_idx}"), "q_rope_q.hex", 8),
            (format!("layer_k_rope_L{l_idx}"), "k_rope_k.hex", 8),
            (format!("layer_attention_L{l_idx}"), "attn_ctx.hex", 8),
            (format!("layer_o_proj_L{l_idx}"), "o_proj_act.hex", 8),
            (format!("layer_res_add_attn_L{l_idx}"), "res_attn_res.hex", 24),
            (format!("layer_mlp_norm_L{l_idx}"), "mlp_norm_act.hex", 8),
            (format!("layer_gate_proj_L{l_idx}"), "gate_proj_gate.hex", 8),
            (format!("layer_up_proj_L{l_idx}"), "up_proj_up.hex", 8),
            (format!("layer_swiglu_L{l_idx}"), "swiglu_swiglu.hex", 8),
            (format!("layer_down_proj_L{l_idx}"), "down_proj_act.hex", 8),
            (format!("layer_res_add_mlp_L{l_idx}"), "res_mlp_res.hex", 24),
        ];

        for (trace_key, filename, bits) in mapping {
            if let Some(arr) = interp.traces.get(&trace_key) {
                let lines: Vec<String> = arr.iter().map(|&x| int_to_twos_comp_hex(x, bits)).collect();
                fs::write(l_dir.join(filename), format!("{}\n", lines.join("\n")))?;
                files_written += 1;
            }
        }
    }

    // 3. Final Norm and LM Head
    let l0_dir = base_dir.join("L0");
    fs::create_dir_all(&l0_dir)?;
    if let Some(fn_arr) = interp.traces.get("final_norm_L0") {
        let lines: Vec<String> = fn_arr.iter().map(|&x| int_to_twos_comp_hex(x, 8)).collect();
        fs::write(l0_dir.join("final_norm_act.hex"), format!("{}\n", lines.join("\n")))?;
        files_written += 1;
    }
    if let Some(lh_arr) = interp.traces.get("lm_head_L0") {
        let lines: Vec<String> = lh_arr.iter().map(|&x| int_to_twos_comp_hex(x, 24)).collect();
        fs::write(l0_dir.join("lm_head_logits.hex"), format!("{}\n", lines.join("\n")))?;
        files_written += 1;
    }

    Ok(files_written)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_top_accel_tb_verilog(
    prompt_tokens: &[usize],
    num_generate: usize,
    seq_id: usize,
    trace_dir: Option<&Path>,
    expected_tokens: Option<&[usize]>,
    timeout_cycles: u64,
    hidden_size: usize,
    intermediate_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
) -> String {
    let total_tokens = prompt_tokens.len() + num_generate;
    let prompt_len = prompt_tokens.len();
    let clean_trace_dir = trace_dir
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_default();
    let exp_toks = expected_tokens.unwrap_or(&[]);

    let mut tb = Vec::new();
    tb.push("`timescale 1ns / 1ps\n".to_string());
    tb.push("/* verilator lint_off BLKSEQ */".to_string());
    tb.push("/* verilator lint_off WIDTHTRUNC */".to_string());
    tb.push("/* verilator lint_off INITIALDLY */".to_string());
    tb.push("/* verilator lint_off UNUSEDPARAM */".to_string());
    tb.push("module tb_top_accel;".to_string());
    tb.push("    reg clk;".to_string());
    tb.push("    reg rst_n;".to_string());
    tb.push("    reg seq_reset;".to_string());
    tb.push("    reg step_valid;".to_string());
    tb.push("    wire step_ready;".to_string());
    tb.push("    reg [15:0] tok_in;".to_string());
    tb.push("    wire out_valid;".to_string());
    tb.push("    wire [15:0] tok_out;\n".to_string());
    tb.push("    top_accel dut (".to_string());
    tb.push("        .clk(clk),".to_string());
    tb.push("        .rst_n(rst_n),".to_string());
    tb.push("        .seq_reset(seq_reset),".to_string());
    tb.push("        .step_valid(step_valid),".to_string());
    tb.push("        .step_ready(step_ready),".to_string());
    tb.push("        .tok_in(tok_in),".to_string());
    tb.push("        .out_valid(out_valid),".to_string());
    tb.push("        .tok_out(tok_out)".to_string());
    tb.push("    );\n".to_string());
    tb.push("    always #5 clk = ~clk;\n".to_string());
    tb.push(format!("    localparam TOTAL_STEPS = {total_tokens};"));
    tb.push(format!("    localparam PROMPT_LEN   = {prompt_len};"));
    tb.push(format!("    localparam SEQ_ID       = {seq_id};"));
    tb.push(format!("    localparam [63:0] TIMEOUT = 64'd{timeout_cycles};\n"));
    tb.push("    integer step_idx;".to_string());
    tb.push("    reg [63:0] cycle_cnt;".to_string());
    tb.push("    integer tok_mismatch_cnt;".to_string());
    tb.push("    integer timing_err_cnt;".to_string());
    tb.push("    integer trace_checked_cnt;".to_string());
    tb.push("    integer trace_mismatch_cnt;".to_string());
    tb.push("    integer f, i;\n".to_string());
    tb.push(format!("    reg [15:0] prompt_mem [0:{}];", prompt_len.max(1) - 1));
    tb.push(format!("    reg [15:0] exp_mem [0:{}];", exp_toks.len().max(1) - 1));
    tb.push(format!("    reg [15:0] recorded_tokens [0:{}];\n", total_tokens - 1));
    tb.push("    initial begin".to_string());

    for (idx, &tok) in prompt_tokens.iter().enumerate() {
        tb.push(format!("        prompt_mem[{idx}] = 16'd{tok};"));
    }
    for (idx, &tok) in exp_toks.iter().enumerate() {
        tb.push(format!("        exp_mem[{idx}] = 16'd{tok};"));
    }
    tb.push("        clk = 0;".to_string());
    tb.push("        rst_n = 0;".to_string());
    tb.push("        seq_reset = 0;".to_string());
    tb.push("        step_valid = 0;".to_string());
    tb.push("        tok_in = 16'd0;".to_string());
    tb.push("        cycle_cnt = 0;".to_string());
    tb.push("        tok_mismatch_cnt = 0;".to_string());
    tb.push("        timing_err_cnt = 0;".to_string());
    tb.push("        trace_checked_cnt = 0;".to_string());
    tb.push("        trace_mismatch_cnt = 0;".to_string());
    tb.push("    end\n".to_string());

    tb.push("    always @(posedge clk) begin".to_string());
    tb.push("        cycle_cnt <= cycle_cnt + 1;".to_string());
    tb.push("        if (cycle_cnt > TIMEOUT) begin".to_string());
    tb.push("            $display(\"ERROR: Simulation TIMEOUT exceeded %0d cycles!\", TIMEOUT);".to_string());
    tb.push("            timing_err_cnt = timing_err_cnt + 1;".to_string());
    tb.push("            $display(\"TB2_RESULT TOKENS=%0d TOK_MISMATCH=%0d TRACE_CHECKED=%0d TRACE_MISMATCH=%0d TIMING_ERR=%0d\",".to_string());
    tb.push("                     step_idx, tok_mismatch_cnt, trace_checked_cnt, trace_mismatch_cnt, timing_err_cnt);".to_string());
    tb.push("            $finish;".to_string());
    tb.push("        end".to_string());
    tb.push("    end\n".to_string());

    if trace_dir.is_some() {
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        tb.push("    reg sample_done;".to_string());
        tb.push("    reg [7:0] sample_pc;".to_string());
        tb.push("    reg [4:0] sample_layer;\n".to_string());
        tb.push("    always @(posedge clk) begin".to_string());
        tb.push("        if (rst_n && dut.seq_inst.state == 3'd5 && dut.seq_inst.active_unit_done) begin".to_string());
        tb.push("            sample_done  <= 1'b1;".to_string());
        tb.push("            sample_pc    <= dut.cur_pc;".to_string());
        tb.push("            sample_layer <= dut.cur_layer;".to_string());
        tb.push("        end else begin".to_string());
        tb.push("            sample_done  <= 1'b0;".to_string());
        tb.push("        end".to_string());
        tb.push("    end\n".to_string());
        tb.push("    always @(negedge clk) begin".to_string());
        tb.push("        if (sample_done) begin".to_string());
        tb.push("            case (sample_pc)".to_string());
        tb.push(format!("                8'd0: begin f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L0/embed_res.hex\", SEQ_ID, step_idx), \"w\"); for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%06x\\n\", dut.buf_res[i][23:0]); $fclose(f); end"));
        tb.push(format!("                8'd1: begin f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/attn_norm_act.hex\", SEQ_ID, step_idx, sample_layer), \"w\"); for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_act[i][7:0]); $fclose(f); end"));
        tb.push(format!("                8'd2: begin f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/q_proj_q.hex\", SEQ_ID, step_idx, sample_layer), \"w\"); for (i = 0; i < {q_dim}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_q[i]); $fclose(f); end"));
        tb.push(format!("                8'd3: begin f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/k_proj_k.hex\", SEQ_ID, step_idx, sample_layer), \"w\"); for (i = 0; i < {kv_dim}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_k[i]); $fclose(f); end"));
        tb.push(format!("                8'd4: begin f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/v_proj_v.hex\", SEQ_ID, step_idx, sample_layer), \"w\"); for (i = 0; i < {kv_dim}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_v[i]); $fclose(f); end"));
        tb.push(format!("                8'd5: begin f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/q_rope_q.hex\", SEQ_ID, step_idx, sample_layer), \"w\"); for (i = 0; i < {q_dim}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_q[i]); $fclose(f); end"));
        tb.push(format!("                8'd6: begin f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/k_rope_k.hex\", SEQ_ID, step_idx, sample_layer), \"w\"); for (i = 0; i < {kv_dim}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_k[i]); $fclose(f); end"));
        tb.push(format!("                8'd7: begin f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/attn_ctx.hex\", SEQ_ID, step_idx, sample_layer), \"w\"); for (i = 0; i < {q_dim}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_attn_out[i]); $fclose(f); end"));
        tb.push(format!("                8'd8: begin f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/o_proj_act.hex\", SEQ_ID, step_idx, sample_layer), \"w\"); for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_act[i][7:0]); $fclose(f); end"));
        tb.push(format!("                8'd9: begin f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/res_attn_res.hex\", SEQ_ID, step_idx, sample_layer), \"w\"); for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%06x\\n\", dut.buf_res[i][23:0]); $fclose(f); end"));
        tb.push(format!("                8'd10: begin f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/mlp_norm_act.hex\", SEQ_ID, step_idx, sample_layer), \"w\"); for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_act[i][7:0]); $fclose(f); end"));
        tb.push(format!("                8'd11: begin f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/gate_proj_gate.hex\", SEQ_ID, step_idx, sample_layer), \"w\"); for (i = 0; i < {intermediate_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_gate[i]); $fclose(f); end"));
        tb.push(format!("                8'd12: begin f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/up_proj_up.hex\", SEQ_ID, step_idx, sample_layer), \"w\"); for (i = 0; i < {intermediate_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_up[i]); $fclose(f); end"));
        tb.push(format!("                8'd13: begin f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/swiglu_swiglu.hex\", SEQ_ID, step_idx, sample_layer), \"w\"); for (i = 0; i < {intermediate_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_swiglu[i]); $fclose(f); end"));
        tb.push(format!("                8'd14: begin f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/down_proj_act.hex\", SEQ_ID, step_idx, sample_layer), \"w\"); for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_act[i][7:0]); $fclose(f); end"));
        tb.push(format!("                8'd15: begin f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L%0d/res_mlp_res.hex\", SEQ_ID, step_idx, sample_layer), \"w\"); for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%06x\\n\", dut.buf_res[i][23:0]); $fclose(f); end"));
        tb.push(format!("                8'd16: begin f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L0/final_norm_act.hex\", SEQ_ID, step_idx), \"w\"); for (i = 0; i < {hidden_size}; i = i + 1) $fwrite(f, \"%02x\\n\", dut.buf_act[i][7:0]); $fclose(f); end"));
        tb.push(format!("                8'd17: begin f = $fopen($sformatf(\"{clean_trace_dir}/seq%0d/tok%0d/L0/lm_head_logits.hex\", SEQ_ID, step_idx), \"w\"); for (i = 0; i < {vocab_size}; i = i + 1) $fwrite(f, \"%06x\\n\", dut.buf_logits[i][23:0]); $fclose(f); end"));
        tb.push("            endcase".to_string());
        tb.push("        end".to_string());
        tb.push("    end\n".to_string());
    }

    tb.push("    initial begin".to_string());
    tb.push("        #20;".to_string());
    tb.push("        rst_n = 1;".to_string());
    tb.push("        #20;".to_string());
    tb.push("        for (step_idx = 0; step_idx < TOTAL_STEPS; step_idx = step_idx + 1) begin".to_string());
    tb.push("            while (!step_ready) @(posedge clk);".to_string());
    tb.push("            @(posedge clk);".to_string());
    tb.push("            if (step_idx == 0) seq_reset <= 1'b1; else seq_reset <= 1'b0;".to_string());
    tb.push("            step_valid <= 1'b1;".to_string());
    tb.push("            if (step_idx < PROMPT_LEN) tok_in <= prompt_mem[step_idx]; else tok_in <= recorded_tokens[step_idx - 1];".to_string());
    tb.push("            @(posedge clk);".to_string());
    tb.push("            seq_reset <= 1'b0;".to_string());
    tb.push("            step_valid <= 1'b0;".to_string());
    tb.push("            while (!out_valid) @(posedge clk);".to_string());
    tb.push("            recorded_tokens[step_idx] = tok_out;".to_string());
    tb.push("            $display(\"STEP %0d TOK_OUT=%0d\", step_idx, tok_out);".to_string());

    if !exp_toks.is_empty() {
        tb.push(format!("            if (step_idx < {}) begin", exp_toks.len()));
        tb.push("                if (tok_out !== exp_mem[step_idx]) begin".to_string());
        tb.push("                    $display(\"MISMATCH at step %0d: expected=%0d actual=%0d\", step_idx, exp_mem[step_idx], tok_out);".to_string());
        tb.push("                    tok_mismatch_cnt = tok_mismatch_cnt + 1;".to_string());
        tb.push("                end".to_string());
        tb.push("            end".to_string());
    }

    tb.push("        end".to_string());
    tb.push("        $display(\"TOTAL_CYCLES=%0d\", cycle_cnt);".to_string());
    tb.push("        $display(\"TB2_RESULT TOKENS=%0d TOK_MISMATCH=%0d TRACE_CHECKED=%0d TRACE_MISMATCH=%0d TIMING_ERR=%0d\",".to_string());
    tb.push("                 TOTAL_STEPS, tok_mismatch_cnt, trace_checked_cnt, trace_mismatch_cnt, timing_err_cnt);".to_string());
    tb.push("        $finish;".to_string());
    tb.push("    end".to_string());
    tb.push("endmodule".to_string());
    tb.push("/* verilator lint_on BLKSEQ */".to_string());
    tb.push("/* verilator lint_on WIDTHTRUNC */\n".to_string());

    tb.join("\n")
}

#[allow(clippy::too_many_arguments)]
pub fn run_top_accel_simulation(
    bundle_dir: &Path,
    rtl_lib_dir: &Path,
    prompt_tokens: &[usize],
    num_generate: usize,
    seq_id: usize,
    trace_dir: Option<&Path>,
    expected_tokens: Option<&[usize]>,
    simulator: &str,
    timeout_cycles: u64,
    model_config: &Value,
    work_dir: Option<&Path>,
) -> anyhow::Result<Tb2SimulationResult> {
    let get_usize = |k: &str, def: usize| -> usize {
        model_config.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(def)
    };
    let num_layers = get_usize("num_hidden_layers", 2);
    let hidden_size = get_usize("hidden_size", 64);
    let intermediate_size = get_usize("intermediate_size", 128);
    let vocab_size = get_usize("vocab_size", 256);
    let num_q_heads = get_usize("num_attention_heads", 4);
    let num_kv_heads = get_usize("num_key_value_heads", 2);
    let head_dim = get_usize("head_dim", 16);

    let total_tokens = prompt_tokens.len() + num_generate;
    if let Some(t_path) = trace_dir {
        for s in 0..total_tokens {
            for l in 0..num_layers {
                fs::create_dir_all(t_path.join(format!("seq{seq_id}")).join(format!("tok{s}")).join(format!("L{l}")))?;
            }
        }
    }

    let tb_code = generate_top_accel_tb_verilog(
        prompt_tokens,
        num_generate,
        seq_id,
        trace_dir,
        expected_tokens,
        timeout_cycles,
        hidden_size,
        intermediate_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        vocab_size,
    );

    let temp_dir_holder;
    let sim_work_dir = if let Some(w) = work_dir {
        fs::create_dir_all(w)?;
        w
    } else {
        temp_dir_holder = tempfile::tempdir()?;
        temp_dir_holder.path()
    };

    let tb_file = sim_work_dir.join("tb_top_accel.v");
    fs::write(&tb_file, tb_code)?;

    let mut sources = Vec::new();
    for entry in fs::read_dir(rtl_lib_dir)? {
        let p = entry?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("v")
            && p.file_name().and_then(|n| n.to_str()) != Some("top_accel.v")
        {
            sources.push(p);
        }
    }
    sources.sort();
    sources.push(bundle_dir.join("top_accel.v"));
    sources.push(tb_file);

    let t0 = Instant::now();
    let raw_output = if simulator == "iverilog" {
        let vvp_out = sim_work_dir.join("tb_top_accel.vvp");
        let mut cmd = Command::new("iverilog");
        cmd.arg("-g2005")
            .arg("-Wall")
            .arg(format!("-I{}", rtl_lib_dir.display()))
            .arg("-o")
            .arg(&vvp_out);
        for s in &sources {
            cmd.arg(s);
        }
        let out_c = cmd.output()?;
        if !out_c.status.success() {
            anyhow::bail!(
                "iverilog compile failed:\n{}",
                String::from_utf8_lossy(&out_c.stderr)
            );
        }
        let out_r = Command::new("vvp").arg(&vvp_out).output()?;
        format!(
            "{}{}",
            String::from_utf8_lossy(&out_r.stdout),
            String::from_utf8_lossy(&out_r.stderr)
        )
    } else if simulator == "verilator" {
        let obj_dir = sim_work_dir.join("obj_dir");
        let mut cmd = Command::new("verilator");
        cmd.args([
            "--binary",
            "--timing",
            "-j",
            "4",
            "-CFLAGS",
            "-O3",
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
            &format!("-I{}", rtl_lib_dir.display()),
            "--top-module",
            "tb_top_accel",
            "-Mdir",
        ])
        .arg(&obj_dir);
        for s in &sources {
            cmd.arg(s);
        }
        let out_c = cmd.output()?;
        if !out_c.status.success() {
            anyhow::bail!(
                "verilator compile failed:\n{}{}",
                String::from_utf8_lossy(&out_c.stderr),
                String::from_utf8_lossy(&out_c.stdout)
            );
        }
        let out_r = Command::new(obj_dir.join("Vtb_top_accel")).output()?;
        format!(
            "{}{}",
            String::from_utf8_lossy(&out_r.stdout),
            String::from_utf8_lossy(&out_r.stderr)
        )
    } else {
        anyhow::bail!("Unsupported simulator: {simulator}");
    };
    let sim_sec = t0.elapsed().as_secs_f64();

    let re_step = Regex::new(r"STEP\s+\d+\s+TOK_OUT=(\d+)").unwrap();
    let tokens: Vec<usize> = re_step
        .captures_iter(&raw_output)
        .filter_map(|c| c[1].parse::<usize>().ok())
        .collect();

    let re_res = Regex::new(
        r"TB2_RESULT\s+TOKENS=(\d+)\s+TOK_MISMATCH=(\d+)\s+TRACE_CHECKED=(\d+)\s+TRACE_MISMATCH=(\d+)\s+TIMING_ERR=(\d+)",
    )
    .unwrap();
    let (mut tok_mismatch, mut trace_checked, mut trace_mismatch, mut timing_err) = (0, 0, 0, 0);
    if let Some(cap) = re_res.captures(&raw_output) {
        tok_mismatch = cap[2].parse().unwrap_or(0);
        trace_checked = cap[3].parse().unwrap_or(0);
        trace_mismatch = cap[4].parse().unwrap_or(0);
        timing_err = cap[5].parse().unwrap_or(0);
    }

    let re_cyc = Regex::new(r"TOTAL_CYCLES=(\d+)").unwrap();
    let cycles = re_cyc
        .captures(&raw_output)
        .and_then(|c| c[1].parse::<u64>().ok())
        .unwrap_or(0);

    let success = tokens.len() == total_tokens
        && tok_mismatch == 0
        && trace_mismatch == 0
        && timing_err == 0;

    Ok(Tb2SimulationResult {
        success,
        tokens,
        total_tokens,
        tok_mismatch,
        trace_checked,
        trace_mismatch,
        timing_err,
        cycles,
        sim_sec,
        raw_output,
    })
}
