"""Unit tests for Parameterized Sequencer & Microprogram Dispatcher (Task UC.12).

Verifies bit-for-bit and step-for-step execution against `compiler/interpreter.py`:
1. Microprogram instruction decoding from `program.hex` (128-bit words).
2. Compact layer-loop execution: prologue (embed) -> N layer iterations -> epilogue (final_norm, lm_head, argmax).
3. Unit protocol dispatch: pulses start to active unit and advances upon receiving done.
4. Output next-token latching on `tok_out` when `argmax` finishes.
5. Opcode and buffer dispatch trace strictly matches Python interpreter step for step.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from compiler.format_assignment import assign_formats_and_scales
from compiler.interpreter import CompiledArtifactInterpreter
from compiler.rom_packing import pack_rom_artifacts
from compiler.schedule import Opcode, generate_schedule
from frontend.importer_v2 import build_graph_v2_from_artifacts, load_model_artifacts_v2

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_model_v2"
CONFIG_FROZEN_PATH = REPO_ROOT / "numerics" / "config_frozen.json"


def run_sequencer_sim(
    program_hex_lines: list[str],
    layer_loop_start: int,
    layer_loop_end: int,
    num_layers: int,
    expected_tok: int = 42,
) -> tuple[list[dict[str, Any]], int]:
    """Simulate sequencer.v and record all dispatched operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "program.hex").write_text("\n".join(program_hex_lines) + "\n")

        tb_code = f"""
`timescale 1ns/1ps
module tb_sequencer;
    reg clk;
    reg rst_n;
    reg start;
    wire busy;
    wire done;
    wire [15:0] tok_out;
    wire tok_out_valid;

    reg [127:0] prog_mem [0:255];
    wire [7:0] prog_rom_addr;
    reg [127:0] prog_rom_data;

    wire [7:0] layer_table_addr;
    reg [31:0] layer_table_weight_base;
    reg [31:0] layer_table_param_base;

    wire embed_start, norm_start, linear_start, rope_start, attn_start, add_start, swiglu_start, argmax_start;
    reg embed_done, norm_done, linear_done, rope_done, attn_done, add_done, swiglu_done, argmax_done;
    reg [15:0] argmax_res_tok;

    wire [7:0] cur_pc;
    wire [4:0] cur_layer;
    wire [7:0] cur_opcode;
    wire [7:0] cur_src_buf;
    wire [7:0] cur_dst_buf;
    wire [7:0] cur_flags;
    wire [15:0] cur_dim0;
    wire [15:0] cur_dim1;
    wire [31:0] cur_weight_base;
    wire [31:0] cur_param_base;

    initial begin
        $readmemh("{tmp_path}/program.hex", prog_mem);
        layer_table_weight_base = 32'd0;
        layer_table_param_base = 32'd0;
        argmax_res_tok = 16'd{expected_tok};
    end

    // 1-cycle synchronous SRAM model for Program ROM
    always @(posedge clk) begin
        prog_rom_data <= prog_mem[prog_rom_addr];
    end

    sequencer #(
        .PROG_ADDR_W(8),
        .LAYER_W(5),
        .BUF_ID_W(8),
        .DIM_W(16),
        .ROM_ADDR_W(32)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .busy(busy),
        .done(done),
        .pos(16'd0),
        .tok_in(16'd12),
        .tok_out(tok_out),
        .tok_out_valid(tok_out_valid),
        .layer_loop_start(8'd{layer_loop_start}),
        .layer_loop_end(8'd{layer_loop_end}),
        .num_layers(5'd{num_layers}),
        .prog_rom_addr(prog_rom_addr),
        .prog_rom_data(prog_rom_data),
        .layer_table_addr(layer_table_addr),
        .layer_table_weight_base(layer_table_weight_base),
        .layer_table_param_base(layer_table_param_base),
        .embed_start(embed_start),
        .embed_done(embed_done),
        .norm_start(norm_start),
        .norm_done(norm_done),
        .linear_start(linear_start),
        .linear_done(linear_done),
        .rope_start(rope_start),
        .rope_done(rope_done),
        .attn_start(attn_start),
        .attn_done(attn_done),
        .add_start(add_start),
        .add_done(add_done),
        .swiglu_start(swiglu_start),
        .swiglu_done(swiglu_done),
        .argmax_start(argmax_start),
        .argmax_done(argmax_done),
        .argmax_res_tok(argmax_res_tok),
        .cur_pc(cur_pc),
        .cur_layer(cur_layer),
        .cur_opcode(cur_opcode),
        .cur_src_buf(cur_src_buf),
        .cur_dst_buf(cur_dst_buf),
        .cur_flags(cur_flags),
        .cur_dim0(cur_dim0),
        .cur_dim1(cur_dim1),
        .cur_weight_base(cur_weight_base),
        .cur_param_base(cur_param_base)
    );

    always #5 clk = ~clk;

    // Unit execution mock: pulse done 2 cycles after start pulse
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            embed_done <= 1'b0;
            norm_done <= 1'b0;
            linear_done <= 1'b0;
            rope_done <= 1'b0;
            attn_done <= 1'b0;
            add_done <= 1'b0;
            swiglu_done <= 1'b0;
            argmax_done <= 1'b0;
        end else begin
            embed_done <= embed_start;
            norm_done <= norm_start;
            linear_done <= linear_start;
            rope_done <= rope_start;
            attn_done <= attn_start;
            add_done <= add_start;
            swiglu_done <= swiglu_start;
            argmax_done <= argmax_start;
        end
    end

    // Log every dispatched unit start
    always @(posedge clk) begin
        if (embed_start || norm_start || linear_start || rope_start || attn_start || add_start || swiglu_start || argmax_start) begin
            $display("DISPATCH pc=%0d layer=%0d op=%0d src=%0d dst=%0d dim0=%0d dim1=%0d",
                     cur_pc, cur_layer, cur_opcode, cur_src_buf, cur_dst_buf, cur_dim0, cur_dim1);
        end
    end

    reg [15:0] captured_tok;
    initial begin
        clk = 0;
        rst_n = 0;
        start = 0;
        captured_tok = 16'd0;
        #20;
        rst_n = 1;
        #20;

        @(posedge clk); #1;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        while (!done) begin
            if (tok_out_valid) begin
                captured_tok = tok_out;
            end
            @(posedge clk);
        end

        #20;
        $display("FINAL_TOK %0d", captured_tok);
        $finish;
    end
endmodule
"""
        (tmp_path / "tb_sequencer.v").write_text(tb_code)

        sim_exe = tmp_path / "sim.vvp"
        compile_cmd = [
            "iverilog",
            "-g2005",
            "-Irtl/lib",
            "-o",
            str(sim_exe),
            str(tmp_path / "tb_sequencer.v"),
            "rtl/lib/sequencer.v",
        ]
        res = subprocess.run(compile_cmd, check=False, capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

        run_cmd = ["vvp", str(sim_exe)]
        res_run = subprocess.run(run_cmd, check=False, capture_output=True, text=True)
        assert res_run.returncode == 0, f"Simulation failed: {res_run.stderr}"

        dispatched_ops: list[dict[str, Any]] = []
        final_tok = 0
        for line in res_run.stdout.splitlines():
            if line.startswith("DISPATCH"):
                parts = line.split()
                record = {}
                for part in parts[1:]:
                    k, v = part.split("=")
                    record[k] = int(v)
                dispatched_ops.append(record)
            elif line.startswith("FINAL_TOK"):
                final_tok = int(line.split()[1])

        return dispatched_ops, final_tok


def test_sequencer_execution_order_vs_interpreter(tmp_path: Path):
    """Verify that sequencer.v dispatches operations in identical order to compiler/interpreter.py."""
    artifacts = load_model_artifacts_v2(TINY_ARTIFACTS_DIR)
    graph = build_graph_v2_from_artifacts(TINY_ARTIFACTS_DIR)

    with open(CONFIG_FROZEN_PATH, "r", encoding="utf-8") as f:
        frozen_config = json.load(f)

    fmt_res = assign_formats_and_scales(
        graph=graph,
        weights_dict=artifacts["tensors"],
        frozen_config=frozen_config,
    )

    rom_dir = tmp_path / "roms"
    addr_map = pack_rom_artifacts(
        graph=graph,
        format_result=fmt_res,
        weights_dict=artifacts["tensors"],
        out_dir=rom_dir,
        tile_r=8,
        tile_c=16,
        max_ctx=32,
        head_dim=16,
    )

    prog = generate_schedule(
        address_map=addr_map,
        config=artifacts["config"],
        tile_r=8,
        tile_c=16,
    )
    prog.save(rom_dir)
    fmt_res.save_json(rom_dir / "format_table.json")

    # 1. Run Python interpreter and trace ops
    interp = CompiledArtifactInterpreter(rom_dir)
    interp_ops: list[dict[str, Any]] = []
    orig_execute = interp._execute_instruction

    def tracing_execute(ins, layer_idx, pos, token_id=0, layer_bases=None):
        dims = ins["dims"]
        d0 = dims[0] if len(dims) > 0 else 0
        d1 = dims[1] if len(dims) > 1 else 0
        interp_ops.append({
            "layer": layer_idx,
            "op": ins["opcode_val"],
            "src": ins["src_buf"],
            "dst": ins["dst_buf"],
            "dim0": d0,
            "dim1": d1,
        })
        return orig_execute(ins, layer_idx, pos, token_id, layer_bases)

    interp._execute_instruction = tracing_execute
    _, interp_next_tok = interp.decode_step(12, pos=0)

    # 2. Run Verilog sequencer simulation
    hex_lines = [ins.to_hex_word() for ins in prog.instructions]
    sim_ops, captured_tok = run_sequencer_sim(
        hex_lines,
        layer_loop_start=prog.layer_loop_start,
        layer_loop_end=prog.layer_loop_end,
        num_layers=prog.num_layers,
        expected_tok=interp_next_tok,
    )

    # 3. Assert exact sequence match
    assert len(sim_ops) == len(interp_ops), f"Op count mismatch: RTL={len(sim_ops)} vs Py={len(interp_ops)}"
    assert captured_tok == interp_next_tok, f"Captured token mismatch: RTL={captured_tok} vs Py={interp_next_tok}"

    for i, (s_op, py_op) in enumerate(zip(sim_ops, interp_ops, strict=True)):
        assert s_op["layer"] == py_op["layer"], f"Op {i}: layer mismatch ({s_op['layer']} vs {py_op['layer']})"
        assert s_op["op"] == py_op["op"], f"Op {i}: opcode mismatch ({s_op['op']} vs {py_op['op']})"
        assert s_op["src"] == py_op["src"], f"Op {i}: src_buf mismatch ({s_op['src']} vs {py_op['src']})"
        assert s_op["dst"] == py_op["dst"], f"Op {i}: dst_buf mismatch ({s_op['dst']} vs {py_op['dst']})"
        assert s_op["dim0"] == py_op["dim0"], f"Op {i}: dim0 mismatch ({s_op['dim0']} vs {py_op['dim0']})"
        assert s_op["dim1"] == py_op["dim1"], f"Op {i}: dim1 mismatch ({s_op['dim1']} vs {py_op['dim1']})"


def test_sequencer_single_layer_loop():
    """Verify sequencer with custom 1-layer microprogram."""
    # Simple program: EMBED -> RMSNORM (layer loop) -> LINEAR (layer loop) -> ARGMAX -> HALT
    # 0: EMBED (pc=0)
    # 1: RMSNORM (pc=1, loop_start)
    # 2: LINEAR (pc=2, loop_end)
    # 3: ARGMAX (pc=3)
    # 4: HALT (pc=4)
    words = [
        f"{int(Opcode.EMBED):02x}000000000000000000000000400000",
        f"{int(Opcode.RMSNORM):02x}000100000000000000000000400000",
        f"{int(Opcode.LINEAR):02x}010200000000000000000000400040",
        f"{int(Opcode.ARGMAX):02x}090000000000000000000001000000",
        f"{int(Opcode.HALT):02x}000000000000000000000000000000",
    ]
    sim_ops, tok = run_sequencer_sim(words, layer_loop_start=1, layer_loop_end=2, num_layers=3, expected_tok=99)
    # 1 embed + 3 * (2 layer ops) + 1 argmax = 8 ops total
    assert len(sim_ops) == 8
    assert tok == 99

    layers_dispatched = [op["layer"] for op in sim_ops]
    assert layers_dispatched == [0, 0, 0, 1, 1, 2, 2, 0]
