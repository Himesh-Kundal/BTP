"""Mutation testing engine v2 for model decoder RTL units (M10-M30 per docs/part2/06 §A6).

Fault models:
- M10: Requant rounds half-down instead of half-up (requant.v)
- M11: Asymmetric saturation to -128 instead of symmetric -127 (requant.v)
- M12: RoPE pairs adjacent elements instead of split halves (rope.v)
- M13: RoPE table index off by one (rope.v)
- M14: Attention K vector corrupted / inverted during KV store (attention.v)
- M15: GQA head mapping uses modulo instead of floor divide (kv_addr_gen.v)
- M16: Causal mask off, attends to future position pos+1 (attention.v)
- M17: Softmax without max subtraction (softmax.v)
- M18: Softmax exp2 LUT index off by one (softmax.v)
- M19: Rsqrt LUT index off by one (rmsnorm.v)
- M20: RMSNorm skips gamma scaling (rmsnorm.v)
- M21: SiLU LUT sign inversion (swiglu.v)
- M22: Residual addition skipped (residual_add.v)
- M23: Argmax ties to highest index instead of lowest (argmax.v)
- M24: Argmax compares raw unscaled accumulators (argmax.v)
- M25: ROM word base offset by +1 (sequencer.v)
- M26: KV write at pos+1 (kv_cache.v)
- M27: Embedding row index off by one (embed_gather.v)
- M28: Linear engine accumulator truncated to 16 bits (linear_engine.v)
- M29: Linear engine padding lane nonzero (linear_engine.v)
- M30: Sequencer premature done assertion (sequencer.v)
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class MutantDefinition:
    id: str
    description: str
    target_rel_path: str
    target_pattern: str
    replacement_pattern: str
    test_args: list[str]


MUTANTS_V2: list[MutantDefinition] = [
    MutantDefinition(
        id="M10",
        description="Requant rounds half-down instead of half-up",
        target_rel_path="rtl/lib/requant.v",
        target_pattern="({{ (PROD_WIDTH-1){1'b0} }, 1'b1} <<< (shift_in - 1'b1))",
        replacement_pattern="{PROD_WIDTH{1'b0}}",
        test_args=["pytest", "tests/test_rtl_numerics.py", "-k", "test_rtl_requant_module"],
    ),
    MutantDefinition(
        id="M11",
        description="Asymmetric saturation to -128 instead of symmetric -127",
        target_rel_path="rtl/lib/numerics.vh",
        target_pattern="sat_int8 = -8'sd127;",
        replacement_pattern="sat_int8 = -8'sd128;",
        test_args=["pytest", "tests/test_rtl_numerics.py", "-k", "test_rtl_rshift_rnd_and_sat_exhaustive_small_widths"],
    ),
    MutantDefinition(
        id="M12",
        description="RoPE pairs adjacent elements instead of split halves",
        target_rel_path="rtl/lib/rope.v",
        target_pattern="wire [15:0] addr_x2   = head_base + half_dim + i_idx;",
        replacement_pattern="wire [15:0] addr_x2   = head_base + (i_idx << 1) + 16'd1;",
        test_args=["pytest", "tests/test_rtl_rope.py"],
    ),
    MutantDefinition(
        id="M13",
        description="RoPE table index off by one",
        target_rel_path="rtl/lib/rope.v",
        target_pattern="wire [15:0] addr_tbl  = pos * half_dim + i_idx;",
        replacement_pattern="wire [15:0] addr_tbl  = pos * half_dim + i_idx + 16'd1;",
        test_args=["pytest", "tests/test_rtl_rope.py"],
    ),
    MutantDefinition(
        id="M14",
        description="Attention K vector corrupted / inverted during KV store",
        target_rel_path="rtl/lib/attention.v",
        target_pattern="kv_wr_data    <= k_rd_data;",
        replacement_pattern="kv_wr_data    <= -k_rd_data;",
        test_args=["pytest", "tests/test_rtl_attention.py", "-k", "test_attention_multi_position_causal"],
    ),
    MutantDefinition(
        id="M15",
        description="GQA head mapping uses modulo instead of floor divide",
        target_rel_path="rtl/lib/kv_addr_gen.v",
        target_pattern="wire [31:0] div_val = q_head_in / GQA_GROUP;",
        replacement_pattern="wire [31:0] div_val = q_head_in % GQA_GROUP;",
        test_args=["pytest", "tests/test_rtl_kv_cache.py", "-k", "test_gqa_mapping_and_coord_calc"],
    ),
    MutantDefinition(
        id="M16",
        description="Causal mask off, attends to future position pos+1",
        target_rel_path="rtl/lib/attention.v",
        target_pattern=".seq_len(pos + 1'b1),",
        replacement_pattern=".seq_len(pos + 2'd2),",
        test_args=["pytest", "tests/test_rtl_attention.py", "-k", "test_attention_multi_position_causal"],
    ),
    MutantDefinition(
        id="M17",
        description="Softmax without max subtraction",
        target_rel_path="rtl/lib/softmax.v",
        target_pattern="cur_int_part <= (max_score - score_data) >> 8;",
        replacement_pattern="cur_int_part <= (-score_data) >> 8;",
        test_args=["pytest", "tests/test_rtl_softmax.py"],
    ),
    MutantDefinition(
        id="M18",
        description="Softmax exp2 LUT index off by one",
        target_rel_path="rtl/lib/softmax.v",
        target_pattern="exp2_lut_addr <= (max_score - score_data);",
        replacement_pattern="exp2_lut_addr <= (max_score - score_data) + 8'd1;",
        test_args=["pytest", "tests/test_rtl_softmax.py"],
    ),
    MutantDefinition(
        id="M19",
        description="Rsqrt LUT index off by one (unshifted index)",
        target_rel_path="rtl/lib/rmsnorm.v",
        target_pattern="wire [6:0]                  calc_sh_amt  = (calc_sh > 7'd8) ? (calc_sh - 7'd8) : 7'd0;",
        replacement_pattern="wire [6:0]                  calc_sh_amt  = 7'd0;",
        test_args=["pytest", "tests/test_rtl_rmsnorm.py"],
    ),
    MutantDefinition(
        id="M20",
        description="RMSNorm skips gamma scaling",
        target_rel_path="rtl/lib/rmsnorm.v",
        target_pattern="wire signed [63:0]          stg2_acc     = stg1_scaled * stg1_gamma;",
        replacement_pattern="wire signed [63:0]          stg2_acc     = stg1_scaled * 64'sd1;",
        test_args=["pytest", "tests/test_rtl_rmsnorm.py"],
    ),
    MutantDefinition(
        id="M21",
        description="SiLU LUT sign inversion",
        target_rel_path="rtl/lib/swiglu.v",
        target_pattern="wire signed [63:0] prod        = {{56{silu_lut_data[LUT_WIDTH-1]}}, silu_lut_data} *",
        replacement_pattern="wire signed [63:0] prod        = -({{56{silu_lut_data[LUT_WIDTH-1]}}, silu_lut_data}) *",
        test_args=["pytest", "tests/test_rtl_swiglu.py"],
    ),
    MutantDefinition(
        id="M22",
        description="Residual addition skipped",
        target_rel_path="rtl/lib/residual_add.v",
        target_pattern="wire signed [WIDTH:0] sum = {{1{a_data[WIDTH-1]}}, a_data} + {{1{b_data[WIDTH-1]}}, b_data};",
        replacement_pattern="wire signed [WIDTH:0] sum = {{1{a_data[WIDTH-1]}}, a_data};",
        test_args=["pytest", "tests/test_rtl_residual_embed.py", "-k", "test_residual_add_with_saturation"],
    ),
    MutantDefinition(
        id="M23",
        description="Argmax ties to highest index instead of lowest",
        target_rel_path="rtl/lib/argmax.v",
        target_pattern="if (stg1_id == {TOK_WIDTH{1'b0}} || stg1_scaled > best_val) begin",
        replacement_pattern="if (stg1_id == {TOK_WIDTH{1'b0}} || stg1_scaled >= best_val) begin",
        test_args=["pytest", "tests/test_rtl_argmax.py", "-k", "test_argmax_tie_breaking"],
    ),
    MutantDefinition(
        id="M24",
        description="Argmax compares raw unscaled accumulators",
        target_rel_path="rtl/lib/argmax.v",
        target_pattern="wire signed [63:0] scaled = rshift_rnd64(prod, s_data);",
        replacement_pattern="wire signed [63:0] scaled = {{32{logit_data[ACC_WIDTH-1]}}, logit_data};",
        test_args=["pytest", "tests/test_rtl_argmax.py", "-k", "test_argmax_scale_inversion"],
    ),
    MutantDefinition(
        id="M25",
        description="ROM word base offset by +1",
        target_rel_path="rtl/lib/sequencer.v",
        target_pattern="cur_weight_base <= prog_rom_data[95:64];",
        replacement_pattern="cur_weight_base <= prog_rom_data[95:64] + 32'd1;",
        test_args=["pytest", "tests/test_rtl_sequencer.py", "-k", "test_sequencer_execution_order_vs_interpreter"],
    ),
    MutantDefinition(
        id="M26",
        description="KV write at pos+1 (causal store corruption)",
        target_rel_path="rtl/lib/kv_cache.v",
        target_pattern="wire [ADDR_W-1:0] wr_calc_addr = calc_addr(wr_layer, wr_is_v, wr_kv_head, wr_pos, wr_dim);",
        replacement_pattern="wire [ADDR_W-1:0] wr_calc_addr = calc_addr(wr_layer, wr_is_v, wr_kv_head, wr_pos + 1'b1, wr_dim);",
        test_args=["pytest", "tests/test_rtl_kv_cache.py", "-k", "test_kv_cache_read_write_ordering"],
    ),
    MutantDefinition(
        id="M27",
        description="Embedding row index off by one",
        target_rel_path="rtl/lib/embed_gather.v",
        target_pattern="token_base <= {16'd0, token_id} * {16'd0, dim};",
        replacement_pattern="token_base <= {16'd0, (token_id + 16'd1)} * {16'd0, dim};",
        test_args=["pytest", "tests/test_rtl_residual_embed.py", "-k", "test_embed_gather"],
    ),
    MutantDefinition(
        id="M28",
        description="Linear engine accumulator truncated to 16 bits",
        target_rel_path="rtl/lib/linear_engine.v",
        target_pattern="row_acc[comp_r_pipe] <= row_acc[comp_r_pipe] + dot_sum;",
        replacement_pattern="row_acc[comp_r_pipe] <= {{16{dot_sum[15]}}, dot_sum[15:0]};",
        test_args=["pytest", "tests/test_rtl_linear_engine.py", "-k", "test_linear_engine_multi_tile_exact"],
    ),
    MutantDefinition(
        id="M29",
        description="Linear engine padding lane nonzero",
        target_rel_path="rtl/lib/linear_engine.v",
        target_pattern="assign prod[gi] = eff_x * w_lane;",
        replacement_pattern="assign prod[gi] = eff_x * w_lane + (is_valid_k ? 16'sd0 : 16'sd1);",
        test_args=["pytest", "tests/test_rtl_linear_engine.py", "-k", "test_linear_engine_non_multiple_shape_with_padding"],
    ),
    MutantDefinition(
        id="M30",
        description="Sequencer premature done assertion",
        target_rel_path="rtl/lib/sequencer.v",
        target_pattern="OP_ARGMAX:    argmax_start <= 1'b1;",
        replacement_pattern="OP_ARGMAX: begin argmax_start <= 1'b1; done <= 1'b1; end",
        test_args=["pytest", "tests/test_rtl_sequencer.py", "-k", "test_sequencer_execution_order_vs_interpreter"],
    ),
]


def run_mutation_campaign(
    mutants: list[MutantDefinition] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute mutation testing campaign across all defined mutant models."""
    if mutants is None:
        mutants = MUTANTS_V2

    results: list[dict[str, Any]] = []
    total = len(mutants)
    killed = 0

    uv_cmd = str(Path.home() / ".local/bin/uv")

    for m in mutants:
        target_file = REPO_ROOT / m.target_rel_path
        assert target_file.exists(), f"Target file does not exist: {target_file}"
        original_content = target_file.read_text(encoding="utf-8")

        assert m.target_pattern in original_content, (
            f"Pattern not found in {m.target_rel_path} for {m.id}: {m.target_pattern}"
        )

        mutated_content = original_content.replace(m.target_pattern, m.replacement_pattern, 1)

        try:
            # Inject mutant
            target_file.write_text(mutated_content, encoding="utf-8")

            # Run test using uv run
            cmd = [uv_cmd, "run"] + m.test_args
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )

            status = "KILLED" if proc.returncode != 0 else "SURVIVED"
            if status == "KILLED":
                killed += 1

            results.append({
                "id": m.id,
                "description": m.description,
                "target_file": m.target_rel_path,
                "status": status,
                "exit_code": proc.returncode,
                "command": " ".join(cmd),
            })
        finally:
            # Restore original file
            target_file.write_text(original_content, encoding="utf-8")

    kill_rate_pct = (killed / total) * 100.0 if total > 0 else 0.0
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_mutants": total,
        "killed_mutants": killed,
        "survived_mutants": total - killed,
        "kill_rate_pct": kill_rate_pct,
        "results": results,
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    out_file = REPO_ROOT / "results" / "mutation_matrix_v2.json"
    res = run_mutation_campaign(output_path=out_file)
    print(f"Mutation Suite v2: {res['killed_mutants']}/{res['total_mutants']} killed ({res['kill_rate_pct']:.1f}%)")
