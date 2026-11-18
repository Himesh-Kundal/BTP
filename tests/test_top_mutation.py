"""Top-Level Mutation Suite (Task UD.5).

Verifies 100% kill rate across 7 top-level accelerator fault models:
- TM1: Causal masking disabled (attends beyond sequence length)
- TM2: GQA KV-head mapping incorrect (uses modulo instead of division)
- TM3: Position counter off-by-one (pos_reg initialized to 1 instead of 0)
- TM4: RoPE table index shifted (addr_tbl offset by +1)
- TM5: Residual addition scaling / sign inverted (sum = a - b)
- TM6: Argmax tie-break inverted (picks highest index on tie instead of lowest)
- TM7: Weight ROM word swapped (swaps adjacent words in weights_128b.hex)

Every mutant must be detected and killed by token mismatch or intermediate trace mismatch.
Generates results/top_mutation_matrix.json recording 100% kill rate.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.top_generator import emit_hardware_bundle
from compiler.interpreter import CompiledArtifactInterpreter
from frontend.importer_v2 import build_graph_v2_from_artifacts, load_model_artifacts_v2
from sim.tb_generator import dump_interpreter_traces, run_top_accel_simulation

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_model_v2"
CONFIG_FROZEN_PATH = REPO_ROOT / "numerics" / "config_frozen.json"
RESULTS_DIR = REPO_ROOT / "results"


def _setup_fresh_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Emit fresh tiny model hardware bundle."""
    artifacts = load_model_artifacts_v2(TINY_ARTIFACTS_DIR)
    graph = build_graph_v2_from_artifacts(TINY_ARTIFACTS_DIR)
    with open(CONFIG_FROZEN_PATH, "r", encoding="utf-8") as f:
        frozen_config = json.load(f)

    return emit_hardware_bundle(
        graph=graph,
        weights_dict=artifacts["tensors"],
        config=artifacts["config"],
        frozen_config=frozen_config,
        out_dir=bundle_dir,
        tile_r=8,
        tile_c=16,
        max_ctx=32,
    )


def test_top_level_mutation_suite_100_percent_kill(tmp_path: Path):
    """Run top-level mutation suite and assert 100% kill rate."""
    bdir = tmp_path / "bundle"
    _setup_fresh_bundle(bdir)
    interp = CompiledArtifactInterpreter(bdir)

    prompt = [0, 1]
    num_gen = 1
    golden_dir = tmp_path / "golden"
    sim_dir = tmp_path / "traces"

    # 1. Establish golden expectations and traces
    interp.reset()
    tok_in = prompt[0]
    expected_tokens: list[int] = []
    for step in range(len(prompt) + num_gen):
        if step < len(prompt):
            tok_in = prompt[step]
        _, out_tok = interp.decode_step(tok_in, pos=step)
        dump_interpreter_traces(interp, golden_dir, seq_id=0, tok_idx=step)
        expected_tokens.append(out_tok)
        tok_in = out_tok

    # Baseline verification
    res_base = run_top_accel_simulation(
        bundle_dir=bdir,
        prompt_tokens=prompt,
        num_generate=num_gen,
        trace_dir=sim_dir,
        golden_trace_dir=golden_dir,
        expected_tokens=expected_tokens,
        simulator="verilator",
    )
    assert res_base.success, f"Baseline simulation failed: {res_base.raw_output}"

    # 2. Define top-level mutant campaign
    mutants = [
        {
            "id": "TM1",
            "name": "causal_off",
            "description": "Causal mask off: Softmax normalizes past current position (pos+2)",
            "file": "rtl/lib/attention.v",
            "target": ".seq_len(pos + 1'b1),",
            "replacement": ".seq_len(pos + 2'd2),",
            "type": "code",
        },
        {
            "id": "TM2",
            "name": "kv_head_map",
            "description": "GQA KV-head mapping uses modulo instead of floor division",
            "file": "rtl/lib/attention.v",
            "target": "wire [31:0] div_val = cur_q_head / GQA_GROUP;",
            "replacement": "wire [31:0] div_val = cur_q_head % GQA_GROUP;",
            "type": "code",
        },
        {
            "id": "TM3",
            "name": "pos_off_by_one",
            "description": "Sequencer / datapath position counter initialized to 1 instead of 0",
            "file": str(bdir / "top_accel.v"),
            "target": "pos_reg <= 16'd0;",
            "replacement": "pos_reg <= 16'd1;",
            "type": "bundle_code",
        },
        {
            "id": "TM4",
            "name": "rope_table_shift",
            "description": "RoPE table index shifted by +1 entry",
            "file": "rtl/lib/rope.v",
            "target": "wire [15:0] addr_tbl  = pos * half_dim + i_idx;",
            "replacement": "wire [15:0] addr_tbl  = pos * half_dim + i_idx + 16'd1;",
            "type": "code",
        },
        {
            "id": "TM5",
            "name": "residual_scale",
            "description": "Residual addition inverted: subtracts residual instead of adding",
            "file": "rtl/lib/residual_add.v",
            "target": "wire signed [WIDTH:0] sum = {{1{a_data[WIDTH-1]}}, a_data} + {{1{b_data[WIDTH-1]}}, b_data};",
            "replacement": "wire signed [WIDTH:0] sum = {{1{a_data[WIDTH-1]}}, a_data} - {{1{b_data[WIDTH-1]}}, b_data};",
            "type": "code",
        },
        {
            "id": "TM6",
            "name": "tie_break",
            "description": "Argmax tie-break inverted: selects highest index instead of lowest",
            "file": "rtl/lib/argmax.v",
            "target": "if (stg1_id == {TOK_WIDTH{1'b0}} || stg1_scaled > best_val) begin",
            "replacement": "if (stg1_id == {TOK_WIDTH{1'b0}} || stg1_scaled >= best_val) begin",
            "type": "tie_break_special",
        },
        {
            "id": "TM7",
            "name": "rom_word_swap",
            "description": "Weight ROM word swap: swap words 0 and 1 in weights_128b.hex",
            "file": str(bdir / "weights_128b.hex"),
            "type": "rom_swap",
        },
    ]

    results: list[dict[str, Any]] = []

    for m in mutants:
        m_id = m["id"]
        m_type = m["type"]
        killed = False
        kill_reason = ""

        if m_type in ("code", "bundle_code"):
            p = Path(m["file"]) if Path(m["file"]).is_absolute() else (REPO_ROOT / m["file"])
            orig_text = p.read_text(encoding="utf-8")
            assert m["target"] in orig_text, f"Target pattern not found for {m_id}"
            try:
                p.write_text(orig_text.replace(m["target"], m["replacement"], 1), encoding="utf-8")
                res = run_top_accel_simulation(
                    bundle_dir=bdir,
                    prompt_tokens=prompt,
                    num_generate=num_gen,
                    trace_dir=sim_dir,
                    golden_trace_dir=golden_dir,
                    expected_tokens=expected_tokens,
                    simulator="verilator",
                )
                if not res.success or res.tok_mismatch > 0 or res.trace_mismatch > 0:
                    killed = True
                    kill_reason = f"tok_mismatch={res.tok_mismatch}, trace_mismatch={res.trace_mismatch}"
            finally:
                p.write_text(orig_text, encoding="utf-8")

        elif m_type == "tie_break_special":
            # For tie-break: create tied lm_head weights (all zeros) so all logits tie at 0
            with open(bdir / "address_map.json") as f:
                am = json.load(f)
            lm_info = am["weights"]["lm_head"]
            b_off = lm_info["base_addr"] // 16
            l_len = lm_info["length"] // 16

            w_path = bdir / "weights_128b.hex"
            orig_w = w_path.read_text(encoding="utf-8").splitlines()
            tied_w = list(orig_w)
            for i in range(b_off, b_off + l_len):
                tied_w[i] = "00" * 16
            w_path.write_text("\n".join(tied_w) + "\n", encoding="utf-8")

            p_argmax = REPO_ROOT / m["file"]
            orig_argmax = p_argmax.read_text(encoding="utf-8")
            assert m["target"] in orig_argmax
            try:
                p_argmax.write_text(orig_argmax.replace(m["target"], m["replacement"], 1), encoding="utf-8")
                res = run_top_accel_simulation(
                    bundle_dir=bdir,
                    prompt_tokens=[0],
                    num_generate=1,
                    expected_tokens=[0, 0],
                    simulator="verilator",
                )
                # Golden selects token 0; mutated >= selects token 255
                if not res.success or res.tok_mismatch > 0 or res.tokens != [0, 0]:
                    killed = True
                    kill_reason = f"tok_mismatch={res.tok_mismatch}, tokens={res.tokens} (expected [0, 0])"
            finally:
                p_argmax.write_text(orig_argmax, encoding="utf-8")
                w_path.write_text("\n".join(orig_w) + "\n", encoding="utf-8")

        elif m_type == "rom_swap":
            w_path = Path(m["file"])
            orig_w = w_path.read_text(encoding="utf-8").splitlines()
            swapped_w = list(orig_w)
            swapped_w[0], swapped_w[1] = swapped_w[1], swapped_w[0]
            w_path.write_text("\n".join(swapped_w) + "\n", encoding="utf-8")
            try:
                res = run_top_accel_simulation(
                    bundle_dir=bdir,
                    prompt_tokens=prompt,
                    num_generate=num_gen,
                    trace_dir=sim_dir,
                    golden_trace_dir=golden_dir,
                    expected_tokens=expected_tokens,
                    simulator="verilator",
                )
                if not res.success or res.tok_mismatch > 0 or res.trace_mismatch > 0:
                    killed = True
                    kill_reason = f"tok_mismatch={res.tok_mismatch}, trace_mismatch={res.trace_mismatch}"
            finally:
                w_path.write_text("\n".join(orig_w) + "\n", encoding="utf-8")

        status = "KILLED" if killed else "SURVIVED"
        results.append({
            "id": m["id"],
            "name": m["name"],
            "description": m["description"],
            "status": status,
            "kill_reason": kill_reason,
        })

    # 3. Compile matrix and assert 100% kill rate
    total_mutants = len(results)
    killed_mutants = sum(1 for r in results if r["status"] == "KILLED")
    survived_mutants = total_mutants - killed_mutants
    kill_rate = (killed_mutants / total_mutants) * 100.0

    matrix = {
        "suite": "top_accel_mutation_suite_v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_mutants": total_mutants,
        "killed_mutants": killed_mutants,
        "survived_mutants": survived_mutants,
        "kill_rate_pct": kill_rate,
        "results": results,
    }

    # Save to results/top_mutation_matrix.json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "top_mutation_matrix.json", "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)

    assert survived_mutants == 0, f"{survived_mutants} mutants survived! Matrix: {results}"
    assert kill_rate == 100.0, f"Expected 100% kill rate, got {kill_rate}%"
