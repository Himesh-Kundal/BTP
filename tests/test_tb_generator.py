"""Unit tests for Testbench Generator & Execution Runner (Task UD.2).

Verifies:
1. Generation and clean compilation of `tb_top_accel.v` for prompt sequences.
2. Production and correct parsing of `TB2_RESULT` line:
   `TB2_RESULT TOKENS=<n> TOK_MISMATCH=<m> TRACE_CHECKED=<k> TRACE_MISMATCH=<j> TIMING_ERR=<t>`
3. Bit-exact trace comparison between RTL dumped traces and Python interpreter golden traces.
4. Rigorous negative controls:
   - Corrupted golden trace causes test failure (TRACE_MISMATCH > 0).
   - Corrupted expected tokens causes test failure (TOK_MISMATCH > 0).
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.top_generator import emit_hardware_bundle
from compiler.interpreter import CompiledArtifactInterpreter
from frontend.importer_v2 import build_graph_v2_from_artifacts, load_model_artifacts_v2
from sim.tb_generator import (
    dump_interpreter_traces,
    generate_top_accel_tb_verilog,
    run_top_accel_simulation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_model_v2"
CONFIG_FROZEN_PATH = REPO_ROOT / "numerics" / "config_frozen.json"


def _setup_tiny_bundle(out_dir: Path) -> dict:
    """Helper to generate hardware bundle for tiny model."""
    artifacts = load_model_artifacts_v2(TINY_ARTIFACTS_DIR)
    graph = build_graph_v2_from_artifacts(TINY_ARTIFACTS_DIR)
    with open(CONFIG_FROZEN_PATH, "r", encoding="utf-8") as f:
        frozen_config = json.load(f)

    return emit_hardware_bundle(
        graph=graph,
        weights_dict=artifacts["tensors"],
        config=artifacts["config"],
        frozen_config=frozen_config,
        out_dir=out_dir,
        tile_r=8,
        tile_c=16,
        max_ctx=32,
    )


def test_tb_generator_emission(tmp_path: Path):
    """Test generating tb_top_accel.v code string."""
    tb_code = generate_top_accel_tb_verilog(
        prompt_tokens=[12, 45],
        num_generate=2,
        seq_id=0,
        trace_dir=tmp_path / "traces",
        expected_tokens=[12, 45, 12, 12],
    )
    assert "module tb_top_accel;" in tb_code
    assert "TB2_RESULT" in tb_code
    assert "TOTAL_STEPS = 4;" in tb_code
    assert "PROMPT_LEN   = 2;" in tb_code


def test_tb_generator_execution_and_tb2_result(tmp_path: Path):
    """Test full execution of tb_top_accel producing TB2_RESULT."""
    bundle_dir = tmp_path / "hw"
    _setup_tiny_bundle(bundle_dir)

    # 1. Run interpreter to get expected token for prompt [12]
    interp = CompiledArtifactInterpreter(bundle_dir)
    _, exp_tok0 = interp.decode_step(12, pos=0)

    # 2. Run simulation
    sim_res = run_top_accel_simulation(
        bundle_dir=bundle_dir,
        prompt_tokens=[12],
        num_generate=0,
        seq_id=0,
        expected_tokens=[exp_tok0],
        simulator="iverilog",
    )

    assert sim_res.success is True
    assert sim_res.total_tokens == 1
    assert sim_res.tokens == [exp_tok0]
    assert sim_res.tok_mismatch == 0
    assert sim_res.timing_err == 0
    assert "TB2_RESULT TOKENS=1 TOK_MISMATCH=0" in sim_res.raw_output


def test_tb_generator_trace_dumping_and_bit_exactness(tmp_path: Path):
    """Test hierarchical trace dumping and bit-for-bit equivalence against interpreter."""
    bundle_dir = tmp_path / "hw"
    _setup_tiny_bundle(bundle_dir)

    # 1. Generate golden traces with Python interpreter
    interp = CompiledArtifactInterpreter(bundle_dir)
    _, exp_tok0 = interp.decode_step(12, pos=0)

    golden_dir = tmp_path / "golden_traces"
    dump_interpreter_traces(interp, golden_dir, seq_id=0, tok_idx=0, num_layers=2)

    # Verify golden traces exist
    assert (golden_dir / "seq0" / "tok0" / "L0" / "embed_res.hex").exists()
    assert (golden_dir / "seq0" / "tok0" / "L0" / "attn_norm_act.hex").exists()
    assert (golden_dir / "seq0" / "tok0" / "L0" / "q_proj_q.hex").exists()
    assert (golden_dir / "seq0" / "tok0" / "L0" / "attn_ctx.hex").exists()
    assert (golden_dir / "seq0" / "tok0" / "L0" / "final_norm_act.hex").exists()
    assert (golden_dir / "seq0" / "tok0" / "L0" / "lm_head_logits.hex").exists()

    # 2. Run RTL simulation with trace dumping enabled
    actual_trace_dir = tmp_path / "actual_traces"
    sim_res = run_top_accel_simulation(
        bundle_dir=bundle_dir,
        prompt_tokens=[12],
        num_generate=0,
        seq_id=0,
        trace_dir=actual_trace_dir,
        golden_trace_dir=golden_dir,
        expected_tokens=[exp_tok0],
        simulator="iverilog",
    )

    assert sim_res.success is True
    assert sim_res.tok_mismatch == 0
    assert sim_res.trace_mismatch == 0
    assert sim_res.diff_result is not None
    assert sim_res.diff_result.is_identical is True
    assert sim_res.diff_result.first_divergence is None


def test_tb_generator_negative_control_corrupt_golden(tmp_path: Path):
    """Negative control: corrupted golden trace MUST cause verification to FAIL."""
    bundle_dir = tmp_path / "hw"
    _setup_tiny_bundle(bundle_dir)

    interp = CompiledArtifactInterpreter(bundle_dir)
    _, exp_tok0 = interp.decode_step(12, pos=0)

    golden_dir = tmp_path / "corrupt_golden"
    dump_interpreter_traces(interp, golden_dir, seq_id=0, tok_idx=0, num_layers=2)

    # Corrupt an element in golden trace
    target_trace = golden_dir / "seq0" / "tok0" / "L0" / "q_proj_q.hex"
    lines = target_trace.read_text().splitlines()
    orig = lines[0]
    lines[0] = "aa" if orig != "aa" else "55"
    target_trace.write_text("\n".join(lines) + "\n")

    actual_trace_dir = tmp_path / "actual_traces"
    sim_res = run_top_accel_simulation(
        bundle_dir=bundle_dir,
        prompt_tokens=[12],
        num_generate=0,
        seq_id=0,
        trace_dir=actual_trace_dir,
        golden_trace_dir=golden_dir,
        expected_tokens=[exp_tok0],
        simulator="iverilog",
    )

    assert sim_res.success is False
    assert sim_res.trace_mismatch > 0
    assert sim_res.diff_result is not None
    assert sim_res.diff_result.is_identical is False
    assert sim_res.diff_result.first_divergence is not None
    assert sim_res.diff_result.first_divergence.element_idx == 0


def test_tb_generator_negative_control_corrupt_expected_token(tmp_path: Path):
    """Negative control: corrupted expected token MUST report TOK_MISMATCH > 0."""
    bundle_dir = tmp_path / "hw"
    _setup_tiny_bundle(bundle_dir)

    interp = CompiledArtifactInterpreter(bundle_dir)
    _, exp_tok0 = interp.decode_step(12, pos=0)

    wrong_token = (exp_tok0 + 1) % 256

    sim_res = run_top_accel_simulation(
        bundle_dir=bundle_dir,
        prompt_tokens=[12],
        num_generate=0,
        seq_id=0,
        expected_tokens=[wrong_token],
        simulator="iverilog",
    )

    assert sim_res.success is False
    assert sim_res.tok_mismatch == 1
    assert "TB2_RESULT TOKENS=1 TOK_MISMATCH=1" in sim_res.raw_output
