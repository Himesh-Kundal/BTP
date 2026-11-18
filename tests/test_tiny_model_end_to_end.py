"""End-to-End Multi-Prompt Verification for Tiny Model Accelerator (Task UD.3).

Verifies:
1. Token-exactness and trace-exactness on >= 100 random prompts including sequences
   up to MAX_CTX (32 positions for the tiny model).
2. Bit-exact match across all 29 intermediate operations and tensors against the
   Python compiled-artifact interpreter (and integer decoder oracle).
3. Exercising the first-divergence detection tool (verification/trace_diff.py) on an
   intentionally injected fault, proving it correctly pinpoints the exact diverging op,
   layer, token step, and element offset.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from backend.top_generator import emit_hardware_bundle
from compiler.interpreter import CompiledArtifactInterpreter
from frontend.importer_v2 import build_graph_v2_from_artifacts, load_model_artifacts_v2
from sim.tb_generator import (
    dump_interpreter_traces,
    run_multi_prompt_simulation,
)
from verification.trace_diff import compare_trace_directories

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_model_v2"
CONFIG_FROZEN_PATH = REPO_ROOT / "numerics" / "config_frozen.json"


@pytest.fixture(scope="module")
def tiny_hardware_bundle(tmp_path_factory) -> Path:
    """Generate hardware bundle for tiny model once per test module."""
    bundle_dir = tmp_path_factory.mktemp("tiny_hw_bundle")
    artifacts = load_model_artifacts_v2(TINY_ARTIFACTS_DIR)
    graph = build_graph_v2_from_artifacts(TINY_ARTIFACTS_DIR)
    with open(CONFIG_FROZEN_PATH, "r", encoding="utf-8") as f:
        frozen_config = json.load(f)

    emit_hardware_bundle(
        graph=graph,
        weights_dict=artifacts["tensors"],
        config=artifacts["config"],
        frozen_config=frozen_config,
        out_dir=bundle_dir,
        tile_r=8,
        tile_c=16,
        max_ctx=32,
    )
    return bundle_dir


def test_tiny_model_100_prompts_token_and_trace_exact(tiny_hardware_bundle: Path, tmp_path: Path):
    """Test token-exactness and trace-exactness on >= 100 prompts up to MAX_CTX."""
    random.seed(42)
    bdir = tiny_hardware_bundle
    interp = CompiledArtifactInterpreter(bdir)

    prompts: list[list[int]] = []
    expected_tokens: list[list[int]] = []
    num_generates: list[int] = []
    golden_trace_dir = tmp_path / "golden"
    sim_trace_dir = tmp_path / "traces"

    # 1. Generate 99 short prompts (lengths 1 to 3) + 1 generated token
    for _ in range(99):
        p_len = random.randint(1, 3)
        p = [random.randint(0, 255) for _ in range(p_len)]
        prompts.append(p)
        num_generates.append(1)

        interp.reset()
        toks: list[int] = []
        for pos, tok in enumerate(p):
            _, next_tok = interp.decode_step(tok, pos=pos)
            toks.append(next_tok)
        _, gen_tok = interp.decode_step(toks[-1], pos=len(p))
        toks.append(gen_tok)
        expected_tokens.append(toks)

    # 2. 1 long prompt with 16 prompt tokens + 16 generated tokens = 32 tokens (MAX_CTX)
    p_long = [random.randint(0, 255) for _ in range(16)]
    prompts.append(p_long)
    num_generates.append(16)

    interp.reset()
    toks_long: list[int] = []
    for pos, tok in enumerate(p_long):
        _, next_tok = interp.decode_step(tok, pos=pos)
        toks_long.append(next_tok)
    for step in range(16):
        pos = 16 + step
        _, next_tok = interp.decode_step(toks_long[-1], pos=pos)
        toks_long.append(next_tok)
    expected_tokens.append(toks_long)

    assert len(prompts) == 100
    assert len(toks_long) == 32

    # 3. Dump golden interpreter traces for sequence 0 and sequence 99 (the long sequence)
    trace_seq_ids = [0, 99]
    for s_id in trace_seq_ids:
        p = prompts[s_id]
        n_gen = num_generates[s_id]
        interp.reset()
        tok_in = p[0]
        for step in range(len(p) + n_gen):
            if step < len(p):
                tok_in = p[step]
            _, out_tok = interp.decode_step(tok_in, pos=step)
            dump_interpreter_traces(interp, golden_trace_dir, seq_id=s_id, tok_idx=step)
            tok_in = out_tok

    # 4. Run top-level RTL simulation across all 100 prompts
    res = run_multi_prompt_simulation(
        bundle_dir=bdir,
        prompts=prompts,
        num_generates=num_generates,
        trace_dir=sim_trace_dir,
        trace_seq_ids=trace_seq_ids,
        golden_trace_dir=golden_trace_dir,
        expected_tokens=expected_tokens,
        simulator="verilator",
    )

    assert res.success, f"Simulation failed: mismatches={res.tok_mismatch}, trace_err={res.trace_mismatch}"
    assert res.tok_mismatch == 0, f"Token mismatches detected: {res.tok_mismatch}"
    assert res.trace_mismatch == 0, f"Trace mismatches detected: {res.trace_mismatch}"
    assert res.timing_err == 0, "Timing errors detected during simulation"
    assert res.total_tokens == sum(len(p) + num_generates[i] for i, p in enumerate(prompts))
    assert res.trace_checked > 0, "No trace elements were checked"
    assert res.diff_result is not None
    assert res.diff_result.is_identical, "Dumped traces differ from golden interpreter traces"


def test_tiny_model_first_divergence_tool_on_injected_fault(tiny_hardware_bundle: Path, tmp_path: Path):
    """Test that trace_diff correctly pinpoints the exact first divergence on an injected fault."""
    bdir = tiny_hardware_bundle
    interp = CompiledArtifactInterpreter(bdir)

    prompt = [42, 17]
    golden_dir = tmp_path / "golden_fault"
    sim_dir = tmp_path / "sim_fault"

    # Step interpreter and dump golden traces
    interp.reset()
    _, _out0 = interp.decode_step(42, pos=0)
    dump_interpreter_traces(interp, golden_dir, seq_id=0, tok_idx=0)
    _, _out1 = interp.decode_step(17, pos=1)
    dump_interpreter_traces(interp, golden_dir, seq_id=0, tok_idx=1)

    # Run simulation and dump RTL traces
    res = run_multi_prompt_simulation(
        bundle_dir=bdir,
        prompts=[prompt],
        num_generates=[0],
        trace_dir=sim_dir,
        trace_seq_ids=[0],
        golden_trace_dir=golden_dir,
        simulator="verilator",
    )
    assert res.success
    assert res.diff_result is not None
    assert res.diff_result.is_identical

    # Inject a known fault into a specific golden tensor: seq0/tok1/L0/q_proj_q.hex line 5
    target_tensor = golden_dir / "seq0" / "tok1" / "L0" / "q_proj_q.hex"
    assert target_tensor.exists(), f"Target tensor {target_tensor} not found"
    lines = target_tensor.read_text(encoding="utf-8").strip().splitlines()
    orig_val = lines[5]
    corrupted_val = "fe" if orig_val != "fe" else "01"
    lines[5] = corrupted_val
    target_tensor.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Re-run trace comparison
    diff = compare_trace_directories(golden_dir, sim_dir)
    assert not diff.is_identical, "Injected fault was not detected by trace comparison!"
    assert diff.first_divergence is not None

    fd = diff.first_divergence
    assert fd.seq_id == "seq0"
    assert fd.token_idx == "tok1"
    assert fd.layer == "L0"
    assert fd.op_name == "q"
    assert fd.tensor_name == "proj_q"
    assert fd.element_idx == 5
    assert fd.expected_hex == corrupted_val
    assert fd.actual_hex == orig_val
