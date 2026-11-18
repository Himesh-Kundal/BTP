"""Unit tests for Determinism, Structural, and Lint Audits (Task UD.6).

Verifies:
1. Top-level accelerator contains NO oracle tokens, activations, or expected values (Golden Rule 9).
2. All weights, tables, and requant parameters live in immutable ROMs with NO write ports (Golden Rule 7).
3. Dynamic writable memories are strictly restricted to authorized activation buffers and KV cache.
4. Determinism of code generation and simulation (Golden Rule 11).
5. Clean lint across Icarus Verilog, Verilator, and Yosys.
6. Comprehensive negative controls for each audit.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from backend.top_generator import emit_hardware_bundle, generate_top_accel_verilog
from frontend.importer_v2 import build_graph_v2_from_artifacts, load_model_artifacts_v2
from sim.tb_generator import run_top_accel_simulation
from verification.structural_audit import (
    audit_generator_determinism,
    audit_no_expected_values_in_rtl,
    audit_top_accel_memory_structure,
    run_multi_tool_lint,
)
from verification.weight_audit_v2 import audit_rom_hex_images, audit_rom_module_source


@pytest.fixture
def top_accel_code() -> str:
    """Load canonical top_accel.v source code."""
    path = Path("rtl/lib/top_accel.v")
    assert path.exists(), "rtl/lib/top_accel.v must exist"
    return path.read_text(encoding="utf-8")


def test_top_accel_no_expected_values_clean(top_accel_code: str):
    """Verify canonical top_accel.v passes the no-expected-values audit."""
    report = audit_no_expected_values_in_rtl(top_accel_code, module_name="top_accel")
    assert report["status"] == "PASS"


def test_negative_control_injected_expected_tokens(top_accel_code: str):
    """Verify audit rejects code containing an injected exp_tokens array."""
    tampered = top_accel_code.replace(
        "reg [15:0] pos_reg;",
        "reg [15:0] pos_reg;\n    reg [15:0] exp_tokens [0:31];",
    )
    with pytest.raises(ValueError, match="Forbidden oracle signal or memory pattern detected"):
        audit_no_expected_values_in_rtl(tampered, module_name="tampered_accel")


def test_negative_control_injected_golden_readmem(top_accel_code: str):
    """Verify audit rejects code attempting to read golden/expected trace files."""
    tampered = top_accel_code.replace(
        "module top_accel #(",
        'module top_accel #(\n    // initial $readmemh("golden_trace.hex", mem);\n',
    )
    # Even in comments it should be cleaned, but when active in code:
    active_tampered = top_accel_code + '\ninitial $readmemh("golden_trace.hex", buf_res);\n'
    with pytest.raises(ValueError, match="Forbidden readmem of oracle file in RTL"):
        audit_no_expected_values_in_rtl(active_tampered, module_name="tampered_accel")


def test_top_accel_memory_structure_clean(top_accel_code: str):
    """Verify canonical top_accel.v passes the memory structure and ROM interface audit."""
    report = audit_top_accel_memory_structure(top_accel_code)
    assert report["status"] == "PASS"
    assert "buf_res" in report["found_buffers"]
    assert "buf_logits" in report["found_buffers"]
    assert "rom_weights_inst" in report["instantiated_roms"]


def test_negative_control_injected_unauthorized_writable_memory(top_accel_code: str):
    """Verify audit rejects unauthorized writable RAM arrays (e.g. secret weight storage)."""
    tampered = top_accel_code.replace(
        "reg signed [23:0] buf_res [0:HIDDEN_SIZE-1];",
        "reg signed [23:0] buf_res [0:HIDDEN_SIZE-1];\n    reg signed [7:0] secret_weights [0:1023];",
    )
    with pytest.raises(ValueError, match="Unauthorized writable memory array declared: 'secret_weights'"):
        audit_top_accel_memory_structure(tampered)


def test_negative_control_injected_rom_write_port(top_accel_code: str):
    """Verify audit rejects any ROM instance that connects a write port."""
    tampered = top_accel_code.replace(
        ".data_out(weight_rom_data)",
        ".data_out(weight_rom_data),\n        .we(1'b0)",
    )
    with pytest.raises(ValueError, match="Forbidden write port 'we' connected to ROM instance 'rom_weights_inst'"):
        audit_top_accel_memory_structure(tampered)


def test_generator_determinism():
    """Verify backend/top_generator generates bit-for-bit identical RTL over repeat runs."""
    config = {
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 32,
    }
    rom_files = {
        "weights": "weights.hex",
        "requant_params": "params.hex",
        "program": "prog.hex",
        "layer_bases": "bases.hex",
    }
    report = audit_generator_determinism(config, rom_files)
    assert report["status"] == "PASS"
    assert len(report["sha256"]) == 64


@pytest.fixture(scope="module")
def tiny_hardware_bundle(tmp_path_factory) -> Path:
    """Generate hardware bundle for tiny model once per test module."""
    bundle_dir = tmp_path_factory.mktemp("tiny_hw_audits")
    tiny_artifacts_dir = Path("artifacts/tiny_model_v2")
    artifacts = load_model_artifacts_v2(tiny_artifacts_dir)
    graph = build_graph_v2_from_artifacts(tiny_artifacts_dir)
    with open("numerics/config_frozen.json", "r", encoding="utf-8") as f:
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


def test_simulation_determinism_identical_runs(tiny_hardware_bundle: Path, tmp_path: Path):
    """Verify two independent simulator runs with identical prompts produce identical tokens."""
    prompt = [42, 107, 19]
    num_gen = 2

    trace_dir1 = tmp_path / "run1"
    trace_dir2 = tmp_path / "run2"

    res1 = run_top_accel_simulation(
        bundle_dir=tiny_hardware_bundle,
        prompt_tokens=prompt,
        num_generate=num_gen,
        trace_dir=trace_dir1,
        simulator="verilator",
    )
    res2 = run_top_accel_simulation(
        bundle_dir=tiny_hardware_bundle,
        prompt_tokens=prompt,
        num_generate=num_gen,
        trace_dir=trace_dir2,
        simulator="verilator",
    )

    assert res1.success is True
    assert res2.success is True
    assert res1.tokens == res2.tokens, f"Tokens diverged: {res1.tokens} vs {res2.tokens}"
    assert res1.cycles == res2.cycles, f"Cycles diverged: {res1.cycles} vs {res2.cycles}"

    # Verify byte-identical trace files
    files1 = sorted([p.relative_to(trace_dir1) for p in trace_dir1.rglob("*.hex")])
    files2 = sorted([p.relative_to(trace_dir2) for p in trace_dir2.rglob("*.hex")])
    assert files1 == files2
    assert len(files1) > 0
    for rel_path in files1:
        content1 = (trace_dir1 / rel_path).read_bytes()
        content2 = (trace_dir2 / rel_path).read_bytes()
        assert content1 == content2, f"Trace file diverged: {rel_path}"


def test_multi_tool_lint_cleanliness():
    """Verify Icarus Verilog, Verilator, and Yosys lint and check pass cleanly on top_accel."""
    report = run_multi_tool_lint()
    assert report["status"] == "PASS"
    assert report["iverilog"]["returncode"] == 0
    assert report["verilator"]["returncode"] == 0
    assert report["yosys"]["returncode"] == 0
