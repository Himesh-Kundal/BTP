"""Unit and integration test suite for the unified LLMHW CLI (T8.1).

Tests all subcommands:
- `llmhw --help`, `llmhw <subcmd> --help`
- `llmhw compile`: Design F, Design P, sliced layers, manifest generation
- `llmhw verify`: bit-exact verification against reference oracle (Claim C1) + static weight audit
- `llmhw synth`: Yosys generic gate synthesis and metrics JSON emission
- `llmhw extract`: mock/dry-run extraction argument handling
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cli import main


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify top-level and subcommand --help output."""
    # Top-level help
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "LLMHW" in captured.out
    assert "extract" in captured.out
    assert "compile" in captured.out
    assert "verify" in captured.out
    assert "synth" in captured.out

    # Subcommands
    for subcmd in ["extract", "compile", "verify", "synth"]:
        with pytest.raises(SystemExit) as excinfo:
            main([subcmd, "--help"])
        assert excinfo.value.code == 0
        captured = capsys.readouterr()
        assert f"usage: llmhw {subcmd}" in captured.out


def test_cli_empty_args(capsys: pytest.CaptureFixture[str]) -> None:
    """Running without args prints help and exits 0."""
    code = main([])
    assert code == 0
    captured = capsys.readouterr()
    assert "LLMHW" in captured.out


def test_cli_compile_custom_weights_design_f(tmp_path: Path) -> None:
    """Test 'llmhw compile' on custom weights array producing Design F."""
    w = np.array([[12, -7, 0, 15], [-3, 8, 22, -1], [5, -12, 3, 0]], dtype=np.int8)
    w_file = tmp_path / "custom_w.npy"
    np.save(w_file, w)

    out_dir = tmp_path / "rtl"
    ret = main([
        "compile",
        "--weights", str(w_file),
        "--arch", "toy_proj",
        "--design", "F",
        "--tile-n", "4",
        "--tile-k", "4",
        "--out", str(out_dir),
    ])
    assert ret == 0

    top_v = out_dir / "llmhw_toy_proj.v"
    manifest_json = out_dir / "llmhw_toy_proj.manifest.json"
    assert top_v.is_file()
    assert manifest_json.is_file()

    content = top_v.read_text(encoding="utf-8")
    assert "module llmhw_toy_proj" in content
    assert "input  wire" in content and "clk" in content
    # Invariant I3: zero weight ports
    assert "w_data" not in content
    assert "w_we" not in content


def test_cli_compile_custom_weights_design_p(tmp_path: Path) -> None:
    """Test 'llmhw compile' on custom weights array producing Design P."""
    w = np.array([[12, -7], [-3, 8]], dtype=np.int8)
    w_file = tmp_path / "custom_w.npy"
    np.save(w_file, w)

    out_dir = tmp_path / "rtl_p"
    ret = main([
        "compile",
        "--weights", str(w_file),
        "--arch", "p_toy",
        "--design", "P",
        "--tile-n", "2",
        "--tile-k", "2",
        "--out", str(out_dir),
    ])
    assert ret == 0

    top_v = out_dir / "llmhw_p_toy_p.v"
    manifest_json = out_dir / "llmhw_p_toy_p.manifest.json"
    assert top_v.is_file()
    assert manifest_json.is_file()

    content = top_v.read_text(encoding="utf-8")
    assert "module llmhw_p_toy_p" in content
    assert "w_data" in content
    assert "w_we" in content
    assert "w_addr" in content


def test_cli_compile_real_layer_slice(tmp_path: Path) -> None:
    """Test 'llmhw compile' on a real SmolLM2 layer slice loaded from artifacts."""
    out_dir = tmp_path / "rtl_real"
    ret = main([
        "compile",
        "--layer", "0",
        "--arch", "q_proj",
        "--slice-n", "8",
        "--slice-k", "16",
        "--design", "F",
        "--out", str(out_dir),
    ])
    assert ret == 0
    top_v = out_dir / "llmhw_l0_q_proj_s8x16.v"
    manifest_json = out_dir / "llmhw_l0_q_proj_s8x16.manifest.json"
    assert top_v.is_file()
    assert manifest_json.is_file()


def test_cli_compile_and_verify_end_to_end_design_f(tmp_path: Path) -> None:
    """Compile custom shape to Design F and verify bit-exact simulation and weight audit."""
    w = np.array([[12, -7, 0, 15], [-3, 8, 22, -1]], dtype=np.int8)
    w_file = tmp_path / "weights.npy"
    np.save(w_file, w)

    rtl_dir = tmp_path / "rtl"
    ret_compile = main([
        "compile",
        "--weights", str(w_file),
        "--arch", "f_e2e",
        "--design", "F",
        "--tile-n", "2",
        "--tile-k", "4",
        "--out", str(rtl_dir),
    ])
    assert ret_compile == 0

    dut_v = rtl_dir / "llmhw_f_e2e.v"
    verify_report = tmp_path / "verify_report.json"

    ret_verify = main([
        "verify",
        "--rtl", str(dut_v),
        "--weights", str(w_file),
        "--sim", "iverilog",
        "--num-vectors", "15",
        "--out", str(verify_report),
    ])
    assert ret_verify == 0
    assert verify_report.is_file()

    data = json.loads(verify_report.read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    assert data["checked"] == 15
    assert data["mismatch"] == 0
    assert data["audit"]["passed"] is True


def test_cli_compile_and_verify_end_to_end_design_p(tmp_path: Path) -> None:
    """Compile custom shape to Design P and verify bit-exact simulation with weight loading."""
    w = np.array([[10, -5], [-20, 30]], dtype=np.int8)
    w_file = tmp_path / "weights.npy"
    np.save(w_file, w)

    rtl_dir = tmp_path / "rtl"
    ret_compile = main([
        "compile",
        "--weights", str(w_file),
        "--arch", "p_e2e",
        "--design", "P",
        "--tile-n", "2",
        "--tile-k", "2",
        "--out", str(rtl_dir),
    ])
    assert ret_compile == 0

    dut_v = rtl_dir / "llmhw_p_e2e_p.v"
    verify_report = tmp_path / "verify_report_p.json"

    ret_verify = main([
        "verify",
        "--rtl", str(dut_v),
        "--weights", str(w_file),
        "--sim", "iverilog",
        "--num-vectors", "10",
        "--out", str(verify_report),
    ])
    assert ret_verify == 0
    assert verify_report.is_file()

    data = json.loads(verify_report.read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    assert data["checked"] == 10
    assert data["mismatch"] == 0


def test_cli_verify_negative_control_corrupted_weights(tmp_path: Path) -> None:
    """Verifying with mismatched weights triggers static audit or simulation failure."""
    w = np.array([[12, -7, 0, 15], [-3, 8, 22, -1]], dtype=np.int8)
    w_file = tmp_path / "weights.npy"
    np.save(w_file, w)

    rtl_dir = tmp_path / "rtl"
    main([
        "compile",
        "--weights", str(w_file),
        "--arch", "neg_test",
        "--design", "F",
        "--tile-n", "2",
        "--tile-k", "4",
        "--out", str(rtl_dir),
    ])
    dut_v = rtl_dir / "llmhw_neg_test.v"

    # Mismatched weights (corrupted)
    w_corrupt = np.array([[99, -7, 0, 15], [-3, 8, 22, -1]], dtype=np.int8)
    corrupt_file = tmp_path / "corrupt_w.npy"
    np.save(corrupt_file, w_corrupt)

    verify_report = tmp_path / "verify_report.json"
    ret = main([
        "verify",
        "--rtl", str(dut_v),
        "--weights", str(corrupt_file),
        "--sim", "iverilog",
        "--num-vectors", "5",
        "--out", str(verify_report),
    ])
    # Must fail
    assert ret != 0


def test_cli_synth(tmp_path: Path) -> None:
    """Test 'llmhw synth' runs Yosys synthesis and saves metrics JSON."""
    w = np.array([[5, -3], [2, 7]], dtype=np.int8)
    w_file = tmp_path / "weights.npy"
    np.save(w_file, w)

    rtl_dir = tmp_path / "rtl"
    main([
        "compile",
        "--weights", str(w_file),
        "--arch", "synth_test",
        "--design", "F",
        "--tile-n", "2",
        "--tile-k", "2",
        "--out", str(rtl_dir),
    ])
    dut_v = rtl_dir / "llmhw_synth_test.v"

    synth_report = tmp_path / "synth_report.json"
    ret = main([
        "synth",
        "--rtl", str(dut_v),
        "--target", "generic",
        "--out", str(synth_report),
    ])
    assert ret == 0
    assert synth_report.is_file()

    data = json.loads(synth_report.read_text(encoding="utf-8"))
    assert data["returncode"] == 0
    assert data["num_cells_total"] > 0
    assert data["ltp_depth"] > 0


def test_cli_extract_mocked(tmp_path: Path) -> None:
    """Test 'llmhw extract' CLI argument handling with mocked loader."""
    mock_model = MagicMock()
    mock_tok = MagicMock()

    with (
        patch("frontend.model_loader.load_model_and_tokenizer", return_value=(mock_model, mock_tok)),
        patch("frontend.linear_extractor.extract_linear_modules", return_value=[]),
    ):
        ret = main([
            "extract",
            "--layer", "0",
            "--proj", "q_proj",
            "--out", str(tmp_path / "artifacts"),
            "--no-capture-calib",
        ])
        assert ret == 0
