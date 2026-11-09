"""Verification tests for Hierarchical Trace Diff Comparator (Task UA.8)."""

from pathlib import Path

from verification.trace_diff import compare_trace_directories


def _create_synthetic_trace_tree(base_dir: Path, corrupt: bool = False, missing: bool = False) -> None:
    """Helper to create hierarchical trace directory tree."""
    trace_dir = base_dir / "seq0" / "tok0" / "L0"
    trace_dir.mkdir(parents=True, exist_ok=True)

    # 1. rmsnorm output
    (trace_dir / "rmsnorm_out.hex").write_text("0a\n1b\n2c\n3d\n")

    # 2. q_proj output
    if corrupt:
        (trace_dir / "q_proj_out.hex").write_text("10\n20\nff\n40\n")  # 3rd elem corrupted
    else:
        (trace_dir / "q_proj_out.hex").write_text("10\n20\n30\n40\n")

    # 3. o_proj output
    if not missing:
        (trace_dir / "o_proj_out.hex").write_text("aa\nbb\ncc\ndd\n")


def test_trace_diff_identical(tmp_path: Path):
    """Verify trace_diff passes when trees are 100% bit-identical."""
    golden = tmp_path / "golden"
    actual = tmp_path / "actual"

    _create_synthetic_trace_tree(golden)
    _create_synthetic_trace_tree(actual)

    res = compare_trace_directories(golden, actual)
    assert res.is_identical is True
    assert res.total_files_checked == 3
    assert res.total_elements_checked == 12
    assert res.first_divergence is None


def test_trace_diff_negative_control_injected_fault(tmp_path: Path):
    """Verify trace_diff catches injected fault and isolates exact element."""
    golden = tmp_path / "golden"
    actual = tmp_path / "actual"

    _create_synthetic_trace_tree(golden)
    _create_synthetic_trace_tree(actual, corrupt=True)

    res = compare_trace_directories(golden, actual)
    assert res.is_identical is False
    assert res.first_divergence is not None

    div = res.first_divergence
    assert div.seq_id == "seq0"
    assert div.token_idx == "tok0"
    assert div.layer == "L0"
    assert div.op_name == "q"
    assert div.element_idx == 2
    assert div.expected_hex == "30"
    assert div.actual_hex == "ff"


def test_trace_diff_negative_control_missing_file(tmp_path: Path):
    """Verify trace_diff catches missing trace file."""
    golden = tmp_path / "golden"
    actual = tmp_path / "actual"

    _create_synthetic_trace_tree(golden)
    _create_synthetic_trace_tree(actual, missing=True)

    res = compare_trace_directories(golden, actual)
    assert res.is_identical is False
    assert len(res.missing_files) == 1
    assert "o_proj_out.hex" in res.missing_files[0]
