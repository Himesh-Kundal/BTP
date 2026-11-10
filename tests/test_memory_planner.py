"""Tests for Memory Planner and Dynamic RAM Allocator (Task UB.6)."""

from pathlib import Path

from compiler.memory_planner import plan_memory
from frontend.importer_v2 import load_model_artifacts_v2

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_model_v2"


def test_memory_planner_tiny_model(tmp_path: Path):
    """Test memory planning for tiny model."""
    artifacts = load_model_artifacts_v2(TINY_ARTIFACTS_DIR)
    config = artifacts["config"]

    plan = plan_memory(config, max_ctx=32, res_bits=16)

    assert plan.max_ctx == 32
    assert len(plan.activation_buffers) == 10
    # KV cache: [2 layers, 2 (K/V), 2 kv_heads, 32 ctx, 16 head_dim] = 4096 bytes
    assert plan.kv_cache_shape == [2, 2, 2, 32, 16]
    assert plan.kv_cache_bytes == 4096

    # Verify report serialization
    report_file = REPO_ROOT / "results" / "part2" / "memory_report_tiny.json"
    plan.save(report_file)
    assert report_file.exists()


def test_memory_planner_full_model():
    """Verify full-model dynamic SRAM memory plan matches 2.81 MB contract."""
    full_config = {
        "hidden_size": 576,
        "intermediate_size": 1536,
        "vocab_size": 49152,
        "num_hidden_layers": 30,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "head_dim": 64,
        "max_position_embeddings": 256,
    }

    plan = plan_memory(full_config, max_ctx=256, res_bits=24)

    assert plan.max_ctx == 256
    # KV cache: 30 * 2 * 3 * 256 * 64 = 2,949,120 bytes
    assert plan.kv_cache_bytes == 2949120
    assert abs(plan.kv_cache_bytes / (1024.0 * 1024.0) - 2.8125) < 1e-3

    report_file = REPO_ROOT / "results" / "part2" / "memory_report_full.json"
    plan.save(report_file)
    assert report_file.exists()
