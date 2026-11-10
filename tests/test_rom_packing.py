"""Tests for ROM Packing Pass and Address Map (Task UB.4)."""

import json
from pathlib import Path

from compiler.format_assignment import assign_formats_and_scales
from compiler.rom_packing import compute_bytes_sha256, pack_rom_artifacts
from frontend.importer_v2 import build_graph_v2_from_artifacts, load_model_artifacts_v2

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_model_v2"
CONFIG_FROZEN_PATH = REPO_ROOT / "numerics" / "config_frozen.json"


def test_rom_packing_tiny_model(tmp_path: Path):
    """Test ROM packing of weights, requant parameters, and lookup tables for tiny model."""
    artifacts = load_model_artifacts_v2(TINY_ARTIFACTS_DIR)
    graph = build_graph_v2_from_artifacts(TINY_ARTIFACTS_DIR)

    with open(CONFIG_FROZEN_PATH, "r", encoding="utf-8") as f:
        frozen_config = json.load(f)

    fmt_res = assign_formats_and_scales(
        graph=graph,
        weights_dict=artifacts["tensors"],
        frozen_config=frozen_config,
    )

    out_dir = tmp_path / "roms"
    tile_r, tile_c = 8, 16

    addr_map = pack_rom_artifacts(
        graph=graph,
        format_result=fmt_res,
        weights_dict=artifacts["tensors"],
        out_dir=out_dir,
        tile_r=tile_r,
        tile_c=tile_c,
        max_ctx=32,
        head_dim=16,
    )

    # 1. Output files exist
    assert (out_dir / "weights.hex").exists()
    assert (out_dir / "requant_params.hex").exists()
    assert (out_dir / "tables.hex").exists()
    assert (out_dir / "address_map.json").exists()

    # 2. Check SHA256 integrity
    for img_key in ("weights_image", "requant_image", "tables_image"):
        img_info = addr_map[img_key]
        file_path = out_dir / img_info["file"]
        actual_sha = compute_bytes_sha256(file_path.read_bytes())
        assert actual_sha == img_info["sha256"]

    # 3. Check weights contiguous and padded to (tile_r, tile_c)
    last_end = 0
    for node_info in addr_map["weights"].values():
        assert node_info["base_addr"] == last_end
        assert node_info["padded_shape"][0] % tile_r == 0
        assert node_info["padded_shape"][1] % tile_c == 0
        assert node_info["length"] == node_info["padded_shape"][0] * node_info["padded_shape"][1]
        last_end += node_info["length"]

    assert last_end == addr_map["weights_image"]["total_words"]

    # 4. Check tables entries
    assert "rope_cos" in addr_map["tables"]
    assert "rope_sin" in addr_map["tables"]
    assert "rsqrt_mantissa_lut" in addr_map["tables"]
    assert "exp2_lut" in addr_map["tables"]
    assert "reciprocal_lut" in addr_map["tables"]
    assert "silu_lut" in addr_map["tables"]
