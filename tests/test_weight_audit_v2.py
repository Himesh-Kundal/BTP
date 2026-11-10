"""Unit tests for ROM Generator and Weight Audit v2 (Task UC.2).

Tests:
1. Valid ROM modules pass audit.
2. Injected write port fails audit (negative control).
3. Injected runtime write assignment fails audit (negative control).
4. Packed ROM images from tiny_model_v2 pass hex integrity audit.
5. Tampered ROM hex image fails audit (negative control).
"""

import json
import tempfile
from pathlib import Path

import pytest

from backend.rom_generator import generate_rom_verilog
from verification.weight_audit_v2 import audit_rom_hex_images, audit_rom_module_source


def test_rom_sync_template_audit():
    """Verify standard rtl/lib/rom_sync.v passes the immutable ROM audit."""
    rom_v = Path("rtl/lib/rom_sync.v").read_text()
    report = audit_rom_module_source(rom_v, module_name="rom_sync")
    assert report["status"] == "PASS"
    assert "clk" in report["input_ports"]
    assert "addr" in report["input_ports"]
    assert len(report["output_ports"]) == 1


def test_generated_rom_modules_audit():
    """Verify backend/rom_generator produces clean ROM modules that pass audit."""
    code = generate_rom_verilog(
        module_name="rom_weights_test",
        data_width=8,
        depth=1024,
        init_file="weights.hex",
    )
    report = audit_rom_module_source(code, module_name="rom_weights_test")
    assert report["status"] == "PASS"


def test_negative_control_injected_write_port():
    """Verify audit rejects a ROM module containing a write enable port."""
    bad_code = """
module rom_bad (
    input wire clk,
    input wire [9:0] addr,
    input wire we,
    output reg [7:0] data_out
);
    reg [7:0] mem [0:1023];
    always @(posedge clk) begin
        data_out <= mem[addr];
    end
endmodule
"""
    with pytest.raises(ValueError, match="Forbidden write-related input port found: 'we'"):
        audit_rom_module_source(bad_code, module_name="rom_bad")


def test_negative_control_injected_memory_write():
    """Verify audit rejects a ROM module with a runtime write to its memory array."""
    bad_code = """
module rom_bad_write (
    input wire clk,
    input wire [9:0] addr,
    output reg [7:0] data_out
);
    reg [7:0] mem [0:1023];
    always @(posedge clk) begin
        mem[addr] <= 8'h00;
        data_out <= mem[addr];
    end
endmodule
"""
    with pytest.raises(ValueError, match="Runtime write assignment to memory array"):
        audit_rom_module_source(bad_code, module_name="rom_bad_write")


def test_rom_hex_images_audit_tiny_model(tmp_path: Path):
    """Verify packed hex files from tiny model pass hex integrity audit."""
    tiny_dir = Path("artifacts/tiny_model_v2")
    if not (tiny_dir / "manifest.json").exists():
        pytest.skip("artifacts/tiny_model_v2/manifest.json not found")

    from compiler.format_assignment import assign_formats_and_scales
    from compiler.rom_packing import pack_rom_artifacts
    from frontend.importer_v2 import build_graph_v2_from_artifacts, load_model_artifacts_v2

    artifacts = load_model_artifacts_v2(tiny_dir)
    graph = build_graph_v2_from_artifacts(tiny_dir)
    frozen_cfg = json.loads(Path("numerics/config_frozen.json").read_text())

    fmt_res = assign_formats_and_scales(
        graph=graph,
        weights_dict=artifacts["tensors"],
        frozen_config=frozen_cfg,
    )

    out_dir = tmp_path / "roms"
    pack_rom_artifacts(
        graph=graph,
        format_result=fmt_res,
        weights_dict=artifacts["tensors"],
        out_dir=out_dir,
        tile_r=8,
        tile_c=16,
        max_ctx=32,
        head_dim=16,
    )

    map_path = out_dir / "address_map.json"
    report = audit_rom_hex_images(map_path)
    assert report["status"] == "PASS"
    assert len(report["checked_roms"]) == 3


def test_negative_control_tampered_hex_file():
    """Verify audit rejects a hex file whose contents have been modified."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        hex_file = tmp_p / "weights.hex"
        hex_file.write_text("01\n02\n03\n")

        # Record real hash
        import hashlib
        real_sha = hashlib.sha256(hex_file.read_bytes()).hexdigest()

        map_data = {
            "weights": {
                "file": "weights.hex",
                "sha256": real_sha,
                "total_words": 3,
                "word_bits": 8,
            }
        }
        map_file = tmp_p / "address_map.json"
        map_file.write_text(json.dumps(map_data))

        # First verify it passes
        assert audit_rom_hex_images(map_file)["status"] == "PASS"

        # Now tamper with one byte
        hex_file.write_text("01\nFF\n03\n")

        with pytest.raises(ValueError, match="ROM integrity violation"):
            audit_rom_hex_images(map_file)
