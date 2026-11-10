"""ROM Generator for Part II Transformer Decoder Accelerator (Task UC.2).

Generates Verilog-2001 synchronous ROM modules and instances for:
- Model Weights (INT8 tile-streamed weights)
- Lookup Tables (SiLU, rsqrt, reciprocal, exp2, RoPE cos/sin)
- Dyadic Requantization Parameters (M, s per layer and projection)
- Microprogram Instructions (128-bit schedule instructions)

Strict Layering (Golden Rule 6):
Only standard library and NumPy/IR are used. No torch or transformers imports.
"""

from __future__ import annotations

import math
from pathlib import Path


def generate_rom_verilog(
    module_name: str,
    data_width: int,
    depth: int,
    init_file: str,
) -> str:
    """Generate a dedicated Verilog ROM module with zero write ports.
    
    Args:
        module_name: Name of the generated Verilog module (e.g. rom_weights).
        data_width: Bitwidth of each data word.
        depth: Total number of words in the ROM.
        init_file: Relative path to the hex file loaded via $readmemh.
        
    Returns:
        Verilog source code string.
    """
    addr_width = max(1, math.ceil(math.log2(depth))) if depth > 1 else 1

    lines = [
        f"// {module_name}.v - Auto-generated ROM module (Immutable Weights)",
        "// Golden Rule 7: Zero write ports, read-only synchronous memory.",
        "`timescale 1ns / 1ps",
        "",
        f"module {module_name} (",
        "    input  wire                  clk,",
        f"    input  wire [{addr_width - 1}:0] addr,",
        f"    output reg  [{data_width - 1}:0] data_out",
        ");",
        "",
        f"    reg [{data_width - 1}:0] mem [0:{depth - 1}];",
        "",
        "    initial begin",
        f'        $readmemh("{init_file}", mem);',
        "    end",
        "",
        "    always @(posedge clk) begin",
        "        data_out <= mem[addr];",
        "    end",
        "",
        "endmodule",
        "",
    ]
    return "\n".join(lines)


def emit_all_rom_modules(
    address_map: dict,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Emit all ROM Verilog files according to the address map.
    
    Args:
        address_map: Address map dictionary loaded from address_map.json.
        output_dir: Target directory to write Verilog files.
        
    Returns:
        Dictionary mapping module name to written file path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated = {}

    rom_configs = [
        ("rom_weights", address_map.get("weights", {})),
        ("rom_requant_params", address_map.get("requant_params", {})),
        ("rom_tables", address_map.get("tables", {})),
    ]

    for mod_name, meta in rom_configs:
        if not meta:
            continue
        data_width = meta.get("word_bits", 8)
        depth = meta.get("total_words", 1)
        hex_file = meta.get("file", f"{mod_name}.hex")

        code = generate_rom_verilog(
            module_name=mod_name,
            data_width=data_width,
            depth=depth,
            init_file=hex_file,
        )
        file_path = out_path / f"{mod_name}.v"
        file_path.write_text(code)
        generated[mod_name] = file_path

    return generated
