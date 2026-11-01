"""Programmable baseline generator (Design P and Design P_bus) (T5.1).

Implements:
1. `generate_tile_core_p`: generic tile core for arbitrary (Nt, Kt) shapes
   using runtime signed multipliers (x_k * w_nk) with explicit sign extension.
2. `generate_full_module_p`: complete module containing W_reg register array,
   write ports (w_we, w_addr, w_data), tile cores, reduction adder trees over j,
   constant bias, and registered wrapper with latency L=1.
3. `generate_full_module_p_bus`: logic-only variant taking weights as a
   primary-input bus w_flat without W_reg storage.
4. Conformance to Rule N10: identical widths to Design F (x: 8, w: 8, prod: 16, acc: 32).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from backend.verilog_codegen import sanitize_identifier

if TYPE_CHECKING:
    from backend.tiling import TileManifest
    from ir.linear import LinearOp


def generate_tile_core_p(
    module_name: str,
    nt: int,
    kt: int,
    acc_bits: int = 32,
    manifest_hash: str = "",
) -> str:
    """Generate synthesizable, combinational Verilog-2001 tile core for Design P.

    Interface:
        module <module_name> (
            input  wire [8*kt-1:0]         x_flat,
            input  wire [8*nt*kt-1:0]      w_flat,
            output wire [acc_bits*nt-1:0]  y_flat
        );

    Args:
        module_name: Module identifier.
        nt: Number of output rows in tile.
        kt: Number of input columns in tile.
        acc_bits: Accumulator bit width (default 32).
        manifest_hash: Manifest hash for header comment.

    Returns:
        Complete Verilog-2001 source code string.
    """
    mod_name = sanitize_identifier(module_name)
    in_bits = 8 * kt
    weight_bits = 8 * nt * kt
    out_bits = acc_bits * nt

    lines: list[str] = [
        "// =============================================================================",
        f"// Module: {mod_name} (Design P Programmable Tile Core)",
        f"// Dimensions: Nt={nt}, Kt={kt}, AccBits={acc_bits}",
        f"// Manifest Hash: {manifest_hash or 'N/A'}",
        "// Standards: Verilog-2001, Explicit Signed Arithmetic (Rule N10)",
        "// =============================================================================",
        "/* verilator lint_save */",
        "/* verilator lint_off WIDTHEXPAND */",
        "/* verilator lint_off UNUSEDSIGNAL */",
        "/* verilator lint_off DECLFILENAME */",
        f"module {mod_name} (",
        f"    input  wire [{in_bits-1}:0] x_flat,",
        f"    input  wire [{weight_bits-1}:0] w_flat,",
        f"    output wire [{out_bits-1}:0] y_flat",
        ");",
        "",
        "    // Unpack input activation vector x_flat",
    ]

    for k in range(kt):
        hi = 8 * (k + 1) - 1
        lo = 8 * k
        lines.append(f"    wire signed [7:0] x_{k} = x_flat[{hi}:{lo}];")

    lines.append("")
    lines.append("    // Unpack input weight vector w_flat and compute signed products")

    for n in range(nt):
        for k in range(kt):
            idx = n * kt + k
            w_hi = 8 * (idx + 1) - 1
            w_lo = 8 * idx
            lines.append(f"    wire signed [7:0] w_{n}_{k} = w_flat[{w_hi}:{w_lo}];")
            lines.append(
                f"    wire signed [15:0] prod_{n}_{k} = x_{k} * w_{n}_{k};"
            )

    lines.append("")
    lines.append(f"    // Accumulate products with explicit {acc_bits}-bit sign extension")

    for n in range(nt):
        ext_terms: list[str] = []
        for k in range(kt):
            ext_terms.append(f"$signed({{{{16{{prod_{n}_{k}[15]}}}}, prod_{n}_{k}}})")
        sum_expr = " + ".join(ext_terms)
        lines.append(f"    wire signed [{acc_bits-1}:0] y_row_{n} = {sum_expr};")

    lines.append("")
    lines.append("    // Pack outputs into y_flat bus (element 0 in LSBs)")
    for n in range(nt):
        y_hi = acc_bits * (n + 1) - 1
        y_lo = acc_bits * n
        lines.append(f"    assign y_flat[{y_hi}:{y_lo}] = y_row_{n};")

    lines.append("")
    lines.append("endmodule")
    lines.append("/* verilator lint_restore */")
    lines.append("")

    return "\n".join(lines)


def generate_full_module_p(
    op: LinearOp,
    manifest: TileManifest,
    acc_bits: int = 32,
) -> str:
    """Generate complete Design P top module with W_reg register array.

    Interface:
        module llmhw_<name>_p (
            input  wire clk,
            input  wire rst_n,
            input  wire valid_in,
            input  wire [8*K'-1:0] x_flat,
            input  wire w_we,
            input  wire [addr_bits-1:0] w_addr,
            input  wire signed [7:0] w_data,
            output reg  valid_out,
            output reg  signed [acc_bits*N-1:0] y_flat
        );
    """
    raw_name = f"llmhw_{op.name}_p"
    top_name = sanitize_identifier(raw_name)
    tile_core_name = sanitize_identifier(f"{raw_name}_tile_core")

    n_padded = manifest.n_padded
    k_padded = manifest.k_padded
    tile_n = manifest.tile_n
    tile_k = manifest.tile_k
    num_n_tiles = n_padded // tile_n
    num_k_tiles = k_padded // tile_k
    total_weights = n_padded * k_padded
    addr_bits = max(1, math.ceil(math.log2(total_weights)))

    in_bits = 8 * k_padded
    out_bits_padded = acc_bits * n_padded

    tile_core_code = generate_tile_core_p(
        tile_core_name,
        nt=tile_n,
        kt=tile_k,
        acc_bits=acc_bits,
        manifest_hash=manifest.manifest_hash,
    )

    lines: list[str] = [
        "// =============================================================================",
        f"// Top-Level Module: {top_name} (Design P Programmable Baseline)",
        f"// Operation: {op.name} (N={op.out_features}, K={op.in_features})",
        f"// Tiling: TileN={tile_n}, TileK={tile_k}, PaddedN={n_padded}, PaddedK={k_padded}",
        f"// Grid: {num_n_tiles}x{num_k_tiles} tiles, Total Weights={total_weights}, AddrBits={addr_bits}",
        f"// Manifest Hash: {manifest.manifest_hash}",
        "// =============================================================================",
        "",
        tile_core_code,
        "/* verilator lint_save */",
        "/* verilator lint_off WIDTHEXPAND */",
        "/* verilator lint_off UNUSEDSIGNAL */",
        "/* verilator lint_off DECLFILENAME */",
        f"module {top_name} (",
        "    input  wire clk,",
        "    input  wire rst_n,",
        "    input  wire valid_in,",
        f"    input  wire [{in_bits-1}:0] x_flat,",
        "    input  wire w_we,",
        f"    input  wire [{addr_bits-1}:0] w_addr,",
        "    input  wire signed [7:0] w_data,",
        "    output reg  valid_out,",
        f"    output reg  [{out_bits_padded-1}:0] y_flat",
        ");",
        "",
        f"    // Weight storage array W_reg ({total_weights} INT8 registers)",
        f"    reg signed [7:0] W_reg [0:{total_weights-1}];",
        "",
        "    always @(posedge clk) begin",
        "        if (w_we) begin",
        "            W_reg[w_addr] <= w_data;",
        "        end",
        "    end",
        "",
        "    // Slicing input activation vector x_flat into per-tile segments",
    ]

    for j in range(num_k_tiles):
        tile_in_bits = 8 * tile_k
        hi = 8 * (j + 1) * tile_k - 1
        lo = 8 * j * tile_k
        lines.append(f"    wire [{tile_in_bits-1}:0] x_tile_{j}_flat = x_flat[{hi}:{lo}];")

    lines.append("")
    lines.append("    // Wiring W_reg storage to tile weight buses and instantiating tile cores")

    for i in range(num_n_tiles):
        for j in range(num_k_tiles):
            # Form packed w_tile_<i_j>_flat
            w_wire_names: list[str] = []
            for r in range(tile_n):
                for c in range(tile_k):
                    loc_idx = r * tile_k + c
                    g_r = i * tile_n + r
                    g_c = j * tile_k + c
                    lin_idx = g_r * k_padded + g_c
                    w_name = f"w_{i}_{j}_{loc_idx}"
                    lines.append(f"    wire signed [7:0] {w_name} = W_reg[{lin_idx}];")
                    w_wire_names.append(w_name)

            # Concatenate MSB to LSB
            reversed_wires = list(reversed(w_wire_names))
            concat_expr = "{ " + ", ".join(reversed_wires) + " }"
            w_bus_bits = 8 * tile_n * tile_k
            lines.append(
                f"    wire [{w_bus_bits-1}:0] w_tile_{i}_{j}_flat = {concat_expr};"
            )

            # Output wire
            tile_out_bits = acc_bits * tile_n
            lines.append(f"    wire [{tile_out_bits-1}:0] y_tile_{i}_{j}_flat;")
            lines.append(f"    {tile_core_name} u_tile_{i}_{j} (")
            lines.append(f"        .x_flat(x_tile_{j}_flat),")
            lines.append(f"        .w_flat(w_tile_{i}_{j}_flat),")
            lines.append(f"        .y_flat(y_tile_{i}_{j}_flat)")
            lines.append("    );")

    lines.append("")
    lines.append("    // Reduction adder tree over j tiles and constant bias addition")

    bias_arr = None
    if op.bias is not None and op.bias.data is not None:
        bias_arr = op.bias.data

    for n in range(n_padded):
        i = n // tile_n
        r = n % tile_n
        y_lo = acc_bits * r
        y_hi = acc_bits * (r + 1) - 1

        terms: list[str] = []
        for j in range(num_k_tiles):
            terms.append(f"$signed(y_tile_{i}_{j}_flat[{y_hi}:{y_lo}])")

        if bias_arr is not None and n < len(bias_arr):
            b_val = int(bias_arr[n])
            if b_val >= 0:
                terms.append(f"{acc_bits}'sd{b_val}")
            else:
                terms.append(f"-{acc_bits}'sd{abs(b_val)}")

        sum_expr = " + ".join(terms)
        lines.append(f"    wire signed [{acc_bits-1}:0] y_reduced_{n} = {sum_expr};")

    lines.append("")
    lines.append("    // Pack reduced rows into combinational y_comb (unpadded)")
    lines.append(f"    wire signed [{out_bits_padded-1}:0] y_padded;")
    for n in range(n_padded):
        p_hi = acc_bits * (n + 1) - 1
        p_lo = acc_bits * n
        lines.append(f"    assign y_padded[{p_hi}:{p_lo}] = y_reduced_{n};")

    lines.append("")
    lines.append(
        f"    wire signed [{out_bits_padded-1}:0] y_comb = y_padded;"
    )
    lines.append("")
    lines.append("    // Registered timing wrapper (Latency L=1)")
    lines.append("    always @(posedge clk) begin")
    lines.append("        if (!rst_n) begin")
    lines.append("            valid_out <= 1'b0;")
    lines.append("        end else begin")
    lines.append("            valid_out <= valid_in;")
    lines.append("        end")
    lines.append("    end")
    lines.append("")
    lines.append("    always @(posedge clk) begin")
    lines.append("        if (valid_in) begin")
    lines.append("            y_flat <= y_comb;")
    lines.append("        end")
    lines.append("    end")
    lines.append("")
    lines.append("endmodule")
    lines.append("/* verilator lint_restore */")
    lines.append("")

    return "\n".join(lines)


def generate_full_module_p_bus(
    op: LinearOp,
    manifest: TileManifest,
    acc_bits: int = 32,
) -> str:
    """Generate complete Design P_bus top module with primary-input w_flat bus.

    Interface:
        module llmhw_<name>_p_bus (
            input  wire clk,
            input  wire rst_n,
            input  wire valid_in,
            input  wire [8*K'-1:0] x_flat,
            input  wire [8*total_weights-1:0] w_flat,
            output reg  valid_out,
            output reg  signed [acc_bits*N-1:0] y_flat
        );
    """
    raw_name = f"llmhw_{op.name}_p_bus"
    top_name = sanitize_identifier(raw_name)
    tile_core_name = sanitize_identifier(f"{raw_name}_tile_core")

    n_padded = manifest.n_padded
    k_padded = manifest.k_padded
    tile_n = manifest.tile_n
    tile_k = manifest.tile_k
    num_n_tiles = n_padded // tile_n
    num_k_tiles = k_padded // tile_k
    total_weights = n_padded * k_padded

    in_bits = 8 * k_padded
    weight_bits = 8 * total_weights
    out_bits_padded = acc_bits * n_padded

    tile_core_code = generate_tile_core_p(
        tile_core_name,
        nt=tile_n,
        kt=tile_k,
        acc_bits=acc_bits,
        manifest_hash=manifest.manifest_hash,
    )

    lines: list[str] = [
        "// =============================================================================",
        f"// Top-Level Module: {top_name} (Design P_bus Logic-Only Baseline)",
        f"// Operation: {op.name} (N={op.out_features}, K={op.in_features})",
        f"// Tiling: TileN={tile_n}, TileK={tile_k}, PaddedN={n_padded}, PaddedK={k_padded}",
        f"// Grid: {num_n_tiles}x{num_k_tiles} tiles, Total Weights={total_weights}",
        f"// Manifest Hash: {manifest.manifest_hash}",
        "// =============================================================================",
        "",
        tile_core_code,
        "/* verilator lint_save */",
        "/* verilator lint_off WIDTHEXPAND */",
        "/* verilator lint_off UNUSEDSIGNAL */",
        "/* verilator lint_off DECLFILENAME */",
        f"module {top_name} (",
        "    input  wire clk,",
        "    input  wire rst_n,",
        "    input  wire valid_in,",
        f"    input  wire [{in_bits-1}:0] x_flat,",
        f"    input  wire [{weight_bits-1}:0] w_flat,",
        "    output reg  valid_out,",
        f"    output reg  [{out_bits_padded-1}:0] y_flat",
        ");",
        "",
        "    // Slicing input activation vector x_flat into per-tile segments",
    ]

    for j in range(num_k_tiles):
        tile_in_bits = 8 * tile_k
        hi = 8 * (j + 1) * tile_k - 1
        lo = 8 * j * tile_k
        lines.append(f"    wire [{tile_in_bits-1}:0] x_tile_{j}_flat = x_flat[{hi}:{lo}];")

    lines.append("")
    lines.append("    // Wiring primary w_flat bus to per-tile weight buses and instantiating tile cores")

    for i in range(num_n_tiles):
        for j in range(num_k_tiles):
            # Extract weights directly from w_flat
            w_wire_names: list[str] = []
            for r in range(tile_n):
                for c in range(tile_k):
                    loc_idx = r * tile_k + c
                    g_r = i * tile_n + r
                    g_c = j * tile_k + c
                    lin_idx = g_r * k_padded + g_c
                    w_name = f"w_{i}_{j}_{loc_idx}"
                    w_hi = 8 * (lin_idx + 1) - 1
                    w_lo = 8 * lin_idx
                    lines.append(
                        f"    wire signed [7:0] {w_name} = w_flat[{w_hi}:{w_lo}];"
                    )
                    w_wire_names.append(w_name)

            reversed_wires = list(reversed(w_wire_names))
            concat_expr = "{ " + ", ".join(reversed_wires) + " }"
            w_bus_bits = 8 * tile_n * tile_k
            lines.append(
                f"    wire [{w_bus_bits-1}:0] w_tile_{i}_{j}_flat = {concat_expr};"
            )

            tile_out_bits = acc_bits * tile_n
            lines.append(f"    wire [{tile_out_bits-1}:0] y_tile_{i}_{j}_flat;")
            lines.append(f"    {tile_core_name} u_tile_{i}_{j} (")
            lines.append(f"        .x_flat(x_tile_{j}_flat),")
            lines.append(f"        .w_flat(w_tile_{i}_{j}_flat),")
            lines.append(f"        .y_flat(y_tile_{i}_{j}_flat)")
            lines.append("    );")

    lines.append("")
    lines.append("    // Reduction adder tree over j tiles and constant bias addition")

    bias_arr = None
    if op.bias is not None and op.bias.data is not None:
        bias_arr = op.bias.data

    for n in range(n_padded):
        i = n // tile_n
        r = n % tile_n
        y_lo = acc_bits * r
        y_hi = acc_bits * (r + 1) - 1

        terms: list[str] = []
        for j in range(num_k_tiles):
            terms.append(f"$signed(y_tile_{i}_{j}_flat[{y_hi}:{y_lo}])")

        if bias_arr is not None and n < len(bias_arr):
            b_val = int(bias_arr[n])
            if b_val >= 0:
                terms.append(f"{acc_bits}'sd{b_val}")
            else:
                terms.append(f"-{acc_bits}'sd{abs(b_val)}")

        sum_expr = " + ".join(terms)
        lines.append(f"    wire signed [{acc_bits-1}:0] y_reduced_{n} = {sum_expr};")

    lines.append("")
    lines.append("    // Pack reduced rows into combinational y_comb (unpadded)")
    lines.append(f"    wire signed [{out_bits_padded-1}:0] y_padded;")
    for n in range(n_padded):
        p_hi = acc_bits * (n + 1) - 1
        p_lo = acc_bits * n
        lines.append(f"    assign y_padded[{p_hi}:{p_lo}] = y_reduced_{n};")

    lines.append("")
    lines.append(
        f"    wire signed [{out_bits_padded-1}:0] y_comb = y_padded;"
    )
    lines.append("")
    lines.append("    // Registered timing wrapper (Latency L=1)")
    lines.append("    always @(posedge clk) begin")
    lines.append("        if (!rst_n) begin")
    lines.append("            valid_out <= 1'b0;")
    lines.append("        end else begin")
    lines.append("            valid_out <= valid_in;")
    lines.append("        end")
    lines.append("    end")
    lines.append("")
    lines.append("    always @(posedge clk) begin")
    lines.append("        if (valid_in) begin")
    lines.append("            y_flat <= y_comb;")
    lines.append("        end")
    lines.append("    end")
    lines.append("")
    lines.append("endmodule")
    lines.append("/* verilator lint_restore */")
    lines.append("")

    return "\n".join(lines)
