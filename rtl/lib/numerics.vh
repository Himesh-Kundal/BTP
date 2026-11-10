// numerics.vh - Part II Numeric Contract v2 Shared Functions and Macros
// Contract references: docs/part2/02_architecture_and_contracts.md §3 (N2-2, N2-3)
//
// Include inside module body:
//   `include "numerics.vh"
//
// Defines:
// - Macros: `RSHIFT_RND(val, sh), `SAT_INT8(val), `SAT_GEN(val, min_v, max_v)
// - Functions: rshift_rnd64, rshift_rnd96, sat_int8, sat_generic, sat_generic96, requant_to_int8

`ifndef _NUMERICS_MACROS_VH_
`define _NUMERICS_MACROS_VH_

// Round-half-up toward +inf macro (expression)
`define RSHIFT_RND(val, sh) \
    (((sh) == 0) ? (val) : (((val) + (64'sd1 <<< ((sh) - 1))) >>> (sh)))

// Symmetric INT8 saturation macro
`define SAT_INT8(val) \
    (((val) > 64'sd127) ? 8'sd127 : (((val) < -64'sd127) ? -8'sd127 : (val)[7:0]))

// Generic saturation macro
`define SAT_GEN(val, min_v, max_v) \
    (((val) > (max_v)) ? (max_v) : (((val) < (min_v)) ? (min_v) : (val)))

`endif // _NUMERICS_MACROS_VH_

// 64-bit signed arithmetic right shift with round-half-up toward +inf
// s = 0 is identity
function signed [63:0] rshift_rnd64;
    input signed [63:0] val;
    input [5:0] sh;
    reg signed [63:0] offset;
    begin
        if (sh == 6'd0) begin
            rshift_rnd64 = val;
        end else begin
            offset = 64'sd1 <<< (sh - 1);
            rshift_rnd64 = (val + offset) >>> sh;
        end
    end
endfunction

// 96-bit signed arithmetic right shift with round-half-up toward +inf
function signed [95:0] rshift_rnd96;
    input signed [95:0] val;
    input [6:0] sh;
    reg signed [95:0] offset;
    begin
        if (sh == 7'd0) begin
            rshift_rnd96 = val;
        end else begin
            offset = 96'sd1 <<< (sh - 1);
            rshift_rnd96 = (val + offset) >>> sh;
        end
    end
endfunction

// Saturate 64-bit signed to symmetric INT8 [-127, 127]
function signed [7:0] sat_int8;
    input signed [63:0] val;
    begin
        if (val > 64'sd127) begin
            sat_int8 = 8'sd127;
        end else if (val < -64'sd127) begin
            sat_int8 = -8'sd127;
        end else begin
            sat_int8 = val[7:0];
        end
    end
endfunction

// Saturate 64-bit signed to arbitrary [min_v, max_v]
function signed [63:0] sat_generic;
    input signed [63:0] val;
    input signed [63:0] min_v;
    input signed [63:0] max_v;
    begin
        if (val > max_v) begin
            sat_generic = max_v;
        end else if (val < min_v) begin
            sat_generic = min_v;
        end else begin
            sat_generic = val;
        end
    end
endfunction

// Saturate 96-bit signed to arbitrary [min_v, max_v]
function signed [95:0] sat_generic96;
    input signed [95:0] val;
    input signed [95:0] min_v;
    input signed [95:0] max_v;
    begin
        if (val > max_v) begin
            sat_generic96 = max_v;
        end else if (val < min_v) begin
            sat_generic96 = min_v;
        end else begin
            sat_generic96 = val;
        end
    end
endfunction

// Dyadic requantization for 64-bit intermediate products
function signed [7:0] requant_to_int8;
    input signed [31:0] acc;
    input signed [31:0] m_scale;
    input [5:0] sh;
    reg signed [63:0] prod;
    reg signed [63:0] shifted;
    begin
        prod = {{32{acc[31]}}, acc} * {{32{m_scale[31]}}, m_scale};
        shifted = rshift_rnd64(prod, sh);
        requant_to_int8 = sat_int8(shifted);
    end
endfunction
