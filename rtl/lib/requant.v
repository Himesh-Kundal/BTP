// requant.v - Parameterized Dyadic Requantization Unit
// Contract references: docs/part2/02_architecture_and_contracts.md §3 (N2-3)
// Formula: y = sat(rshift_rnd(acc * M, shift), OUT_MIN, OUT_MAX)

`timescale 1ns / 1ps

module requant #(
    parameter ACC_WIDTH   = 32,
    parameter M_WIDTH     = 32,
    parameter SHIFT_WIDTH = 6,
    parameter OUT_WIDTH   = 8,
    parameter signed [OUT_WIDTH-1:0] OUT_MIN = -8'sd127,
    parameter signed [OUT_WIDTH-1:0] OUT_MAX = 8'sd127,
    parameter PIPELINED   = 1
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          valid_in,
    input  wire signed [ACC_WIDTH-1:0]   acc_in,
    input  wire signed [M_WIDTH-1:0]     M_in,
    input  wire        [SHIFT_WIDTH-1:0] shift_in,
    output reg                           valid_out,
    output reg  signed [OUT_WIDTH-1:0]   data_out
);

    `include "numerics.vh"

    localparam PROD_WIDTH = ACC_WIDTH + M_WIDTH;

    // Intermediate product
    wire signed [PROD_WIDTH-1:0] prod;
    assign prod = acc_in * M_in;

    // Rounding offset: 1 <<< (shift - 1) when shift > 0, else 0
    wire signed [PROD_WIDTH-1:0] offset;
    assign offset = (shift_in == {SHIFT_WIDTH{1'b0}}) ? {PROD_WIDTH{1'b0}} :
                    ({{ (PROD_WIDTH-1){1'b0} }, 1'b1} <<< (shift_in - 1'b1));

    wire signed [PROD_WIDTH-1:0] sum;
    assign sum = prod + offset;

    wire signed [PROD_WIDTH-1:0] shifted;
    assign shifted = (shift_in == {SHIFT_WIDTH{1'b0}}) ? prod : (sum >>> shift_in);

    // Saturation logic
    reg signed [OUT_WIDTH-1:0] sat_comb;
    always @(*) begin
        if (shifted > $signed({{ (PROD_WIDTH-OUT_WIDTH){OUT_MAX[OUT_WIDTH-1]} }, OUT_MAX})) begin
            sat_comb = OUT_MAX;
        end else if (shifted < $signed({{ (PROD_WIDTH-OUT_WIDTH){OUT_MIN[OUT_WIDTH-1]} }, OUT_MIN})) begin
            sat_comb = OUT_MIN;
        end else begin
            sat_comb = shifted[OUT_WIDTH-1:0];
        end
    end

    generate
        if (PIPELINED) begin : gen_pipelined
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    valid_out <= 1'b0;
                    data_out  <= {OUT_WIDTH{1'b0}};
                end else begin
                    valid_out <= valid_in;
                    if (valid_in) begin
                        data_out <= sat_comb;
                    end
                end
            end
        end else begin : gen_combinational
            always @(*) begin
                valid_out = valid_in;
                data_out  = sat_comb;
            end
        end
    endgenerate

endmodule
