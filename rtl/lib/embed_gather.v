// embed_gather.v - Embedding Table Gather Unit
// Contract references: docs/part2/02_architecture_and_contracts.md §3 (N2-5), §6
//
// Computes embedding vector gather for token_id:
// addr_i = token_id * dim + i
// raw_w_i = embed_table[addr_i]
// out_i = sat(rshift_rnd(raw_w_i * M, s), OUT_MIN, OUT_MAX)

`timescale 1ns / 1ps

/* verilator lint_off BLKSEQ */
module embed_gather #(
    parameter WEIGHT_WIDTH = 8,
    parameter OUT_WIDTH    = 24,
    parameter signed [OUT_WIDTH-1:0] OUT_MIN = -24'sd8388608,
    parameter signed [OUT_WIDTH-1:0] OUT_MAX = 24'sd8388607
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,
    input  wire [15:0]                   token_id,
    input  wire [15:0]                   dim,
    input  wire signed [31:0]            M,
    input  wire [5:0]                    s,
    output reg                           busy,
    output reg                           done,

    // Embedding ROM read interface (1-cycle synchronous)
    output reg  [31:0]                   embed_addr,
    input  wire signed [WEIGHT_WIDTH-1:0] embed_data,

    // Destination buffer write interface
    output reg                           dst_we,
    output reg  [15:0]                   dst_addr,
    output reg  signed [OUT_WIDTH-1:0]   dst_data
);

    `include "numerics.vh"

    // FSM States
    localparam S_IDLE   = 2'd0;
    localparam S_STREAM = 2'd1;
    localparam S_DONE   = 2'd2;

    reg [1:0] state;

    reg [31:0] token_base;
    reg [15:0] rd_cnt;
    reg [15:0] out_cnt;
    reg [1:0]  read_pipe;

    // Intermediate calculations
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [63:0] prod    = {{56{embed_data[WEIGHT_WIDTH-1]}}, embed_data} * {{32{M[31]}}, M};
    wire signed [63:0] shifted = rshift_rnd64(prod, s);
    wire signed [63:0] sat_out = sat_generic(shifted,
                                             {{(64-OUT_WIDTH){OUT_MIN[OUT_WIDTH-1]}}, OUT_MIN},
                                             {{(64-OUT_WIDTH){OUT_MAX[OUT_WIDTH-1]}}, OUT_MAX});
    /* verilator lint_on UNUSEDSIGNAL */

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            busy       <= 1'b0;
            done       <= 1'b0;
            embed_addr <= 32'd0;
            dst_we     <= 1'b0;
            dst_addr   <= 16'd0;
            dst_data   <= {OUT_WIDTH{1'b0}};
            token_base <= 32'd0;
            rd_cnt     <= 16'd0;
            out_cnt    <= 16'd0;
            read_pipe  <= 2'b00;
        end else begin
            dst_we <= 1'b0;
            done   <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy       <= 1'b1;
                        token_base <= {16'd0, token_id} * {16'd0, dim};
                        rd_cnt     <= 16'd0;
                        out_cnt    <= 16'd0;
                        read_pipe  <= 2'b00;
                        state      <= S_STREAM;
                    end
                end

                S_STREAM: begin
                    // Stage 0: Drive address to embedding ROM
                    if (rd_cnt < dim) begin
                        embed_addr <= token_base + {16'd0, rd_cnt};
                        rd_cnt     <= rd_cnt + 16'd1;
                        read_pipe  <= {read_pipe[0], 1'b1};
                    end else begin
                        read_pipe  <= {read_pipe[0], 1'b0};
                    end

                    // Stage 1: Capture data from ROM and scale
                    if (read_pipe[1]) begin
                        dst_data <= sat_out[OUT_WIDTH-1:0];
                        dst_addr <= out_cnt;
                        dst_we   <= 1'b1;
                        out_cnt  <= out_cnt + 16'd1;

                        if (out_cnt + 16'd1 == dim) begin
                            state <= S_DONE;
                        end
                    end
                end

                S_DONE: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
/* verilator lint_on BLKSEQ */
