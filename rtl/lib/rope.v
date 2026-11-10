// rope.v - Parameterized Rotary Position Embedding (RoPE) Unit
// Contract references: docs/part2/02_architecture_and_contracts.md §3 (N2-8), §6
//
// Implements rotate-half coordinate transform:
// x1' = sat(rshift_rnd(x1 * cos - x2 * sin, TABLE_SHIFT), -127, 127)
// x2' = sat(rshift_rnd(x2 * cos + x1 * sin, TABLE_SHIFT), -127, 127)

`timescale 1ns / 1ps

/* verilator lint_off BLKSEQ */
module rope #(
    parameter DATA_WIDTH  = 8,
    parameter TABLE_WIDTH = 16,
    parameter TABLE_SHIFT = 14,
    parameter OUT_WIDTH   = 8,
    parameter signed [OUT_WIDTH-1:0] OUT_MIN = -8'sd127,
    parameter signed [OUT_WIDTH-1:0] OUT_MAX = 8'sd127
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,
    input  wire [15:0]                   num_heads,
    input  wire [15:0]                   head_dim,
    input  wire [15:0]                   pos,
    output reg                           busy,
    output reg                           done,

    // Source buffer read interface
    output reg  [15:0]                   src_addr,
    input  wire signed [DATA_WIDTH-1:0]  src_data,

    // Table ROM read interface
    output reg  [15:0]                   table_addr,
    input  wire signed [TABLE_WIDTH-1:0] cos_data,
    input  wire signed [TABLE_WIDTH-1:0] sin_data,

    // Destination buffer write interface
    output reg                           dst_we,
    output reg  [15:0]                   dst_addr,
    output reg  signed [OUT_WIDTH-1:0]   dst_data
);

    `include "numerics.vh"

    // FSM States
    localparam S_IDLE       = 3'd0;
    localparam S_REQ_X1     = 3'd1;
    localparam S_REQ_X2     = 3'd2;
    localparam S_CAP_X1     = 3'd3;
    localparam S_CAP_X2     = 3'd4;
    localparam S_WRITE_ROT1 = 3'd5;
    localparam S_WRITE_ROT2 = 3'd6;
    localparam S_DONE       = 3'd7;

    reg [2:0] state;

    reg [15:0] head_idx;
    reg [15:0] i_idx;
    reg [15:0] half_dim;

    reg signed [DATA_WIDTH-1:0]  reg_x1;
    reg signed [DATA_WIDTH-1:0]  reg_x2;
    reg signed [TABLE_WIDTH-1:0] reg_cos;
    reg signed [TABLE_WIDTH-1:0] reg_sin;

    // Address calculations
    wire [15:0] head_base = head_idx * head_dim;
    wire [15:0] addr_x1   = head_base + i_idx;
    wire [15:0] addr_x2   = head_base + half_dim + i_idx;
    wire [15:0] addr_tbl  = pos * half_dim + i_idx;

    // Arithmetic computations
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [63:0] prod1_a = {{56{reg_x1[DATA_WIDTH-1]}}, reg_x1} * {{48{reg_cos[TABLE_WIDTH-1]}}, reg_cos};
    wire signed [63:0] prod1_b = {{56{reg_x2[DATA_WIDTH-1]}}, reg_x2} * {{48{reg_sin[TABLE_WIDTH-1]}}, reg_sin};
    wire signed [63:0] diff1   = prod1_a - prod1_b;
    wire signed [63:0] rot1_64 = rshift_rnd64(diff1, TABLE_SHIFT[5:0]);
    wire signed [63:0] sat1_64 = sat_generic(rot1_64,
                                             {{(64-OUT_WIDTH){OUT_MIN[OUT_WIDTH-1]}}, OUT_MIN},
                                             {{(64-OUT_WIDTH){OUT_MAX[OUT_WIDTH-1]}}, OUT_MAX});

    wire signed [63:0] prod2_a = {{56{reg_x2[DATA_WIDTH-1]}}, reg_x2} * {{48{reg_cos[TABLE_WIDTH-1]}}, reg_cos};
    wire signed [63:0] prod2_b = {{56{reg_x1[DATA_WIDTH-1]}}, reg_x1} * {{48{reg_sin[TABLE_WIDTH-1]}}, reg_sin};
    wire signed [63:0] sum2    = prod2_a + prod2_b;
    wire signed [63:0] rot2_64 = rshift_rnd64(sum2, TABLE_SHIFT[5:0]);
    wire signed [63:0] sat2_64 = sat_generic(rot2_64,
                                             {{(64-OUT_WIDTH){OUT_MIN[OUT_WIDTH-1]}}, OUT_MIN},
                                             {{(64-OUT_WIDTH){OUT_MAX[OUT_WIDTH-1]}}, OUT_MAX});
    /* verilator lint_on UNUSEDSIGNAL */

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            busy       <= 1'b0;
            done       <= 1'b0;
            src_addr   <= 16'd0;
            table_addr <= 16'd0;
            dst_we     <= 1'b0;
            dst_addr   <= 16'd0;
            dst_data   <= {OUT_WIDTH{1'b0}};
            head_idx   <= 16'd0;
            i_idx      <= 16'd0;
            half_dim   <= 16'd0;
            reg_x1     <= {DATA_WIDTH{1'b0}};
            reg_x2     <= {DATA_WIDTH{1'b0}};
            reg_cos    <= {TABLE_WIDTH{1'b0}};
            reg_sin    <= {TABLE_WIDTH{1'b0}};
        end else begin
            dst_we <= 1'b0;
            done   <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy     <= 1'b1;
                        head_idx <= 16'd0;
                        i_idx    <= 16'd0;
                        half_dim <= head_dim >> 1;
                        state    <= S_REQ_X1;
                    end
                end

                S_REQ_X1: begin
                    src_addr   <= addr_x1;
                    table_addr <= addr_tbl;
                    state      <= S_REQ_X2;
                end

                S_REQ_X2: begin
                    src_addr <= addr_x2;
                    state    <= S_CAP_X1;
                end

                S_CAP_X1: begin
                    reg_x1  <= src_data;
                    reg_cos <= cos_data;
                    reg_sin <= sin_data;
                    state   <= S_CAP_X2;
                end

                S_CAP_X2: begin
                    reg_x2 <= src_data;
                    state  <= S_WRITE_ROT1;
                end

                S_WRITE_ROT1: begin
                    dst_we   <= 1'b1;
                    dst_addr <= addr_x1;
                    dst_data <= sat1_64[OUT_WIDTH-1:0];
                    state    <= S_WRITE_ROT2;
                end

                S_WRITE_ROT2: begin
                    dst_we   <= 1'b1;
                    dst_addr <= addr_x2;
                    dst_data <= sat2_64[OUT_WIDTH-1:0];

                    // Advance indices
                    if (i_idx + 16'd1 < half_dim) begin
                        i_idx <= i_idx + 16'd1;
                        state <= S_REQ_X1;
                    end else begin
                        i_idx <= 16'd0;
                        if (head_idx + 16'd1 < num_heads) begin
                            head_idx <= head_idx + 16'd1;
                            state    <= S_REQ_X1;
                        end else begin
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
