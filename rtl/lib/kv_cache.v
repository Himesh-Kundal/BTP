// kv_cache.v - Dual-Port Parameterized Synchronous RAM for KV Cache
// Contract references: docs/part2/02_architecture_and_contracts.md §3 (N2-9), §6

`timescale 1ns / 1ps

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
module kv_cache #(
    parameter NUM_LAYERS   = 30,
    parameter NUM_KV_HEADS = 3,
    parameter MAX_CTX      = 256,
    parameter HEAD_DIM     = 64,
    parameter DATA_WIDTH   = 8
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // Configuration / mode
    input  wire                          use_direct_addr,

    // Port A: Write Interface (Synchronous)
    input  wire                          wr_en,
    input  wire [ADDR_W-1:0]             wr_addr_in,
    input  wire [LAYER_W-1:0]            wr_layer,
    input  wire                          wr_is_v,
    input  wire [KV_HEAD_W-1:0]          wr_kv_head,
    input  wire [POS_W-1:0]              wr_pos,
    input  wire [DIM_W-1:0]              wr_dim,
    input  wire signed [DATA_WIDTH-1:0]  wr_data,

    // Port B: Read Interface (Synchronous, 1-cycle latency)
    input  wire                          rd_en,
    input  wire [ADDR_W-1:0]             rd_addr_in,
    input  wire [LAYER_W-1:0]            rd_layer,
    input  wire                          rd_is_v,
    input  wire [KV_HEAD_W-1:0]          rd_kv_head,
    input  wire [POS_W-1:0]              rd_pos,
    input  wire [DIM_W-1:0]              rd_dim,
    output reg  signed [DATA_WIDTH-1:0]  rd_data
);

    // Derived dimensions
    localparam TOTAL_WORDS = NUM_LAYERS * 2 * NUM_KV_HEADS * MAX_CTX * HEAD_DIM;

    localparam LAYER_W   = (NUM_LAYERS > 1) ? $clog2(NUM_LAYERS) : 1;
    localparam KV_HEAD_W = (NUM_KV_HEADS > 1) ? $clog2(NUM_KV_HEADS) : 1;
    localparam POS_W     = (MAX_CTX > 1) ? $clog2(MAX_CTX) : 1;
    localparam DIM_W     = (HEAD_DIM > 1) ? $clog2(HEAD_DIM) : 1;
    localparam ADDR_W    = (TOTAL_WORDS > 1) ? $clog2(TOTAL_WORDS) : 1;

    // Address Calculation Function
    function [ADDR_W-1:0] calc_addr;
        input [LAYER_W-1:0]   l;
        input                 v;
        input [KV_HEAD_W-1:0] h;
        input [POS_W-1:0]     p;
        input [DIM_W-1:0]     d;
        reg [ADDR_W+15:0]     temp;
        begin
            temp = l;
            temp = temp * 2 + v;
            temp = temp * NUM_KV_HEADS + h;
            temp = temp * MAX_CTX + p;
            temp = temp * HEAD_DIM + d;
            calc_addr = temp[ADDR_W-1:0];
        end
    endfunction

    wire [ADDR_W-1:0] wr_calc_addr = calc_addr(wr_layer, wr_is_v, wr_kv_head, wr_pos, wr_dim);
    wire [ADDR_W-1:0] rd_calc_addr = calc_addr(rd_layer, rd_is_v, rd_kv_head, rd_pos, rd_dim);

    wire [ADDR_W-1:0] effective_wr_addr = use_direct_addr ? wr_addr_in : wr_calc_addr;
    wire [ADDR_W-1:0] effective_rd_addr = use_direct_addr ? rd_addr_in : rd_calc_addr;

    // Memory array
    reg signed [DATA_WIDTH-1:0] mem [0:TOTAL_WORDS-1];

    // Synchronous write
    always @(posedge clk) begin
        if (wr_en) begin
            mem[effective_wr_addr] <= wr_data;
        end
    end

    // Synchronous read (1-cycle read latency)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_data <= {DATA_WIDTH{1'b0}};
        end else if (rd_en) begin
            rd_data <= mem[effective_rd_addr];
        end
    end

endmodule
/* verilator lint_on WIDTHEXPAND */
/* verilator lint_on WIDTHTRUNC */
