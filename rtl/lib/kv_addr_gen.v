// kv_addr_gen.v - Parameterized Address Generator and GQA Mapper for KV Cache
// Contract references: docs/part2/02_architecture_and_contracts.md §3 (N2-9), §6

`timescale 1ns / 1ps

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
module kv_addr_gen #(
    parameter NUM_LAYERS   = 30,
    parameter NUM_KV_HEADS = 3,
    parameter NUM_Q_HEADS  = 9,
    parameter MAX_CTX      = 256,
    parameter HEAD_DIM     = 64
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // GQA Mapping static port
    input  wire [Q_HEAD_W-1:0]           q_head_in,
    output wire [KV_HEAD_W-1:0]          kv_head_out,

    // Direct Coordinate translation static port
    input  wire [LAYER_W-1:0]            coord_layer,
    input  wire                          coord_is_v,
    input  wire [KV_HEAD_W-1:0]          coord_kv_head,
    input  wire [POS_W-1:0]              coord_pos,
    input  wire [DIM_W-1:0]              coord_dim,
    output wire [ADDR_W-1:0]             coord_addr,

    // Streaming generator interface
    input  wire                          start,
    input  wire                          step,        // Clock enable / advance signal
    input  wire                          mode,        // 0: single vector (dim 0..hd-1), 1: sequence (pos 0..max_pos, dim 0..hd-1)
    input  wire [LAYER_W-1:0]            stream_layer,
    input  wire                          stream_is_v,
    input  wire [KV_HEAD_W-1:0]          stream_kv_head,
    input  wire [POS_W-1:0]              stream_pos,      // Used for mode 0
    input  wire [POS_W-1:0]              stream_max_pos,  // Used for mode 1
    output reg                           busy,
    output reg                           done,
    output reg                           out_valid,
    output wire [ADDR_W-1:0]             out_addr,
    output reg  [POS_W-1:0]              out_pos,
    output reg  [DIM_W-1:0]              out_dim,
    output wire                          last_dim,
    output wire                          last_token
);

    // Derived width calculations
    localparam GQA_GROUP   = NUM_Q_HEADS / NUM_KV_HEADS;
    localparam TOTAL_WORDS = NUM_LAYERS * 2 * NUM_KV_HEADS * MAX_CTX * HEAD_DIM;

    localparam LAYER_W   = (NUM_LAYERS > 1) ? $clog2(NUM_LAYERS) : 1;
    localparam KV_HEAD_W = (NUM_KV_HEADS > 1) ? $clog2(NUM_KV_HEADS) : 1;
    localparam Q_HEAD_W  = (NUM_Q_HEADS > 1) ? $clog2(NUM_Q_HEADS) : 1;
    localparam POS_W     = (MAX_CTX > 1) ? $clog2(MAX_CTX) : 1;
    localparam DIM_W     = (HEAD_DIM > 1) ? $clog2(HEAD_DIM) : 1;
    localparam ADDR_W    = (TOTAL_WORDS > 1) ? $clog2(TOTAL_WORDS) : 1;

    // GQA Head Mapping: q_head / GQA_GROUP
    /* verilator lint_off UNUSEDSIGNAL */
    wire [31:0] div_val = q_head_in / GQA_GROUP;
    /* verilator lint_on UNUSEDSIGNAL */
    assign kv_head_out = div_val[KV_HEAD_W-1:0];

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

    // Direct static address output
    assign coord_addr = calc_addr(coord_layer, coord_is_v, coord_kv_head, coord_pos, coord_dim);

    // Streaming generator registers
    reg [LAYER_W-1:0]   reg_layer;
    reg                 reg_is_v;
    reg [KV_HEAD_W-1:0] reg_kv_head;
    reg [POS_W-1:0]     reg_max_pos;
    reg                 reg_mode;

    assign out_addr = calc_addr(reg_layer, reg_is_v, reg_kv_head, out_pos, out_dim);
    assign last_dim = (out_dim == (HEAD_DIM[DIM_W-1:0] - 1'b1));
    assign last_token = (reg_mode == 1'b0) ? last_dim :
                        (last_dim && (out_pos == reg_max_pos));

    // Streaming FSM
    localparam S_IDLE   = 2'd0;
    localparam S_STREAM = 2'd1;
    localparam S_DONE   = 2'd2;

    reg [1:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            busy        <= 1'b0;
            done        <= 1'b0;
            out_valid   <= 1'b0;
            out_pos     <= {POS_W{1'b0}};
            out_dim     <= {DIM_W{1'b0}};
            reg_layer   <= {LAYER_W{1'b0}};
            reg_is_v    <= 1'b0;
            reg_kv_head <= {KV_HEAD_W{1'b0}};
            reg_max_pos <= {POS_W{1'b0}};
            reg_mode    <= 1'b0;
        end else begin
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy        <= 1'b1;
                        reg_layer   <= stream_layer;
                        reg_is_v    <= stream_is_v;
                        reg_kv_head <= stream_kv_head;
                        reg_mode    <= mode;
                        if (mode == 1'b0) begin
                            out_pos     <= stream_pos;
                            reg_max_pos <= stream_pos;
                        end else begin
                            out_pos     <= {POS_W{1'b0}};
                            reg_max_pos <= stream_max_pos;
                        end
                        out_dim     <= {DIM_W{1'b0}};
                        out_valid   <= 1'b1;
                        state       <= S_STREAM;
                    end else begin
                        out_valid   <= 1'b0;
                    end
                end

                S_STREAM: begin
                    if (step) begin
                        if (last_token) begin
                            out_valid <= 1'b0;
                            state     <= S_DONE;
                        end else if (last_dim) begin
                            out_dim <= {DIM_W{1'b0}};
                            out_pos <= out_pos + 1'b1;
                        end else begin
                            out_dim <= out_dim + 1'b1;
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
/* verilator lint_on WIDTHEXPAND */
/* verilator lint_on WIDTHTRUNC */
