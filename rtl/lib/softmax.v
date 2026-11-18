// softmax.v - Parameterized Softmax Unit (Base-2 Exponential Decomposition)
// Contract references: docs/part2/02_architecture_and_contracts.md §3 (N2-10), §6

`timescale 1ns / 1ps

/* verilator lint_off BLKSEQ */
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
module softmax #(
    parameter SCORES_WIDTH = 32,
    parameter EXP_WIDTH    = 18,
    parameter PROB_WIDTH   = 16,
    parameter MAX_CTX      = 256
)(
    input  wire                             clk,
    input  wire                             rst_n,
    input  wire                             start,
    input  wire [15:0]                      seq_len,
    output reg                              busy,
    output reg                              done,

    // Scores buffer read interface (1-cycle synchronous SRAM)
    output reg  [15:0]                      score_addr,
    input  wire signed [SCORES_WIDTH-1:0]   score_data,

    // Exp2 LUT ROM read interface
    output reg  [7:0]                       exp2_lut_addr,
    input  wire [EXP_WIDTH-1:0]             exp2_lut_data,

    // Probabilities buffer write interface
    output reg                              prob_we,
    output reg  [15:0]                      prob_addr,
    output reg  [PROB_WIDTH-1:0]            prob_data
);

    // FSM States
    localparam S_IDLE           = 4'd0;
    localparam S_PASS1_READ     = 4'd1;
    localparam S_PASS2_INIT     = 4'd2;
    localparam S_PASS2_REQ_SC   = 4'd3;
    localparam S_PASS2_WAIT_SC  = 4'd4;
    localparam S_PASS2_REQ_LUT  = 4'd5;
    localparam S_PASS2_WAIT_LUT = 4'd6;
    localparam S_PASS2_ACC      = 4'd7;
    localparam S_PASS3_NORM     = 4'd8;
    localparam S_DONE           = 4'd9;

    reg [3:0] state;

    reg signed [SCORES_WIDTH-1:0] max_score;
    reg [31:0]                    sigma;
    reg [15:0]                    rd_cnt;
    reg [15:0]                    acc_cnt;
    reg [1:0]                     read_pipe;

    localparam POS_W = (MAX_CTX > 1) ? $clog2(MAX_CTX) : 1;

    // Temporary local buffer for exp values
    reg [EXP_WIDTH-1:0] exp_mem [0:MAX_CTX-1];

    reg [31:0] cur_int_part;

    // Intermediate wires
    /* verilator lint_off UNUSEDSIGNAL */
    wire [EXP_WIDTH-1:0] exp_shift = exp2_lut_data >> (cur_int_part > 32'd16 ? 5'd16 : cur_int_part[4:0]);
    wire [EXP_WIDTH-1:0] exp_val   = (exp_shift == {EXP_WIDTH{1'b0}}) ? 16'd1 : exp_shift;

    wire [47:0] prod_num   = {32'd0, exp_mem[acc_cnt[POS_W-1:0]]} * 48'd32767;
    wire [31:0] safe_sigma = (sigma == 32'd0) ? 32'd1 : sigma;
    wire [47:0] div_res    = prod_num / {16'd0, safe_sigma};
    /* verilator lint_on UNUSEDSIGNAL */

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            busy          <= 1'b0;
            done          <= 1'b0;
            score_addr    <= 16'd0;
            exp2_lut_addr <= 8'd0;
            prob_we       <= 1'b0;
            prob_addr     <= 16'd0;
            prob_data     <= {PROB_WIDTH{1'b0}};
            max_score     <= -32'sh7fffffff;
            sigma         <= 32'd0;
            rd_cnt        <= 16'd0;
            acc_cnt       <= 16'd0;
            read_pipe     <= 2'b00;
            cur_int_part  <= 32'd0;
        end else begin
            prob_we <= 1'b0;
            done    <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy       <= 1'b1;
                        max_score  <= -32'sh7fffffff;
                        rd_cnt     <= 16'd0;
                        acc_cnt    <= 16'd0;
                        read_pipe  <= 2'b00;
                        state      <= S_PASS1_READ;
                    end
                end

                // ==============================================================
                // Pass 1: Find Max Score
                // ==============================================================
                S_PASS1_READ: begin
                    if (rd_cnt < seq_len) begin
                        score_addr <= rd_cnt;
                        rd_cnt     <= rd_cnt + 16'd1;
                        read_pipe  <= {read_pipe[0], 1'b1};
                    end else begin
                        read_pipe  <= {read_pipe[0], 1'b0};
                    end

                    if (read_pipe[1]) begin
                        if (acc_cnt == 16'd0 || score_data > max_score) begin
                            max_score <= score_data;
                        end
                        acc_cnt <= acc_cnt + 16'd1;
                        if (acc_cnt + 16'd1 == seq_len) begin
                            state <= S_PASS2_INIT;
                        end
                    end
                end

                // ==============================================================
                // Pass 2: Base-2 Exp & Sum Accumulation
                // ==============================================================
                S_PASS2_INIT: begin
                    acc_cnt <= 16'd0;
                    sigma   <= 32'd0;
                    state   <= S_PASS2_REQ_SC;
                end

                S_PASS2_REQ_SC: begin
                    score_addr <= acc_cnt;
                    state      <= S_PASS2_WAIT_SC;
                end

                S_PASS2_WAIT_SC: begin
                    // score_mem is registering score_addr
                    state <= S_PASS2_REQ_LUT;
                end

                S_PASS2_REQ_LUT: begin
                    // score_data is now available
                    cur_int_part <= (max_score - score_data) >> 8;
                    /* verilator lint_off WIDTHTRUNC */
                    exp2_lut_addr <= (max_score - score_data);
                    /* verilator lint_on WIDTHTRUNC */
                    state <= S_PASS2_WAIT_LUT;
                end

                S_PASS2_WAIT_LUT: begin
                    // exp2_mem is registering exp2_lut_addr
                    state <= S_PASS2_ACC;
                end

                S_PASS2_ACC: begin
                    // exp2_lut_data is now available
                    exp_mem[acc_cnt[POS_W-1:0]] <= exp_val;
                    sigma                       <= sigma + {{14{1'b0}}, exp_val};
                    acc_cnt               <= acc_cnt + 16'd1;

                    if (acc_cnt + 16'd1 < seq_len) begin
                        state <= S_PASS2_REQ_SC;
                    end else begin
                        acc_cnt <= 16'd0;
                        state   <= S_PASS3_NORM;
                    end
                end

                // ==============================================================
                // Pass 3: Normalize to Probabilities
                // ==============================================================
                S_PASS3_NORM: begin
                    prob_data <= div_res[PROB_WIDTH-1:0];
                    prob_addr <= acc_cnt;
                    prob_we   <= 1'b1;

                    acc_cnt <= acc_cnt + 16'd1;
                    if (acc_cnt + 16'd1 == seq_len) begin
                        state <= S_DONE;
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
/* verilator lint_on WIDTHEXPAND */
/* verilator lint_on WIDTHTRUNC */
