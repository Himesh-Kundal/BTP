// argmax.v - Greedy ArgMax Unit with Per-Row Requantization Scale Normalization
// Contract references: docs/part2/02_architecture_and_contracts.md §3 (N2-12, N2-13), §6
//
// 1. Scans vocabulary entries 0 .. vocab_size-1
// 2. Normalizes raw accumulators: scaled_logit = rshift_rnd(raw_acc * M, s)
// 3. Resolves ties to the lowest token ID (Rule N2-13)

`timescale 1ns / 1ps

/* verilator lint_off BLKSEQ */
module argmax #(
    parameter ACC_WIDTH   = 32,
    parameter M_WIDTH     = 32,
    parameter SHIFT_WIDTH = 6,
    parameter TOK_WIDTH   = 16
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,
    input  wire [TOK_WIDTH-1:0]          vocab_size,
    output reg                           busy,
    output reg                           done,
    output reg  [TOK_WIDTH-1:0]          best_token_id,

    // Raw logits accumulator read interface (1-cycle synchronous)
    output reg  [TOK_WIDTH-1:0]          logit_addr,
    input  wire signed [ACC_WIDTH-1:0]   logit_data,

    // Per-row scale read interface (1-cycle synchronous)
    output reg  [TOK_WIDTH-1:0]          scale_addr,
    input  wire signed [M_WIDTH-1:0]     M_data,
    input  wire [SHIFT_WIDTH-1:0]        s_data
);

    `include "numerics.vh"

    // FSM States
    localparam S_IDLE   = 2'd0;
    localparam S_STREAM = 2'd1;
    localparam S_DONE   = 2'd2;

    reg [1:0] state;

    reg [TOK_WIDTH-1:0] rd_cnt;
    reg [TOK_WIDTH-1:0] tok_cnt;
    reg [1:0]           read_pipe;

    reg signed [63:0]   best_val;
    reg [TOK_WIDTH-1:0] best_id;

    // Pipeline registers
    reg signed [63:0]   stg1_scaled;
    reg [TOK_WIDTH-1:0] stg1_id;
    reg                 stg1_valid;

    // Intermediate calculations
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [63:0] prod   = {{32{logit_data[ACC_WIDTH-1]}}, logit_data} *
                                {{32{M_data[M_WIDTH-1]}}, M_data};
    wire signed [63:0] scaled = rshift_rnd64(prod, s_data);
    /* verilator lint_on UNUSEDSIGNAL */

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            busy          <= 1'b0;
            done          <= 1'b0;
            best_token_id <= {TOK_WIDTH{1'b0}};
            logit_addr    <= {TOK_WIDTH{1'b0}};
            scale_addr    <= {TOK_WIDTH{1'b0}};
            rd_cnt        <= {TOK_WIDTH{1'b0}};
            tok_cnt       <= {TOK_WIDTH{1'b0}};
            read_pipe     <= 2'b00;
            best_val      <= -64'sh7fffffffffffffff;
            best_id       <= {TOK_WIDTH{1'b0}};
            stg1_scaled   <= 64'sd0;
            stg1_id       <= {TOK_WIDTH{1'b0}};
            stg1_valid    <= 1'b0;
        end else begin
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy       <= 1'b1;
                        rd_cnt     <= {TOK_WIDTH{1'b0}};
                        tok_cnt    <= {TOK_WIDTH{1'b0}};
                        read_pipe  <= 2'b00;
                        best_val   <= -64'sh7fffffffffffffff;
                        best_id    <= {TOK_WIDTH{1'b0}};
                        stg1_valid <= 1'b0;
                        state      <= S_STREAM;
                    end
                end

                S_STREAM: begin
                    // Stage 0: Address drive
                    if (rd_cnt < vocab_size) begin
                        logit_addr <= rd_cnt;
                        scale_addr <= rd_cnt;
                        rd_cnt     <= rd_cnt + 1'b1;
                        read_pipe  <= {read_pipe[0], 1'b1};
                    end else begin
                        read_pipe  <= {read_pipe[0], 1'b0};
                    end

                    // Stage 1: Capture data from memories, compute scaled logit
                    if (read_pipe[1]) begin
                        stg1_scaled <= scaled;
                        stg1_id     <= tok_cnt;
                        stg1_valid  <= 1'b1;
                        tok_cnt     <= tok_cnt + 1'b1;
                    end else begin
                        stg1_valid  <= 1'b0;
                    end

                    // Stage 2: Compare & update running argmax
                    if (stg1_valid) begin
                        if (stg1_id == {TOK_WIDTH{1'b0}} || stg1_scaled > best_val) begin
                            best_val <= stg1_scaled;
                            best_id  <= stg1_id;
                        end

                        if (stg1_id + 1'b1 == vocab_size) begin
                            best_token_id <= (stg1_id == {TOK_WIDTH{1'b0}} || stg1_scaled > best_val) ?
                                             stg1_id : best_id;
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
