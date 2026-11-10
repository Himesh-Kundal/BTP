// residual_add.v - Saturating Residual Addition Unit
// Contract references: docs/part2/02_architecture_and_contracts.md §3 (N2-4), §6
//
// Computes elementwise saturating addition:
// out_i = sat(a_i + b_i, MIN_VAL, MAX_VAL)

`timescale 1ns / 1ps

/* verilator lint_off BLKSEQ */
module residual_add #(
    parameter WIDTH = 24,
    parameter signed [WIDTH-1:0] MIN_VAL = -24'sd8388608,
    parameter signed [WIDTH-1:0] MAX_VAL = 24'sd8388607
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         start,
    input  wire [15:0]                  dim,
    output reg                          busy,
    output reg                          done,

    // Source A buffer interface (1-cycle synchronous SRAM)
    output reg  [15:0]                  a_addr,
    input  wire signed [WIDTH-1:0]      a_data,

    // Source B buffer interface (1-cycle synchronous SRAM)
    output reg  [15:0]                  b_addr,
    input  wire signed [WIDTH-1:0]      b_data,

    // Destination buffer write interface
    output reg                          dst_we,
    output reg  [15:0]                  dst_addr,
    output reg  signed [WIDTH-1:0]      dst_data
);

    // FSM States
    localparam S_IDLE   = 2'd0;
    localparam S_STREAM = 2'd1;
    localparam S_DONE   = 2'd2;

    reg [1:0] state;

    reg [15:0] rd_cnt;
    reg [15:0] out_cnt;
    reg [1:0]  read_pipe;

    // Saturating adder
    wire signed [WIDTH:0] sum = {{1{a_data[WIDTH-1]}}, a_data} + {{1{b_data[WIDTH-1]}}, b_data};
    wire signed [WIDTH-1:0] sat_sum = (sum > $signed({{1{MAX_VAL[WIDTH-1]}}, MAX_VAL})) ? MAX_VAL :
                                      ((sum < $signed({{1{MIN_VAL[WIDTH-1]}}, MIN_VAL})) ? MIN_VAL :
                                      sum[WIDTH-1:0]);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            busy      <= 1'b0;
            done      <= 1'b0;
            a_addr    <= 16'd0;
            b_addr    <= 16'd0;
            dst_we    <= 1'b0;
            dst_addr  <= 16'd0;
            dst_data  <= {WIDTH{1'b0}};
            rd_cnt    <= 16'd0;
            out_cnt   <= 16'd0;
            read_pipe <= 2'b00;
        end else begin
            dst_we <= 1'b0;
            done   <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy      <= 1'b1;
                        rd_cnt    <= 16'd0;
                        out_cnt   <= 16'd0;
                        read_pipe <= 2'b00;
                        state     <= S_STREAM;
                    end
                end

                S_STREAM: begin
                    // Stage 0: Drive addresses to memories
                    if (rd_cnt < dim) begin
                        a_addr    <= rd_cnt;
                        b_addr    <= rd_cnt;
                        rd_cnt    <= rd_cnt + 16'd1;
                        read_pipe <= {read_pipe[0], 1'b1};
                    end else begin
                        read_pipe <= {read_pipe[0], 1'b0};
                    end

                    // Stage 1: Capture data from memories and write
                    if (read_pipe[1]) begin
                        dst_data <= sat_sum;
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
