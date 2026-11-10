// swiglu.v - Parameterized SwiGLU Activation Unit
// Contract references: docs/part2/02_architecture_and_contracts.md §3 (N2-11), §6
//
// Computes elementwise SwiGLU:
// 1. silu_val = silu_lut[gate + offset]
// 2. prod = silu_val * up
// 3. out = sat(rshift_rnd(prod * M, s), OUT_MIN, OUT_MAX)

`timescale 1ns / 1ps

/* verilator lint_off BLKSEQ */
module swiglu #(
    parameter DATA_WIDTH  = 8,
    parameter LUT_WIDTH   = 8,
    parameter LUT_SIZE    = 256,
    parameter OUT_WIDTH   = 8,
    parameter signed [OUT_WIDTH-1:0] OUT_MIN = -8'sd127,
    parameter signed [OUT_WIDTH-1:0] OUT_MAX = 8'sd127
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,
    input  wire [15:0]                   dim,
    input  wire signed [31:0]            M,
    input  wire [5:0]                    s,
    output reg                           busy,
    output reg                           done,

    // Gate buffer read interface (1-cycle synchronous SRAM)
    output reg  [15:0]                   gate_addr,
    input  wire signed [DATA_WIDTH-1:0]  gate_data,

    // Up buffer read interface (1-cycle synchronous SRAM)
    output reg  [15:0]                   up_addr,
    input  wire signed [DATA_WIDTH-1:0]  up_data,

    // SiLU LUT ROM read interface
    output reg  [8:0]                    silu_lut_addr,
    input  wire signed [LUT_WIDTH-1:0]   silu_lut_data,

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

    reg [15:0] rd_cnt;
    reg [15:0] stg1_cnt;
    reg [1:0]  read_pipe;

    // Pipeline registers
    reg signed [DATA_WIDTH-1:0]  stg1_up;
    reg [15:0]                   stg1_idx;
    reg                          stg1_valid;

    reg signed [DATA_WIDTH-1:0]  stg2_up;
    reg [15:0]                   stg2_idx;
    reg                          stg2_valid;

    // Intermediate calculations
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [63:0] prod        = {{56{silu_lut_data[LUT_WIDTH-1]}}, silu_lut_data} *
                                     {{56{stg2_up[DATA_WIDTH-1]}}, stg2_up};
    wire signed [63:0] scaled_prod = rshift_rnd64(prod * {{32{M[31]}}, M}, s);
    wire signed [63:0] sat_out     = sat_generic(scaled_prod,
                                                 {{(64-OUT_WIDTH){OUT_MIN[OUT_WIDTH-1]}}, OUT_MIN},
                                                 {{(64-OUT_WIDTH){OUT_MAX[OUT_WIDTH-1]}}, OUT_MAX});
    /* verilator lint_on UNUSEDSIGNAL */

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            busy          <= 1'b0;
            done          <= 1'b0;
            gate_addr     <= 16'd0;
            up_addr       <= 16'd0;
            silu_lut_addr <= 9'd0;
            dst_we        <= 1'b0;
            dst_addr      <= 16'd0;
            dst_data      <= {OUT_WIDTH{1'b0}};
            rd_cnt        <= 16'd0;
            stg1_cnt      <= 16'd0;
            read_pipe     <= 2'b00;
            stg1_up       <= {DATA_WIDTH{1'b0}};
            stg1_idx      <= 16'd0;
            stg1_valid    <= 1'b0;
            stg2_up       <= {DATA_WIDTH{1'b0}};
            stg2_idx      <= 16'd0;
            stg2_valid    <= 1'b0;
        end else begin
            dst_we <= 1'b0;
            done   <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy       <= 1'b1;
                        rd_cnt     <= 16'd0;
                        stg1_cnt   <= 16'd0;
                        read_pipe  <= 2'b00;
                        stg1_valid <= 1'b0;
                        stg2_valid <= 1'b0;
                        state      <= S_STREAM;
                    end
                end

                S_STREAM: begin
                    // Stage 0: Address drive to gate and up memories
                    if (rd_cnt < dim) begin
                        gate_addr <= rd_cnt;
                        up_addr   <= rd_cnt;
                        rd_cnt    <= rd_cnt + 16'd1;
                        read_pipe <= {read_pipe[0], 1'b1};
                    end else begin
                        read_pipe <= {read_pipe[0], 1'b0};
                    end

                    // Stage 1: Capture read data and drive SiLU LUT address
                    if (read_pipe[1]) begin
                        stg1_up    <= up_data;
                        stg1_idx   <= stg1_cnt;
                        stg1_valid <= 1'b1;
                        stg1_cnt   <= stg1_cnt + 16'd1;

                        if (LUT_SIZE == 512) begin
                            silu_lut_addr <= $unsigned($signed({{1{gate_data[DATA_WIDTH-1]}}, gate_data}) + 9'sd256);
                        end else begin
                            silu_lut_addr <= {1'b0, $unsigned($signed(gate_data) + 8'sd128)};
                        end
                    end else begin
                        stg1_valid <= 1'b0;
                    end

                    // Stage 2: SiLU ROM registers address, pipeline forward
                    stg2_valid <= stg1_valid;
                    stg2_idx   <= stg1_idx;
                    stg2_up    <= stg1_up;

                    // Stage 3: SiLU LUT data arrives, multiply and requantize
                    if (stg2_valid) begin
                        dst_data <= sat_out[OUT_WIDTH-1:0];
                        dst_addr <= stg2_idx;
                        dst_we   <= 1'b1;

                        if (stg2_idx + 16'd1 == dim) begin
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
