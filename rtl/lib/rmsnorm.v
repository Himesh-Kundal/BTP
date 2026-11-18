// rmsnorm.v - Parameterized RMSNorm Unit
// Contract references: docs/part2/02_architecture_and_contracts.md §3 (N2-6), §6

`timescale 1ns / 1ps

/* verilator lint_off BLKSEQ */
module rmsnorm #(
    parameter DATA_WIDTH  = 24,
    parameter ACC_WIDTH   = 48,
    parameter GAMMA_WIDTH = 16,
    parameter RSQRT_WIDTH = 16,
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

    // Source buffer read interface (1-cycle synchronous)
    output reg  [15:0]                   src_addr,
    input  wire signed [DATA_WIDTH-1:0]  src_data,

    // Gamma ROM read interface
    output reg  [15:0]                   gamma_addr,
    input  wire signed [GAMMA_WIDTH-1:0] gamma_data,

    // Rsqrt LUT ROM interface
    output reg  [7:0]                    rsqrt_lut_addr,
    input  wire [RSQRT_WIDTH-1:0]        rsqrt_lut_data,

    // Destination buffer write interface
    output reg                           dst_we,
    output reg  [15:0]                   dst_addr,
    output reg  signed [OUT_WIDTH-1:0]   dst_data
);

    `include "numerics.vh"

    // Leading zeros counter on 64 bits
    function [6:0] lzc64;
        input [63:0] val;
        integer k;
        reg found;
        begin
            lzc64 = 7'd64;
            found = 1'b0;
            for (k = 63; k >= 0; k = k - 1) begin
                if (!found && val[k]) begin
                    lzc64 = 7'd63 - k[6:0];
                    found = 1'b1;
                end
            end
        end
    endfunction

    // FSM States
    localparam S_IDLE          = 3'd0;
    localparam S_SUM_SQ        = 3'd1;
    localparam S_CALC_MEAN     = 3'd2;
    localparam S_REQ_RSQRT     = 3'd3;
    localparam S_WAIT_RSQRT    = 3'd4;
    localparam S_CAPTURE_RSQRT = 3'd5;
    localparam S_NORM_STREAM   = 3'd6;
    localparam S_DONE          = 3'd7;

    reg [2:0] state;

    reg signed [ACC_WIDTH-1:0] ss;
    reg [63:0]                 mean;
    reg [RSQRT_WIDTH-1:0]      rsqrt_val;

    reg [15:0] rd_cnt;
    reg [15:0] acc_cnt;
    reg [1:0]  read_pipe;

    // Normalization pipeline registers
    reg [15:0] out_cnt;
    reg        stg1_valid;
    reg signed [63:0]            stg1_scaled;
    reg signed [GAMMA_WIDTH-1:0] stg1_gamma;
    reg [15:0] stg1_idx;

    // Intermediate calculations
    /* verilator lint_off UNUSEDSIGNAL */
    wire [6:0]                  calc_lzc     = lzc64(mean);
    wire [6:0]                  calc_sh      = (7'd64 - calc_lzc) & 7'h7E;
    wire [6:0]                  calc_sh_amt  = (calc_sh > 7'd8) ? (calc_sh - 7'd8) : 7'd0;
    wire [63:0]                 shifted_mean = mean >> calc_sh_amt;
    wire [7:0]                  mantissa_8b  = shifted_mean[7:0];

    reg [5:0] stg1_s;

    wire signed [63:0]          stg2_acc     = stg1_scaled * stg1_gamma;
    wire signed [63:0]          stg2_scaled  = rshift_rnd64(stg2_acc * {{32{M[31]}}, M}, stg1_s);
    wire signed [63:0]          stg2_sat64   = sat_generic(stg2_scaled,
                                                           {{(64-OUT_WIDTH){OUT_MIN[OUT_WIDTH-1]}}, OUT_MIN},
                                                           {{(64-OUT_WIDTH){OUT_MAX[OUT_WIDTH-1]}}, OUT_MAX});
    wire signed [OUT_WIDTH-1:0] stg2_sat     = stg2_sat64[OUT_WIDTH-1:0];
    /* verilator lint_on UNUSEDSIGNAL */

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            busy           <= 1'b0;
            done           <= 1'b0;
            src_addr       <= 16'd0;
            gamma_addr     <= 16'd0;
            rsqrt_lut_addr <= 8'd0;
            dst_we         <= 1'b0;
            dst_addr       <= 16'd0;
            dst_data       <= {OUT_WIDTH{1'b0}};
            ss             <= {ACC_WIDTH{1'b0}};
            mean           <= 64'd0;
            rsqrt_val      <= {RSQRT_WIDTH{1'b0}};
            rd_cnt         <= 16'd0;
            acc_cnt        <= 16'd0;
            read_pipe      <= 2'b00;
            out_cnt        <= 16'd0;
            stg1_valid     <= 1'b0;
            stg1_scaled    <= 64'sd0;
            stg1_gamma     <= {GAMMA_WIDTH{1'b0}};
            stg1_s         <= 6'd0;
            stg1_idx       <= 16'd0;
        end else begin
            dst_we <= 1'b0;
            done   <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy       <= 1'b1;
                        ss         <= {ACC_WIDTH{1'b0}};
                        rd_cnt     <= 16'd0;
                        acc_cnt    <= 16'd0;
                        read_pipe  <= 2'b00;
                        src_addr   <= 16'd0;
                        state      <= S_SUM_SQ;
                    end
                end

                S_SUM_SQ: begin
                    if (rd_cnt < dim) begin
                        src_addr  <= rd_cnt;
                        rd_cnt    <= rd_cnt + 16'd1;
                        read_pipe <= {read_pipe[0], 1'b1};
                    end else begin
                        read_pipe <= {read_pipe[0], 1'b0};
                    end

                    if (read_pipe[1]) begin
                        ss      <= ss + (src_data * src_data);
                        acc_cnt <= acc_cnt + 16'd1;
                        if (acc_cnt + 16'd1 == dim) begin
                            state <= S_CALC_MEAN;
                        end
                    end
                end

                S_CALC_MEAN: begin
                    if (dim > 16'd0) begin
                        mean <= {{16{ss[ACC_WIDTH-1]}}, ss} / {{48{1'b0}}, dim};
                    end else begin
                        mean <= 64'd0;
                    end
                    state <= S_REQ_RSQRT;
                end

                S_REQ_RSQRT: begin
                    if (mean == 64'd0) begin
                        rsqrt_val <= {RSQRT_WIDTH{1'b0}};
                        rd_cnt    <= 16'd0;
                        out_cnt   <= 16'd0;
                        read_pipe <= 2'b00;
                        state     <= S_NORM_STREAM;
                    end else begin
                        rsqrt_lut_addr <= mantissa_8b;
                        state          <= S_WAIT_RSQRT;
                    end
                end

                S_WAIT_RSQRT: begin
                    state <= S_CAPTURE_RSQRT;
                end

                S_CAPTURE_RSQRT: begin
                    rsqrt_val <= rsqrt_lut_data;
                    rd_cnt    <= 16'd0;
                    out_cnt   <= 16'd0;
                    read_pipe <= 2'b00;
                    state     <= S_NORM_STREAM;
                end

                S_NORM_STREAM: begin
                    // Stage 0: Address drive
                    if (rd_cnt < dim) begin
                        src_addr   <= rd_cnt;
                        gamma_addr <= rd_cnt;
                        rd_cnt     <= rd_cnt + 16'd1;
                        read_pipe  <= {read_pipe[0], 1'b1};
                    end else begin
                        read_pipe  <= {read_pipe[0], 1'b0};
                    end

                    // Stage 1: Capture data and multiply by rsqrt
                    if (read_pipe[1]) begin
                        stg1_scaled <= rshift_rnd64(src_data * $signed({1'b0, rsqrt_val}), 6'd16);
                        stg1_gamma  <= gamma_data;
                        stg1_s      <= s;
                        stg1_idx    <= out_cnt;
                        stg1_valid  <= 1'b1;
                        out_cnt     <= out_cnt + 16'd1;
                    end else begin
                        stg1_valid  <= 1'b0;
                    end

                    // Stage 2: Scale by gamma and requantize
                    if (stg1_valid) begin
                        dst_data <= stg2_sat;
                        dst_addr <= stg1_idx;
                        dst_we   <= 1'b1;
                        if (stg1_idx + 16'd1 == dim) begin
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
