// linear_engine.v - Parameterized RxC Matrix Multiplication Engine with ROM Streaming
// Contract references: docs/part2/02_architecture_and_contracts.md §3 (N2-3, N2-7), §6

`timescale 1ns / 1ps

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off BLKSEQ */
module linear_engine #(
    parameter R               = 8,
    parameter C               = 16,
    parameter ACT_WIDTH       = 8,
    parameter WEIGHT_WIDTH    = 8,
    parameter ACC_WIDTH       = 32,
    parameter M_WIDTH         = 32,
    parameter SHIFT_WIDTH     = 6,
    parameter BUF_ADDR_WIDTH  = 16,
    parameter ROM_ADDR_WIDTH  = 32
)(
    input  wire                               clk,
    input  wire                               rst_n,

    // Control interface
    input  wire                               start,
    output reg                                busy,
    output reg                                done,
    input  wire [15:0]                        matrix_n,
    input  wire [15:0]                        matrix_k,
    input  wire [ROM_ADDR_WIDTH-1:0]          weight_base_addr,
    input  wire [ROM_ADDR_WIDTH-1:0]          param_base_addr,
    input  wire                               out_is_res,

    // Input activation buffer interface (1-cycle synchronous read)
    output reg  [BUF_ADDR_WIDTH-1:0]          act_rd_addr,
    input  wire signed [C*ACT_WIDTH-1:0]      act_rd_data,

    // Weight ROM interface (1-cycle synchronous read)
    output reg  [ROM_ADDR_WIDTH-1:0]          weight_rd_addr,
    input  wire signed [C*WEIGHT_WIDTH-1:0]   weight_rd_data,

    // Requant parameter ROM interface (1-cycle synchronous read)
    output reg  [ROM_ADDR_WIDTH-1:0]          param_rd_addr,
    input  wire signed [M_WIDTH-1:0]          param_m_data,
    input  wire [SHIFT_WIDTH-1:0]             param_s_data,

    // Output buffer write interface
    output reg                                out_wr_en,
    output reg  [BUF_ADDR_WIDTH-1:0]          out_wr_addr,
    output reg  signed [23:0]                 out_wr_data
);

    `include "numerics.vh"

    // FSM States
    localparam S_IDLE       = 3'd0;
    localparam S_LOAD_ACT   = 3'd1;
    localparam S_ACT_WAIT   = 3'd2;
    localparam S_PIPE_WAIT  = 3'd3;
    localparam S_COMPUTE    = 3'd4;
    localparam S_DRAIN_READ = 3'd5;
    localparam S_DRAIN_PIPE = 3'd6;
    localparam S_DRAIN_OUT  = 3'd7;

    reg [2:0] state;

    // Loop iteration counters
    reg [15:0] tr;          // Tile row index: 0 .. num_tiles_r - 1
    reg [15:0] tc;          // Tile col index: 0 .. num_tiles_c - 1
    reg [15:0] r_cnt;       // Inner row counter: 0 .. R - 1
    reg [15:0] dr_cnt;      // Drain counter: 0 .. R - 1

    reg [15:0] num_tiles_r;
    reg [15:0] num_tiles_c;

    reg [ROM_ADDR_WIDTH-1:0] weight_ptr;

    // Internal input activation capture register
    reg signed [C*ACT_WIDTH-1:0] x_reg;

    // Accumulator registers for R rows
    reg signed [ACC_WIDTH-1:0] row_acc [0:R-1];

    // Pipeline delay registers for compute
    /* verilator lint_off UNUSEDSIGNAL */
    reg [15:0] comp_r_pipe;
    /* verilator lint_on UNUSEDSIGNAL */
    reg        comp_valid_pipe;

    // Pipeline registers for drain
    reg [15:0]                 drain_idx_pipe1;
    reg signed [ACC_WIDTH-1:0] drain_acc_pipe1;
    reg                        drain_valid_pipe1;

    // Multiplier array & Adder tree
    wire signed [15:0] prod [0:C-1];
    reg  signed [ACC_WIDTH-1:0] dot_sum;

    genvar gi;
    generate
        for (gi = 0; gi < C; gi = gi + 1) begin : g_mult
            wire signed [ACT_WIDTH-1:0] x_lane = x_reg[gi*ACT_WIDTH +: ACT_WIDTH];
            wire signed [WEIGHT_WIDTH-1:0] w_lane = weight_rd_data[gi*WEIGHT_WIDTH +: WEIGHT_WIDTH];
            wire is_valid_k = ((tc * C + gi) < matrix_k);
            wire signed [ACT_WIDTH-1:0] eff_x = is_valid_k ? x_lane : {ACT_WIDTH{1'b0}};
            assign prod[gi] = eff_x * w_lane;
        end
    endgenerate

    integer ci;
    always @(*) begin
        dot_sum = 0;
        for (ci = 0; ci < C; ci = ci + 1) begin
            dot_sum = dot_sum + {{16{prod[ci][15]}}, prod[ci]};
        end
    end

    // Requant calculation for drain
    wire signed [63:0] drain_prod = {{32{drain_acc_pipe1[ACC_WIDTH-1]}}, drain_acc_pipe1} *
                                    {{32{param_m_data[M_WIDTH-1]}}, param_m_data};
    wire signed [63:0] drain_scaled = rshift_rnd64(drain_prod, param_s_data);
    wire signed [7:0]  drain_sat_int8 = sat_int8(drain_scaled);
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [63:0] drain_sat_res64 = sat_generic(drain_scaled, -64'sd8388608, 64'sd8388607);
    /* verilator lint_on UNUSEDSIGNAL */
    wire signed [23:0] drain_sat_res  = drain_sat_res64[23:0];

    integer ri;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state             <= S_IDLE;
            busy              <= 1'b0;
            done              <= 1'b0;
            out_wr_en         <= 1'b0;
            out_wr_addr       <= {BUF_ADDR_WIDTH{1'b0}};
            out_wr_data       <= 24'sd0;
            act_rd_addr       <= {BUF_ADDR_WIDTH{1'b0}};
            weight_rd_addr    <= {ROM_ADDR_WIDTH{1'b0}};
            param_rd_addr     <= {ROM_ADDR_WIDTH{1'b0}};
            tr                <= 16'd0;
            tc                <= 16'd0;
            r_cnt             <= 16'd0;
            dr_cnt            <= 16'd0;
            num_tiles_r       <= 16'd0;
            num_tiles_c       <= 16'd0;
            weight_ptr        <= {ROM_ADDR_WIDTH{1'b0}};
            x_reg             <= {(C*ACT_WIDTH){1'b0}};
            comp_r_pipe       <= 16'd0;
            comp_valid_pipe   <= 1'b0;
            drain_idx_pipe1   <= 16'd0;
            drain_acc_pipe1   <= 32'sd0;
            drain_valid_pipe1 <= 1'b0;
            for (ri = 0; ri < R; ri = ri + 1) begin
                row_acc[ri] <= 32'sd0;
            end
        end else begin
            out_wr_en <= 1'b0;
            done      <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy         <= 1'b1;
                        num_tiles_r  <= (matrix_n + R - 1) / R;
                        num_tiles_c  <= (matrix_k + C - 1) / C;
                        tr           <= 16'd0;
                        tc           <= 16'd0;
                        weight_ptr   <= weight_base_addr;
                        state        <= S_LOAD_ACT;
                        for (ri = 0; ri < R; ri = ri + 1) begin
                            row_acc[ri] <= 32'sd0;
                        end
                    end
                end

                S_LOAD_ACT: begin
                    // Drive activation read address for current tile column
                    act_rd_addr <= tc;
                    state       <= S_ACT_WAIT;
                end

                S_ACT_WAIT: begin
                    state <= S_PIPE_WAIT;
                end

                S_PIPE_WAIT: begin
                    // 1 cycle latency: capture act_rd_data into x_reg
                    x_reg          <= act_rd_data;
                    r_cnt          <= 16'd0;
                    // Drive address of first row of weights
                    weight_rd_addr <= weight_ptr;
                    weight_ptr     <= weight_ptr + 1'b1;
                    state          <= S_COMPUTE;
                end

                S_COMPUTE: begin
                    // At each cycle, weight_rd_data is valid for row comp_r_pipe
                    if (comp_valid_pipe) begin
                        if (tc == 16'd0) begin
                            row_acc[comp_r_pipe] <= dot_sum;
                        end else begin
                            row_acc[comp_r_pipe] <= row_acc[comp_r_pipe] + dot_sum;
                        end
                    end

                    comp_r_pipe     <= r_cnt;
                    comp_valid_pipe <= 1'b1;

                    if (r_cnt + 1'b1 < R) begin
                        r_cnt          <= r_cnt + 1'b1;
                        weight_rd_addr <= weight_ptr;
                        weight_ptr     <= weight_ptr + 1'b1;
                    end else begin
                        // Last row address already issued
                        r_cnt <= r_cnt + 1'b1;
                    end

                    // Once the R-th row dot product is accumulated (when r_cnt == R)
                    if (r_cnt == R) begin
                        comp_valid_pipe <= 1'b0;
                        if (tc + 1'b1 < num_tiles_c) begin
                            tc    <= tc + 1'b1;
                            state <= S_LOAD_ACT;
                        end else begin
                            // All K elements accumulated for this tile row
                            dr_cnt <= 16'd0;
                            state  <= S_DRAIN_READ;
                        end
                    end
                end

                S_DRAIN_READ: begin
                    if (dr_cnt < R) begin
                        if (tr * R + dr_cnt < matrix_n) begin
                            param_rd_addr   <= param_base_addr + (tr * R + dr_cnt);
                            drain_idx_pipe1 <= tr * R + dr_cnt;
                            drain_acc_pipe1 <= row_acc[dr_cnt];
                            drain_valid_pipe1 <= 1'b1;
                        end else begin
                            drain_valid_pipe1 <= 1'b0;
                        end
                        dr_cnt <= dr_cnt + 1'b1;
                        state  <= S_DRAIN_PIPE;
                    end else begin
                        state  <= S_DRAIN_OUT;
                    end
                end

                S_DRAIN_PIPE: begin
                    // Memory returns param_m_data, param_s_data on this cycle
                    state <= S_DRAIN_OUT;
                end

                S_DRAIN_OUT: begin
                    if (drain_valid_pipe1) begin
                        out_wr_en   <= 1'b1;
                        out_wr_addr <= drain_idx_pipe1;
                        out_wr_data <= out_is_res ? drain_sat_res : {{16{drain_sat_int8[7]}}, drain_sat_int8};
                    end

                    if (dr_cnt < R) begin
                        state <= S_DRAIN_READ;
                    end else begin
                        // Finished draining this tile row
                        if (tr + 1'b1 < num_tiles_r) begin
                            tr    <= tr + 1'b1;
                            tc    <= 16'd0;
                            state <= S_LOAD_ACT;
                            for (ri = 0; ri < R; ri = ri + 1) begin
                                row_acc[ri] <= 32'sd0;
                            end
                        end else begin
                            busy  <= 1'b0;
                            done  <= 1'b1;
                            state <= S_IDLE;
                        end
                    end
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
/* verilator lint_on BLKSEQ */
