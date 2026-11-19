// attention.v - Parameterized GQA Attention Datapath with KV Cache & Softmax Integration
// Contract references: docs/part2/02_architecture_and_contracts.md §3 (N2-9, N2-10), §6

`timescale 1ns / 1ps

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off BLKSEQ */
module attention #(
    parameter NUM_LAYERS   = 30,
    parameter NUM_KV_HEADS = 3,
    parameter NUM_Q_HEADS  = 9,
    parameter HEAD_DIM     = 64,
    parameter MAX_CTX      = 256,
    parameter DATA_WIDTH   = 8,
    parameter SCORES_WIDTH = 32,
    parameter EXP_WIDTH    = 18,
    parameter PROB_WIDTH   = 16,
    parameter BUF_ADDR_W   = 16
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // Unit Control Protocol
    input  wire                          start,
    output reg                           busy,
    output reg                           done,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [4:0]                    layer_idx,
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire [15:0]                   pos,

    // Activation Buffer Interfaces (BUF_Q, BUF_K, BUF_V)
    output reg  [BUF_ADDR_W-1:0]         q_rd_addr,
    input  wire signed [DATA_WIDTH-1:0]  q_rd_data,

    output reg  [BUF_ADDR_W-1:0]         k_rd_addr,
    input  wire signed [DATA_WIDTH-1:0]  k_rd_data,

    output reg  [BUF_ADDR_W-1:0]         v_rd_addr,
    input  wire signed [DATA_WIDTH-1:0]  v_rd_data,

    // Attention Output Buffer Interface (BUF_ATTN_OUT)
    output reg                           attn_wr_en,
    output reg  [BUF_ADDR_W-1:0]         attn_wr_addr,
    output reg  signed [DATA_WIDTH-1:0]  attn_wr_data,

    // Exp2 LUT ROM Port for Softmax
    output wire [7:0]                    exp2_lut_addr,
    input  wire [EXP_WIDTH-1:0]          exp2_lut_data,

    // KV Cache Interface (Write & Read)
    output reg                           kv_wr_en,
    output reg  [LAYER_W-1:0]            kv_wr_layer,
    output reg                           kv_wr_is_v,
    output reg  [KV_HEAD_W-1:0]          kv_wr_kv_head,
    output reg  [POS_W-1:0]              kv_wr_pos,
    output reg  [DIM_W-1:0]              kv_wr_dim,
    output reg  signed [DATA_WIDTH-1:0]  kv_wr_data,

    output reg                           kv_rd_en,
    output reg  [LAYER_W-1:0]            kv_rd_layer,
    output reg                           kv_rd_is_v,
    output reg  [KV_HEAD_W-1:0]          kv_rd_kv_head,
    output reg  [POS_W-1:0]              kv_rd_pos,
    output reg  [DIM_W-1:0]              kv_rd_dim,
    input  wire signed [DATA_WIDTH-1:0]  kv_rd_data
);

    `include "numerics.vh"

    localparam GQA_GROUP = NUM_Q_HEADS / NUM_KV_HEADS;
    localparam LAYER_W   = (NUM_LAYERS > 1) ? $clog2(NUM_LAYERS) : 1;
    localparam KV_HEAD_W = (NUM_KV_HEADS > 1) ? $clog2(NUM_KV_HEADS) : 1;
    localparam Q_HEAD_W  = (NUM_Q_HEADS > 1) ? $clog2(NUM_Q_HEADS) : 1;
    localparam POS_W     = (MAX_CTX > 1) ? $clog2(MAX_CTX) : 1;
    localparam DIM_W     = (HEAD_DIM > 1) ? $clog2(HEAD_DIM) : 1;

    // FSM States
    localparam S_IDLE            = 4'd0;
    localparam S_KV_STORE_REQ    = 4'd1;
    localparam S_KV_STORE_PIPE   = 4'd2;
    localparam S_KV_STORE_WR_K   = 4'd3;
    localparam S_KV_STORE_WR_V   = 4'd4;
    localparam S_Q_HEAD_INIT     = 4'd5;
    localparam S_Q_LOAD_REQ      = 4'd6;
    localparam S_Q_LOAD_CAP      = 4'd7;
    localparam S_SCORE_INIT      = 4'd8;
    localparam S_SCORE_STREAM    = 4'd9;
    localparam S_SOFTMAX_WAIT    = 4'd10;
    localparam S_CTX_INIT        = 4'd11;
    localparam S_CTX_STREAM      = 4'd12;
    localparam S_DONE            = 4'd13;

    reg [3:0] state;

    // Counters
    reg [Q_HEAD_W-1:0]  cur_q_head;
    reg [KV_HEAD_W-1:0] cur_kv_head;
    reg [POS_W-1:0]     cur_pos_cnt;
    reg [DIM_W-1:0]     cur_dim_cnt;

    // Scratchpad Memories
    reg signed [DATA_WIDTH-1:0]   q_local [0:HEAD_DIM-1];
    reg signed [SCORES_WIDTH-1:0] scores_mem [0:MAX_CTX-1];
    reg [PROB_WIDTH-1:0]          prob_mem [0:MAX_CTX-1];

    // Score accumulation registers
    reg signed [31:0] dot_acc;
    reg signed [63:0] ctx_acc;

    // Pipelining registers
    reg [1:0]       read_pipe;
    reg [DIM_W-1:0] pipe_dim [0:1];
    reg [POS_W-1:0] pipe_pos [0:1];

    // Softmax unit wires
    reg         sm_start;
    /* verilator lint_off UNUSEDSIGNAL */
    wire        sm_busy;
    wire        sm_done;
    wire [15:0] sm_score_addr;
    wire [15:0] sm_prob_addr;
    /* verilator lint_on UNUSEDSIGNAL */
    wire        sm_prob_we;
    wire [PROB_WIDTH-1:0] sm_prob_data;
    reg signed [SCORES_WIDTH-1:0] sm_score_data_reg;

    always @(posedge clk) begin
        sm_score_data_reg <= scores_mem[sm_score_addr[POS_W-1:0]];
    end

    // Softmax instance
    softmax #(
        .SCORES_WIDTH(SCORES_WIDTH),
        .EXP_WIDTH(EXP_WIDTH),
        .PROB_WIDTH(PROB_WIDTH),
        .MAX_CTX(MAX_CTX)
    ) sm_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(sm_start),
        .seq_len(pos + 1'b1),
        .busy(sm_busy),
        .done(sm_done),
        .score_addr(sm_score_addr),
        .score_data(sm_score_data_reg),
        .exp2_lut_addr(exp2_lut_addr),
        .exp2_lut_data(exp2_lut_data),
        .prob_we(sm_prob_we),
        .prob_addr(sm_prob_addr),
        .prob_data(sm_prob_data)
    );

    // Save softmax output into prob_mem
    always @(posedge clk) begin
        if (sm_prob_we) begin
            prob_mem[sm_prob_addr[POS_W-1:0]] <= sm_prob_data;
        end
    end

    // GQA helper
    /* verilator lint_off UNUSEDSIGNAL */
    wire [31:0] div_val = cur_q_head / GQA_GROUP;
    wire signed [63:0] cur_ctx_tot = ctx_acc + ($signed({16'd0, prob_mem[pipe_pos[1]]}) * kv_rd_data);
    wire signed [63:0] cur_ctx_rnd = rshift_rnd64(cur_ctx_tot, 6'd15);
    /* verilator lint_on UNUSEDSIGNAL */
    wire [KV_HEAD_W-1:0] mapped_kv_head = div_val[KV_HEAD_W-1:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            busy           <= 1'b0;
            done           <= 1'b0;
            attn_wr_en     <= 1'b0;
            attn_wr_addr   <= {BUF_ADDR_W{1'b0}};
            attn_wr_data   <= {DATA_WIDTH{1'b0}};
            kv_wr_en       <= 1'b0;
            kv_wr_layer    <= {LAYER_W{1'b0}};
            kv_wr_is_v     <= 1'b0;
            kv_wr_kv_head  <= {KV_HEAD_W{1'b0}};
            kv_wr_pos      <= {POS_W{1'b0}};
            kv_wr_dim      <= {DIM_W{1'b0}};
            kv_wr_data     <= {DATA_WIDTH{1'b0}};
            kv_rd_en       <= 1'b0;
            kv_rd_layer    <= {LAYER_W{1'b0}};
            kv_rd_is_v     <= 1'b0;
            kv_rd_kv_head  <= {KV_HEAD_W{1'b0}};
            kv_rd_pos      <= {POS_W{1'b0}};
            kv_rd_dim      <= {DIM_W{1'b0}};
            q_rd_addr      <= {BUF_ADDR_W{1'b0}};
            k_rd_addr      <= {BUF_ADDR_W{1'b0}};
            v_rd_addr      <= {BUF_ADDR_W{1'b0}};
            cur_q_head     <= {Q_HEAD_W{1'b0}};
            cur_kv_head    <= {KV_HEAD_W{1'b0}};
            cur_pos_cnt    <= {POS_W{1'b0}};
            cur_dim_cnt    <= {DIM_W{1'b0}};
            dot_acc        <= 32'sd0;
            ctx_acc        <= 64'sd0;
            sm_start       <= 1'b0;
            read_pipe      <= 2'b00;
            pipe_dim[0]    <= {DIM_W{1'b0}};
            pipe_dim[1]    <= {DIM_W{1'b0}};
            pipe_pos[0]    <= {POS_W{1'b0}};
            pipe_pos[1]    <= {POS_W{1'b0}};
        end else begin
            attn_wr_en <= 1'b0;
            kv_wr_en   <= 1'b0;
            done       <= 1'b0;
            sm_start   <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy        <= 1'b1;
                        cur_kv_head <= {KV_HEAD_W{1'b0}};
                        cur_dim_cnt <= {DIM_W{1'b0}};
                        state       <= S_KV_STORE_REQ;
                    end
                end

                // -------------------------------------------------------------
                // 1. KV Store: Stream incoming K and V into kv_cache at pos
                // -------------------------------------------------------------
                S_KV_STORE_REQ: begin
                    k_rd_addr <= cur_kv_head * HEAD_DIM + cur_dim_cnt;
                    v_rd_addr <= cur_kv_head * HEAD_DIM + cur_dim_cnt;
                    state     <= S_KV_STORE_PIPE;
                end

                S_KV_STORE_PIPE: begin
                    state <= S_KV_STORE_WR_K;
                end

                S_KV_STORE_WR_K: begin
                    kv_wr_en      <= 1'b1;
                    kv_wr_layer   <= layer_idx[LAYER_W-1:0];
                    kv_wr_is_v    <= 1'b0;
                    kv_wr_kv_head <= cur_kv_head;
                    kv_wr_pos     <= pos[POS_W-1:0];
                    kv_wr_dim     <= cur_dim_cnt;
                    kv_wr_data    <= k_rd_data;
                    state         <= S_KV_STORE_WR_V;
                end

                S_KV_STORE_WR_V: begin
                    kv_wr_en      <= 1'b1;
                    kv_wr_layer   <= layer_idx[LAYER_W-1:0];
                    kv_wr_is_v    <= 1'b1;
                    kv_wr_kv_head <= cur_kv_head;
                    kv_wr_pos     <= pos[POS_W-1:0];
                    kv_wr_dim     <= cur_dim_cnt;
                    kv_wr_data    <= v_rd_data;

                    if (cur_dim_cnt + 1'b1 < HEAD_DIM) begin
                        cur_dim_cnt <= cur_dim_cnt + 1'b1;
                        state       <= S_KV_STORE_REQ;
                    end else begin
                        cur_dim_cnt <= {DIM_W{1'b0}};
                        if (cur_kv_head + 1'b1 < NUM_KV_HEADS) begin
                            cur_kv_head <= cur_kv_head + 1'b1;
                            state       <= S_KV_STORE_REQ;
                        end else begin
                            cur_q_head <= {Q_HEAD_W{1'b0}};
                            state      <= S_Q_HEAD_INIT;
                        end
                    end
                end

                // -------------------------------------------------------------
                // 2. Q Head Init & Load Q vector into local buffer
                // -------------------------------------------------------------
                S_Q_HEAD_INIT: begin
                    cur_kv_head <= mapped_kv_head;
                    cur_dim_cnt <= {DIM_W{1'b0}};
                    q_rd_addr   <= cur_q_head * HEAD_DIM;
                    state       <= S_Q_LOAD_REQ;
                end

                S_Q_LOAD_REQ: begin
                    state <= S_Q_LOAD_CAP;
                end

                S_Q_LOAD_CAP: begin
                    q_local[cur_dim_cnt] <= q_rd_data;
                    if (cur_dim_cnt + 1'b1 < HEAD_DIM) begin
                        cur_dim_cnt <= cur_dim_cnt + 1'b1;
                        q_rd_addr   <= cur_q_head * HEAD_DIM + (cur_dim_cnt + 1'b1);
                        state       <= S_Q_LOAD_REQ;
                    end else begin
                        state <= S_SCORE_INIT;
                    end
                end

                // -------------------------------------------------------------
                // 3. Score Loop: Pipelined stream of K vectors
                // -------------------------------------------------------------
                S_SCORE_INIT: begin
                    cur_pos_cnt <= {POS_W{1'b0}};
                    cur_dim_cnt <= {DIM_W{1'b0}};
                    dot_acc     <= 32'sd0;
                    read_pipe   <= 2'b00;
                    state       <= S_SCORE_STREAM;
                end

                S_SCORE_STREAM: begin
                    // Address Generation Stage
                    if (cur_pos_cnt <= pos[POS_W-1:0]) begin
                        kv_rd_en      <= 1'b1;
                        kv_rd_layer   <= layer_idx[LAYER_W-1:0];
                        kv_rd_is_v    <= 1'b0; // Key
                        kv_rd_kv_head <= cur_kv_head;
                        kv_rd_pos     <= cur_pos_cnt;
                        kv_rd_dim     <= cur_dim_cnt;
                        read_pipe     <= {read_pipe[0], 1'b1};
                        pipe_dim[0]   <= cur_dim_cnt;
                        pipe_pos[0]   <= cur_pos_cnt;

                        if (cur_dim_cnt + 1'b1 < HEAD_DIM) begin
                            cur_dim_cnt <= cur_dim_cnt + 1'b1;
                        end else begin
                            cur_dim_cnt <= {DIM_W{1'b0}};
                            cur_pos_cnt <= cur_pos_cnt + 1'b1;
                        end
                    end else begin
                        kv_rd_en  <= 1'b0;
                        read_pipe <= {read_pipe[0], 1'b0};
                    end

                    pipe_dim[1] <= pipe_dim[0];
                    pipe_pos[1] <= pipe_pos[0];

                    // Accumulation Stage (valid on read_pipe[1])
                    if (read_pipe[1]) begin
                        dot_acc <= dot_acc + (q_local[pipe_dim[1]] * kv_rd_data);

                        if (pipe_dim[1] + 1'b1 == HEAD_DIM) begin
                            // Last element of position: store score = (dot + term) >>> 1
                            scores_mem[pipe_pos[1]] <= (dot_acc + (q_local[pipe_dim[1]] * kv_rd_data)) >>> 1;
                            dot_acc <= 32'sd0;

                            if (pipe_pos[1] == pos[POS_W-1:0]) begin
                                sm_start <= 1'b1;
                                state    <= S_SOFTMAX_WAIT;
                            end
                        end
                    end
                end

                // -------------------------------------------------------------
                // 4. Softmax over scores
                // -------------------------------------------------------------
                S_SOFTMAX_WAIT: begin
                    if (sm_done) begin
                        state <= S_CTX_INIT;
                    end
                end

                // -------------------------------------------------------------
                // 5. Context Weighted Sum: Pipelined stream of V vectors
                // -------------------------------------------------------------
                S_CTX_INIT: begin
                    cur_dim_cnt <= {DIM_W{1'b0}};
                    cur_pos_cnt <= {POS_W{1'b0}};
                    ctx_acc     <= 64'sd0;
                    read_pipe   <= 2'b00;
                    state       <= S_CTX_STREAM;
                end

                S_CTX_STREAM: begin
                    // Address Generation: outer loop dim (0..hd-1), inner loop pos (0..pos)
                    if (cur_dim_cnt < HEAD_DIM) begin
                        kv_rd_en      <= 1'b1;
                        kv_rd_layer   <= layer_idx[LAYER_W-1:0];
                        kv_rd_is_v    <= 1'b1; // Value
                        kv_rd_kv_head <= cur_kv_head;
                        kv_rd_pos     <= cur_pos_cnt;
                        kv_rd_dim     <= cur_dim_cnt;
                        read_pipe     <= {read_pipe[0], 1'b1};
                        pipe_dim[0]   <= cur_dim_cnt;
                        pipe_pos[0]   <= cur_pos_cnt;

                        if (cur_pos_cnt < pos[POS_W-1:0]) begin
                            cur_pos_cnt <= cur_pos_cnt + 1'b1;
                        end else begin
                            cur_pos_cnt <= {POS_W{1'b0}};
                            cur_dim_cnt <= cur_dim_cnt + 1'b1;
                        end
                    end else begin
                        kv_rd_en  <= 1'b0;
                        read_pipe <= {read_pipe[0], 1'b0};
                    end

                    pipe_dim[1] <= pipe_dim[0];
                    pipe_pos[1] <= pipe_pos[0];

                    // Context Accumulation Stage
                    if (read_pipe[1]) begin
                        ctx_acc <= ctx_acc + ($signed({16'd0, prob_mem[pipe_pos[1]]}) * kv_rd_data);

                        if (pipe_pos[1] == pos[POS_W-1:0]) begin
                            attn_wr_en   <= 1'b1;
                            attn_wr_addr <= cur_q_head * HEAD_DIM + pipe_dim[1];
                            attn_wr_data <= sat_int8(cur_ctx_rnd);
                            ctx_acc      <= 64'sd0;

                            if (pipe_dim[1] + 1'b1 == HEAD_DIM) begin
                                if (cur_q_head + 1'b1 < NUM_Q_HEADS) begin
                                    cur_q_head <= cur_q_head + 1'b1;
                                    state      <= S_Q_HEAD_INIT;
                                end else begin
                                    state <= S_DONE;
                                end
                            end
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
/* verilator lint_on BLKSEQ */
