// top_accel.v - Complete Transformer Decoder Accelerator
// Contract references: docs/part2/02_architecture_and_contracts.md §1, §6; docs/part2/04 §UD.1

`timescale 1ns / 1ps

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off BLKSEQ */
/* verilator lint_off UNUSEDSIGNAL */
module top_accel #(
    parameter VOCAB_SIZE        = 256,
    parameter HIDDEN_SIZE       = 64,
    parameter INTERMEDIATE_SIZE = 128,
    parameter NUM_LAYERS        = 2,
    parameter NUM_Q_HEADS       = 4,
    parameter NUM_KV_HEADS      = 2,
    parameter HEAD_DIM          = 16,
    parameter MAX_CTX           = 32,
    parameter TILE_R            = 8,
    parameter TILE_C            = 16,
    parameter PROG_ADDR_W       = 8,
    parameter LAYER_W           = 5,
    parameter ROM_WEIGHTS_ADDR_W = 24,
    parameter ROM_PARAMS_ADDR_W  = 20,
    parameter ROM_BASES_ADDR_W   = 9,
    parameter ROM_WEIGHTS_DEPTH = 32768,
    parameter ROM_PARAMS_DEPTH  = 2048,
    parameter ROM_PROG_DEPTH    = 256,
    parameter ROM_BASES_DEPTH   = 512,
    parameter ROM_WEIGHTS_FILE  = "",
    parameter ROM_PARAMS_FILE   = "",
    parameter ROM_PROG_FILE     = "",
    parameter ROM_BASES_FILE    = "",
    parameter ROM_EXP2_FILE     = "",
    parameter ROM_RSQRT_FILE    = "",
    parameter ROM_SILU_FILE     = "",
    parameter ROM_COS_FILE      = "",
    parameter ROM_SIN_FILE      = ""
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        seq_reset,
    input  wire        step_valid,
    output wire        step_ready,
    input  wire [15:0] tok_in,
    output reg         out_valid,
    output reg  [15:0] tok_out
);

    // =========================================================================
    // 1. Step Interface FSM & Position Counter
    // =========================================================================
    reg [15:0] pos_reg;
    reg        step_running;
    reg        seq_start;

    wire seq_busy;
    wire seq_done;
    wire [15:0] seq_tok_out;
    wire seq_tok_out_valid;

    assign step_ready = (!seq_busy && !step_running);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pos_reg      <= 16'd0;
            step_running <= 1'b0;
            seq_start    <= 1'b0;
            out_valid    <= 1'b0;
            tok_out      <= 16'd0;
        end else begin
            seq_start <= 1'b0;
            out_valid <= 1'b0;

            if (seq_reset) begin
                pos_reg <= 16'd0;
            end

            if (step_valid && step_ready) begin
                step_running <= 1'b1;
                seq_start    <= 1'b1;
            end

            if (seq_tok_out_valid) begin
                tok_out   <= seq_tok_out;
                out_valid <= 1'b1;
            end

            if (seq_done) begin
                step_running <= 1'b0;
                pos_reg      <= pos_reg + 1'b1;
            end
        end
    end

    // =========================================================================
    // 2. Program ROM & Layer Base Table ROM
    // =========================================================================
    wire [PROG_ADDR_W-1:0] prog_rom_addr;
    wire [127:0]           prog_rom_data;

    rom_sync #(
        .DATA_WIDTH(128),
        .ADDR_WIDTH(PROG_ADDR_W),
        .DEPTH(ROM_PROG_DEPTH),
        .INIT_FILE(ROM_PROG_FILE)
    ) prog_rom_inst (
        .clk(clk),
        .addr(prog_rom_addr),
        .data_out(prog_rom_data)
    );

    wire [ROM_BASES_ADDR_W-1:0] layer_table_addr;
    wire [63:0]                 layer_table_data;

    rom_sync #(
        .DATA_WIDTH(64),
        .ADDR_WIDTH(ROM_BASES_ADDR_W),
        .DEPTH(ROM_BASES_DEPTH),
        .INIT_FILE(ROM_BASES_FILE)
    ) layer_bases_rom_inst (
        .clk(clk),
        .addr(layer_table_addr),
        .data_out(layer_table_data)
    );

    wire [31:0] layer_table_weight_base = layer_table_data[63:32];
    wire [31:0] layer_table_param_base  = layer_table_data[31:0];

    // =========================================================================
    // 3. Sequencer Microprogram Dispatcher
    // =========================================================================
    wire embed_start, embed_done;
    wire norm_start, norm_done;
    wire linear_start, linear_done;
    wire rope_start, rope_done;
    wire attn_start, attn_done;
    wire add_start, add_done;
    wire swiglu_start, swiglu_done;
    wire argmax_start, argmax_done;
    wire [15:0] argmax_res_tok;

    wire [PROG_ADDR_W-1:0] cur_pc;
    wire [LAYER_W-1:0]     cur_layer;
    wire [7:0]             cur_opcode;
    wire [7:0]             cur_src_buf;
    wire [7:0]             cur_dst_buf;
    wire [7:0]             cur_flags;
    wire [15:0]            cur_dim0;
    wire [15:0]            cur_dim1;
    wire [31:0]            cur_weight_base;
    wire [31:0]            cur_param_base;

    sequencer #(
        .PROG_ADDR_W(PROG_ADDR_W),
        .LAYER_W(LAYER_W),
        .BUF_ID_W(8),
        .DIM_W(16),
        .ROM_ADDR_W(32)
    ) seq_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(seq_start),
        .busy(seq_busy),
        .done(seq_done),
        .pos(pos_reg),
        .tok_in(tok_in),
        .tok_out(seq_tok_out),
        .tok_out_valid(seq_tok_out_valid),
        .layer_loop_start(8'd1),
        .layer_loop_end(8'd15),
        .num_layers(NUM_LAYERS[LAYER_W-1:0]),
        .prog_rom_addr(prog_rom_addr),
        .prog_rom_data(prog_rom_data),
        .layer_table_addr(layer_table_addr),
        .layer_table_weight_base(layer_table_weight_base),
        .layer_table_param_base(layer_table_param_base),
        .embed_start(embed_start),
        .embed_done(embed_done),
        .norm_start(norm_start),
        .norm_done(norm_done),
        .linear_start(linear_start),
        .linear_done(linear_done),
        .rope_start(rope_start),
        .rope_done(rope_done),
        .attn_start(attn_start),
        .attn_done(attn_done),
        .add_start(add_start),
        .add_done(add_done),
        .swiglu_start(swiglu_start),
        .swiglu_done(swiglu_done),
        .argmax_start(argmax_start),
        .argmax_done(argmax_done),
        .argmax_res_tok(argmax_res_tok),
        .cur_pc(cur_pc),
        .cur_layer(cur_layer),
        .cur_opcode(cur_opcode),
        .cur_src_buf(cur_src_buf),
        .cur_dst_buf(cur_dst_buf),
        .cur_flags(cur_flags),
        .cur_dim0(cur_dim0),
        .cur_dim1(cur_dim1),
        .cur_weight_base(cur_weight_base),
        .cur_param_base(cur_param_base)
    );

    // =========================================================================
    // 4. Weight ROM & Requant Parameter ROM Instances
    // =========================================================================
    wire [ROM_WEIGHTS_ADDR_W-1:0] weight_rom_addr;
    wire [127:0]                  weight_rom_data;

    rom_sync #(
        .DATA_WIDTH(128),
        .ADDR_WIDTH(ROM_WEIGHTS_ADDR_W),
        .DEPTH(ROM_WEIGHTS_DEPTH),
        .INIT_FILE(ROM_WEIGHTS_FILE)
    ) rom_weights_inst (
        .clk(clk),
        .addr(weight_rom_addr),
        .data_out(weight_rom_data)
    );

    wire [ROM_PARAMS_ADDR_W-1:0] param_rom_addr;
    wire [63:0]                  param_rom_data;

    rom_sync #(
        .DATA_WIDTH(64),
        .ADDR_WIDTH(ROM_PARAMS_ADDR_W),
        .DEPTH(ROM_PARAMS_DEPTH),
        .INIT_FILE(ROM_PARAMS_FILE)
    ) rom_params_inst (
        .clk(clk),
        .addr(param_rom_addr),
        .data_out(param_rom_data)
    );

    // =========================================================================
    // 5. Lookup Tables (RoPE, RMSNorm rsqrt, Softmax exp2, SwiGLU silu)
    // =========================================================================
    wire [15:0] rope_cos_addr, rope_sin_addr;
    wire [15:0] rope_cos_data, rope_sin_data;

    rom_sync #(
        .DATA_WIDTH(16),
        .ADDR_WIDTH(16),
        .DEPTH(MAX_CTX * (HEAD_DIM / 2)),
        .INIT_FILE(ROM_COS_FILE)
    ) rom_rope_cos_inst (
        .clk(clk),
        .addr(rope_cos_addr),
        .data_out(rope_cos_data)
    );

    rom_sync #(
        .DATA_WIDTH(16),
        .ADDR_WIDTH(16),
        .DEPTH(MAX_CTX * (HEAD_DIM / 2)),
        .INIT_FILE(ROM_SIN_FILE)
    ) rom_rope_sin_inst (
        .clk(clk),
        .addr(rope_sin_addr),
        .data_out(rope_sin_data)
    );

    wire [7:0]  norm_rsqrt_addr;
    wire [15:0] norm_rsqrt_data;

    rom_sync #(
        .DATA_WIDTH(16),
        .ADDR_WIDTH(8),
        .DEPTH(256),
        .INIT_FILE(ROM_RSQRT_FILE)
    ) rom_rsqrt_inst (
        .clk(clk),
        .addr(norm_rsqrt_addr),
        .data_out(norm_rsqrt_data)
    );

    wire [7:0]  attn_exp2_addr;
    wire [17:0] attn_exp2_data;

    rom_sync #(
        .DATA_WIDTH(18),
        .ADDR_WIDTH(8),
        .DEPTH(256),
        .INIT_FILE(ROM_EXP2_FILE)
    ) rom_exp2_inst (
        .clk(clk),
        .addr(attn_exp2_addr),
        .data_out(attn_exp2_data)
    );

    wire [8:0] swiglu_silu_addr;
    wire [7:0] swiglu_silu_data;

    rom_sync #(
        .DATA_WIDTH(8),
        .ADDR_WIDTH(9),
        .DEPTH(512),
        .INIT_FILE(ROM_SILU_FILE)
    ) rom_silu_inst (
        .clk(clk),
        .addr(swiglu_silu_addr),
        .data_out(swiglu_silu_data)
    );

    // =========================================================================
    // 6. Dynamic Activation Buffers (BUF 0 .. 9)
    // =========================================================================
    // 0: BUF_RES (Residual container, 24 bits)
    reg signed [23:0] buf_res [0:HIDDEN_SIZE-1];

    // 1: BUF_ACT (Normed activations or intermediate linear outputs, 24 bits)
    reg signed [23:0] buf_act [0:HIDDEN_SIZE-1];

    // 2: BUF_Q (Query head activations, 8 bits)
    reg signed [7:0] buf_q [0:NUM_Q_HEADS*HEAD_DIM-1];

    // 3: BUF_K (Key head activations, 8 bits)
    reg signed [7:0] buf_k [0:NUM_KV_HEADS*HEAD_DIM-1];

    // 4: BUF_V (Value head activations, 8 bits)
    reg signed [7:0] buf_v [0:NUM_KV_HEADS*HEAD_DIM-1];

    // 5: BUF_ATTN_OUT (Attention context output, 8 bits)
    reg signed [7:0] buf_attn_out [0:HIDDEN_SIZE-1];

    // 6: BUF_MLP_GATE (Gate projection output, 8 bits)
    reg signed [7:0] buf_gate [0:INTERMEDIATE_SIZE-1];

    // 7: BUF_MLP_UP (Up projection output, 8 bits)
    reg signed [7:0] buf_up [0:INTERMEDIATE_SIZE-1];

    // 8: BUF_MLP_SWIGLU (SwiGLU output, 8 bits)
    reg signed [7:0] buf_swiglu [0:INTERMEDIATE_SIZE-1];

    // 9: BUF_LOGITS (Vocab logits, 24 bits)
    reg signed [23:0] buf_logits [0:VOCAB_SIZE-1];

    // =========================================================================
    // 7. Unit Declarations and Interconnect
    // =========================================================================

    // 7.1 Embed Gather Unit
    wire        embed_busy;
    wire [31:0] embed_addr;
    wire        embed_dst_we;
    wire [15:0] embed_dst_addr;
    wire signed [23:0] embed_dst_data;

    // Tiled weight mapping for embedding gather (32-bit clean for large vocab):
    wire [31:0] embed_tok_id   = {16'd0, tok_in};
    wire [31:0] embed_cnt      = embed_addr - (embed_tok_id * HIDDEN_SIZE);
    wire [31:0] embed_tc       = embed_cnt >> 4;
    wire [3:0]  embed_c        = embed_cnt[3:0];
    wire [31:0] embed_tr       = embed_tok_id >> 3;
    wire [2:0]  embed_r        = embed_tok_id[2:0];
    wire [31:0] embed_num_tc   = HIDDEN_SIZE >> 4;
    wire [31:0] embed_word_idx = (embed_tr * embed_num_tc + embed_tc) * 8 + {29'd0, embed_r};
    wire [31:0] embed_word_addr = {4'd0, cur_weight_base[31:4]} + embed_word_idx;
    reg [3:0] embed_c_pipe;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            embed_c_pipe <= 4'd0;
        end else begin
            embed_c_pipe <= embed_c;
        end
    end
    wire signed [7:0] embed_weight_byte = weight_rom_data[embed_c_pipe*8 +: 8];

    // Parameter lookup for embedding gather:
    wire signed [31:0] embed_M = param_rom_data[63:32];
    wire [5:0]         embed_s = param_rom_data[5:0];

    embed_gather #(
        .WEIGHT_WIDTH(8),
        .OUT_WIDTH(24),
        .OUT_MIN(-24'sd8388608),
        .OUT_MAX(24'sd8388607)
    ) embed_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(embed_start),
        .token_id(tok_in),
        .dim(cur_dim0),
        .M(embed_M),
        .s(embed_s),
        .busy(embed_busy),
        .done(embed_done),
        .embed_addr(embed_addr),
        .embed_data(embed_weight_byte),
        .dst_we(embed_dst_we),
        .dst_addr(embed_dst_addr),
        .dst_data(embed_dst_data)
    );

    // 7.2 RMSNorm Unit
    wire        norm_busy;
    wire [15:0] norm_src_addr;
    reg  signed [23:0] norm_src_data;
    wire [15:0] norm_gamma_addr;
    wire        norm_dst_we;
    wire [15:0] norm_dst_addr;
    wire signed [7:0] norm_dst_data;

    always @(posedge clk) begin
        norm_src_data <= buf_res[norm_src_addr];
    end

    wire signed [15:0] norm_gamma_data = param_rom_data[47:32];

    rmsnorm #(
        .DATA_WIDTH(24),
        .ACC_WIDTH(48),
        .GAMMA_WIDTH(16),
        .RSQRT_WIDTH(16),
        .OUT_WIDTH(8),
        .OUT_MIN(-8'sd127),
        .OUT_MAX(8'sd127)
    ) norm_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(norm_start),
        .dim(cur_dim0),
        .M(32'sd1),
        .s(param_rom_data[5:0]),
        .busy(norm_busy),
        .done(norm_done),
        .src_addr(norm_src_addr),
        .src_data(norm_src_data),
        .gamma_addr(norm_gamma_addr),
        .gamma_data(norm_gamma_data),
        .rsqrt_lut_addr(norm_rsqrt_addr),
        .rsqrt_lut_data(norm_rsqrt_data),
        .dst_we(norm_dst_we),
        .dst_addr(norm_dst_addr),
        .dst_data(norm_dst_data)
    );

    // 7.3 Linear Engine
    wire        linear_busy;
    wire [15:0] linear_act_rd_addr;
    wire [31:0] linear_weight_rd_addr;
    wire [31:0] linear_param_rd_addr;
    wire        linear_out_wr_en;
    wire [15:0] linear_out_wr_addr;
    wire signed [23:0] linear_out_wr_data;

    // Buffer read multiplexer for linear engine:
    reg signed [127:0] buf_act_tc_data;
    reg signed [127:0] buf_attn_out_tc_data;
    reg signed [127:0] buf_swiglu_tc_data;

    integer bi;
    always @(posedge clk) begin
        for (bi = 0; bi < 16; bi = bi + 1) begin
            buf_act_tc_data[bi*8 +: 8]      <= buf_act[linear_act_rd_addr * 16 + bi][7:0];
            buf_attn_out_tc_data[bi*8 +: 8] <= buf_attn_out[linear_act_rd_addr * 16 + bi];
            buf_swiglu_tc_data[bi*8 +: 8]   <= buf_swiglu[linear_act_rd_addr * 16 + bi];
        end
    end

    wire signed [127:0] linear_act_rd_data = (cur_src_buf == 8'd5) ? buf_attn_out_tc_data :
                                            ((cur_src_buf == 8'd8) ? buf_swiglu_tc_data :
                                            buf_act_tc_data);

    wire linear_out_is_res = (cur_dst_buf == 8'd0) || (cur_dst_buf == 8'd9) || (cur_flags[0]) ||
                             (cur_opcode == 8'd3 && cur_dst_buf == 8'd1 && (cur_pc == 8'd8 || cur_pc == 8'd14));

    linear_engine #(
        .R(TILE_R),
        .C(TILE_C),
        .ACT_WIDTH(8),
        .WEIGHT_WIDTH(8),
        .ACC_WIDTH(32),
        .M_WIDTH(32),
        .SHIFT_WIDTH(6),
        .BUF_ADDR_WIDTH(16),
        .ROM_ADDR_WIDTH(32)
    ) linear_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(linear_start),
        .busy(linear_busy),
        .done(linear_done),
        .matrix_n(cur_dim0),
        .matrix_k(cur_dim1),
        .weight_base_addr({4'd0, cur_weight_base[31:4]}),
        .param_base_addr(cur_param_base),
        .out_is_res(linear_out_is_res),
        .act_rd_addr(linear_act_rd_addr),
        .act_rd_data(linear_act_rd_data),
        .weight_rd_addr(linear_weight_rd_addr),
        .weight_rd_data(weight_rom_data),
        .param_rd_addr(linear_param_rd_addr),
        .param_m_data(param_rom_data[63:32]),
        .param_s_data(param_rom_data[5:0]),
        .out_wr_en(linear_out_wr_en),
        .out_wr_addr(linear_out_wr_addr),
        .out_wr_data(linear_out_wr_data)
    );

    // 7.4 RoPE Unit
    wire        rope_busy;
    wire [15:0] rope_src_addr;
    reg  signed [7:0] rope_src_data;
    wire [15:0] rope_table_addr;
    wire        rope_dst_we;
    wire [15:0] rope_dst_addr;
    wire signed [7:0] rope_dst_data;

    always @(posedge clk) begin
        if (cur_src_buf == 8'd2) begin
            rope_src_data <= buf_q[rope_src_addr];
        end else begin
            rope_src_data <= buf_k[rope_src_addr];
        end
    end

    assign rope_cos_addr = rope_table_addr;
    assign rope_sin_addr = rope_table_addr;

    rope #(
        .DATA_WIDTH(8),
        .TABLE_WIDTH(16),
        .TABLE_SHIFT(14),
        .OUT_WIDTH(8),
        .OUT_MIN(-8'sd127),
        .OUT_MAX(8'sd127)
    ) rope_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(rope_start),
        .num_heads(cur_dim0),
        .head_dim(cur_dim1),
        .pos(pos_reg),
        .busy(rope_busy),
        .done(rope_done),
        .src_addr(rope_src_addr),
        .src_data(rope_src_data),
        .table_addr(rope_table_addr),
        .cos_data(rope_cos_data),
        .sin_data(rope_sin_data),
        .dst_we(rope_dst_we),
        .dst_addr(rope_dst_addr),
        .dst_data(rope_dst_data)
    );

    // 7.5 KV Cache & GQA Attention Unit
    wire        attn_busy;
    wire [15:0] attn_q_rd_addr, attn_k_rd_addr, attn_v_rd_addr;
    reg  signed [7:0] attn_q_rd_data, attn_k_rd_data, attn_v_rd_data;
    wire        attn_wr_en;
    wire [15:0] attn_wr_addr;
    wire signed [7:0] attn_wr_data;

    always @(posedge clk) begin
        attn_q_rd_data <= buf_q[attn_q_rd_addr];
        attn_k_rd_data <= buf_k[attn_k_rd_addr];
        attn_v_rd_data <= buf_v[attn_v_rd_addr];
    end

    localparam KV_LAYER_W   = (NUM_LAYERS > 1) ? $clog2(NUM_LAYERS) : 1;
    localparam KV_KV_HEAD_W = (NUM_KV_HEADS > 1) ? $clog2(NUM_KV_HEADS) : 1;
    localparam KV_POS_W     = (MAX_CTX > 1) ? $clog2(MAX_CTX) : 1;
    localparam KV_DIM_W     = (HEAD_DIM > 1) ? $clog2(HEAD_DIM) : 1;

    wire                   attn_kv_wr_en;
    wire [KV_LAYER_W-1:0]   attn_kv_wr_layer;
    wire                   attn_kv_wr_is_v;
    wire [KV_KV_HEAD_W-1:0] attn_kv_wr_kv_head;
    wire [KV_POS_W-1:0]     attn_kv_wr_pos;
    wire [KV_DIM_W-1:0]     attn_kv_wr_dim;
    wire signed [7:0]      attn_kv_wr_data;

    wire                   attn_kv_rd_en;
    wire [KV_LAYER_W-1:0]   attn_kv_rd_layer;
    wire                   attn_kv_rd_is_v;
    wire [KV_KV_HEAD_W-1:0] attn_kv_rd_kv_head;
    wire [KV_POS_W-1:0]     attn_kv_rd_pos;
    wire [KV_DIM_W-1:0]     attn_kv_rd_dim;
    wire signed [7:0]      attn_kv_rd_data;

    attention #(
        .NUM_LAYERS(NUM_LAYERS),
        .NUM_KV_HEADS(NUM_KV_HEADS),
        .NUM_Q_HEADS(NUM_Q_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .MAX_CTX(MAX_CTX),
        .DATA_WIDTH(8),
        .SCORES_WIDTH(32),
        .EXP_WIDTH(18),
        .PROB_WIDTH(16),
        .BUF_ADDR_W(16)
    ) attn_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(attn_start),
        .busy(attn_busy),
        .done(attn_done),
        .layer_idx(cur_layer[4:0]),
        .pos(pos_reg),
        .q_rd_addr(attn_q_rd_addr),
        .q_rd_data(attn_q_rd_data),
        .k_rd_addr(attn_k_rd_addr),
        .k_rd_data(attn_k_rd_data),
        .v_rd_addr(attn_v_rd_addr),
        .v_rd_data(attn_v_rd_data),
        .attn_wr_en(attn_wr_en),
        .attn_wr_addr(attn_wr_addr),
        .attn_wr_data(attn_wr_data),
        .exp2_lut_addr(attn_exp2_addr),
        .exp2_lut_data(attn_exp2_data),
        .kv_wr_en(attn_kv_wr_en),
        .kv_wr_layer(attn_kv_wr_layer),
        .kv_wr_is_v(attn_kv_wr_is_v),
        .kv_wr_kv_head(attn_kv_wr_kv_head),
        .kv_wr_pos(attn_kv_wr_pos),
        .kv_wr_dim(attn_kv_wr_dim),
        .kv_wr_data(attn_kv_wr_data),
        .kv_rd_en(attn_kv_rd_en),
        .kv_rd_layer(attn_kv_rd_layer),
        .kv_rd_is_v(attn_kv_rd_is_v),
        .kv_rd_kv_head(attn_kv_rd_kv_head),
        .kv_rd_pos(attn_kv_rd_pos),
        .kv_rd_dim(attn_kv_rd_dim),
        .kv_rd_data(attn_kv_rd_data)
    );

    localparam KV_TOTAL_ELS = NUM_LAYERS * 2 * NUM_KV_HEADS * MAX_CTX * HEAD_DIM;
    localparam KV_ADDR_W    = (KV_TOTAL_ELS > 1) ? $clog2(KV_TOTAL_ELS) : 1;

    kv_cache #(
        .NUM_LAYERS(NUM_LAYERS),
        .NUM_KV_HEADS(NUM_KV_HEADS),
        .MAX_CTX(MAX_CTX),
        .HEAD_DIM(HEAD_DIM),
        .DATA_WIDTH(8)
    ) kv_cache_inst (
        .clk(clk),
        .rst_n(rst_n),
        .use_direct_addr(1'b0),
        .wr_en(attn_kv_wr_en),
        .wr_addr_in({KV_ADDR_W{1'b0}}),
        .wr_layer(attn_kv_wr_layer),
        .wr_is_v(attn_kv_wr_is_v),
        .wr_kv_head(attn_kv_wr_kv_head),
        .wr_pos(attn_kv_wr_pos),
        .wr_dim(attn_kv_wr_dim),
        .wr_data(attn_kv_wr_data),
        .rd_en(attn_kv_rd_en),
        .rd_addr_in({KV_ADDR_W{1'b0}}),
        .rd_layer(attn_kv_rd_layer),
        .rd_is_v(attn_kv_rd_is_v),
        .rd_kv_head(attn_kv_rd_kv_head),
        .rd_pos(attn_kv_rd_pos),
        .rd_dim(attn_kv_rd_dim),
        .rd_data(attn_kv_rd_data)
    );

    // 7.6 Residual Add Unit
    wire        add_busy;
    wire [15:0] add_a_addr, add_b_addr;
    reg  signed [23:0] add_a_data, add_b_data;
    wire        add_dst_we;
    wire [15:0] add_dst_addr;
    wire signed [23:0] add_dst_data;

    always @(posedge clk) begin
        add_a_data <= buf_act[add_a_addr];
        add_b_data <= buf_res[add_b_addr];
    end

    residual_add #(
        .WIDTH(24),
        .MIN_VAL(-24'sd8388608),
        .MAX_VAL(24'sd8388607)
    ) add_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(add_start),
        .dim(cur_dim0),
        .busy(add_busy),
        .done(add_done),
        .a_addr(add_a_addr),
        .a_data(add_a_data),
        .b_addr(add_b_addr),
        .b_data(add_b_data),
        .dst_we(add_dst_we),
        .dst_addr(add_dst_addr),
        .dst_data(add_dst_data)
    );

    // 7.7 SwiGLU Unit
    wire        swiglu_busy;
    wire [15:0] swiglu_gate_addr, swiglu_up_addr;
    reg  signed [7:0] swiglu_gate_data, swiglu_up_data;
    wire        swiglu_dst_we;
    wire [15:0] swiglu_dst_addr;
    wire signed [7:0] swiglu_dst_data;

    always @(posedge clk) begin
        swiglu_gate_data <= buf_gate[swiglu_gate_addr];
        swiglu_up_data   <= buf_up[swiglu_up_addr];
    end

    swiglu #(
        .DATA_WIDTH(8),
        .LUT_WIDTH(8),
        .LUT_SIZE(512),
        .OUT_WIDTH(8),
        .OUT_MIN(-8'sd127),
        .OUT_MAX(8'sd127)
    ) swiglu_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(swiglu_start),
        .dim(cur_dim0),
        .M(32'sd1),
        .s(6'd6),
        .busy(swiglu_busy),
        .done(swiglu_done),
        .gate_addr(swiglu_gate_addr),
        .gate_data(swiglu_gate_data),
        .up_addr(swiglu_up_addr),
        .up_data(swiglu_up_data),
        .silu_lut_addr(swiglu_silu_addr),
        .silu_lut_data(swiglu_silu_data),
        .dst_we(swiglu_dst_we),
        .dst_addr(swiglu_dst_addr),
        .dst_data(swiglu_dst_data)
    );

    // 7.8 ArgMax Unit
    wire        argmax_busy;
    wire [15:0] argmax_logit_addr;
    reg  signed [31:0] argmax_logit_data;
    wire [15:0] argmax_scale_addr;

    always @(posedge clk) begin
        argmax_logit_data <= {{8{buf_logits[argmax_logit_addr][23]}}, buf_logits[argmax_logit_addr]};
    end

    argmax #(
        .ACC_WIDTH(32),
        .M_WIDTH(32),
        .SHIFT_WIDTH(6),
        .TOK_WIDTH(16)
    ) argmax_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(argmax_start),
        .vocab_size(cur_dim0),
        .busy(argmax_busy),
        .done(argmax_done),
        .best_token_id(argmax_res_tok),
        .logit_addr(argmax_logit_addr),
        .logit_data(argmax_logit_data),
        .scale_addr(argmax_scale_addr),
        .M_data(32'sd1),
        .s_data(6'd0)
    );

    // =========================================================================
    // 8. Buffer Write Routing
    // =========================================================================
    always @(posedge clk) begin
        // BUF 0: BUF_RES
        if (embed_dst_we) begin
            buf_res[embed_dst_addr] <= embed_dst_data;
        end else if (add_dst_we) begin
            buf_res[add_dst_addr] <= add_dst_data;
        end

        // BUF 1: BUF_ACT
        if (norm_dst_we) begin
            buf_act[norm_dst_addr] <= {{16{norm_dst_data[7]}}, norm_dst_data};
        end else if (linear_out_wr_en && cur_dst_buf == 8'd1) begin
            buf_act[linear_out_wr_addr] <= linear_out_wr_data;
        end

        // BUF 2: BUF_Q
        if (linear_out_wr_en && cur_dst_buf == 8'd2) begin
            buf_q[linear_out_wr_addr] <= linear_out_wr_data[7:0];
        end else if (rope_dst_we && cur_dst_buf == 8'd2) begin
            buf_q[rope_dst_addr] <= rope_dst_data;
        end

        // BUF 3: BUF_K
        if (linear_out_wr_en && cur_dst_buf == 8'd3) begin
            buf_k[linear_out_wr_addr] <= linear_out_wr_data[7:0];
        end else if (rope_dst_we && cur_dst_buf == 8'd3) begin
            buf_k[rope_dst_addr] <= rope_dst_data;
        end

        // BUF 4: BUF_V
        if (linear_out_wr_en && cur_dst_buf == 8'd4) begin
            buf_v[linear_out_wr_addr] <= linear_out_wr_data[7:0];
        end

        // BUF 5: BUF_ATTN_OUT
        if (attn_wr_en) begin
            buf_attn_out[attn_wr_addr] <= attn_wr_data;
        end

        // BUF 6: BUF_MLP_GATE
        if (linear_out_wr_en && cur_dst_buf == 8'd6) begin
            buf_gate[linear_out_wr_addr] <= linear_out_wr_data[7:0];
        end

        // BUF 7: BUF_MLP_UP
        if (linear_out_wr_en && cur_dst_buf == 8'd7) begin
            buf_up[linear_out_wr_addr] <= linear_out_wr_data[7:0];
        end

        // BUF 8: BUF_MLP_SWIGLU
        if (swiglu_dst_we) begin
            buf_swiglu[swiglu_dst_addr] <= swiglu_dst_data;
        end

        // BUF 9: BUF_LOGITS
        if (linear_out_wr_en && cur_dst_buf == 8'd9) begin
            buf_logits[linear_out_wr_addr] <= linear_out_wr_data;
        end
    end

    // ROM address muxing:
    assign weight_rom_addr = embed_busy ? embed_word_addr[ROM_WEIGHTS_ADDR_W-1:0] :
                             linear_weight_rd_addr[ROM_WEIGHTS_ADDR_W-1:0];
    assign param_rom_addr  = embed_busy ? (cur_param_base[ROM_PARAMS_ADDR_W-1:0] + {{(ROM_PARAMS_ADDR_W-16){1'b0}}, tok_in}) :
                             (norm_busy ? (cur_param_base[ROM_PARAMS_ADDR_W-1:0] + {{(ROM_PARAMS_ADDR_W-16){1'b0}}, norm_gamma_addr}) :
                             linear_param_rd_addr[ROM_PARAMS_ADDR_W-1:0]);

endmodule
/* verilator lint_on WIDTHEXPAND */
/* verilator lint_on WIDTHTRUNC */
/* verilator lint_on BLKSEQ */
/* verilator lint_on UNUSEDSIGNAL */
