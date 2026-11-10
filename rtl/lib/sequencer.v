// sequencer.v - Microprogram Sequencer & Op Dispatcher
// Contract references: docs/part2/02_architecture_and_contracts.md §6, docs/part2/04 §UC.12

`timescale 1ns / 1ps

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
module sequencer #(
    parameter PROG_ADDR_W = 8,
    parameter LAYER_W     = 5,
    parameter BUF_ID_W    = 8,
    parameter DIM_W       = 16,
    parameter ROM_ADDR_W  = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // Step Execution Control Handshake
    input  wire                          start,
    output reg                           busy,
    output reg                           done,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [15:0]                   pos,
    input  wire [15:0]                   tok_in,
    /* verilator lint_on UNUSEDSIGNAL */
    output reg  [15:0]                   tok_out,
    output reg                           tok_out_valid,

    // Microprogram Boundaries
    input  wire [PROG_ADDR_W-1:0]        layer_loop_start,
    input  wire [PROG_ADDR_W-1:0]        layer_loop_end,
    input  wire [LAYER_W-1:0]            num_layers,

    // Program ROM Read Interface (1-cycle synchronous SRAM)
    output reg  [PROG_ADDR_W-1:0]        prog_rom_addr,
    input  wire [127:0]                  prog_rom_data,

    // Layer Base Table Interface
    output wire [7:0]                    layer_table_addr,
    input  wire [ROM_ADDR_W-1:0]         layer_table_weight_base,
    input  wire [ROM_ADDR_W-1:0]         layer_table_param_base,

    // Unit Dispatch Protocols: Start & Done Handshakes
    output reg                           embed_start,
    input  wire                          embed_done,

    output reg                           norm_start,
    input  wire                          norm_done,

    output reg                           linear_start,
    input  wire                          linear_done,

    output reg                           rope_start,
    input  wire                          rope_done,

    output reg                           attn_start,
    input  wire                          attn_done,

    output reg                           add_start,
    input  wire                          add_done,

    output reg                           swiglu_start,
    input  wire                          swiglu_done,

    output reg                           argmax_start,
    input  wire                          argmax_done,
    input  wire [15:0]                   argmax_res_tok,

    // Decoded Instruction Parameters Exposed to Datapaths
    output reg  [PROG_ADDR_W-1:0]        cur_pc,
    output reg  [LAYER_W-1:0]            cur_layer,
    output reg  [7:0]                    cur_opcode,
    output reg  [BUF_ID_W-1:0]           cur_src_buf,
    output reg  [BUF_ID_W-1:0]           cur_dst_buf,
    output reg  [7:0]                    cur_flags,
    output reg  [DIM_W-1:0]              cur_dim0,
    output reg  [DIM_W-1:0]              cur_dim1,
    output reg  [ROM_ADDR_W-1:0]         cur_weight_base,
    output reg  [ROM_ADDR_W-1:0]         cur_param_base
);

    // Microprogram Opcodes matching compiler/schedule.py
    localparam OP_HALT      = 8'd0;
    localparam OP_EMBED     = 8'd1;
    localparam OP_RMSNORM   = 8'd2;
    localparam OP_LINEAR    = 8'd3;
    localparam OP_ROPE      = 8'd4;
    localparam OP_ATTENTION = 8'd5;
    localparam OP_ADD       = 8'd6;
    localparam OP_SWIGLU    = 8'd7;
    localparam OP_ARGMAX    = 8'd8;

    // FSM States
    localparam S_IDLE       = 3'd0;
    localparam S_FETCH      = 3'd1;
    localparam S_WAIT_MEM   = 3'd2;
    localparam S_DECODE     = 3'd3;
    localparam S_DISPATCH   = 3'd4;
    localparam S_WAIT_UNIT  = 3'd5;

    reg [2:0] state;

    // Relative PC inside layer loop
    wire in_layer_loop = (cur_pc >= layer_loop_start) && (cur_pc <= layer_loop_end);
    wire [3:0] layer_op_idx = cur_pc[3:0] - layer_loop_start[3:0];
    assign layer_table_addr = {cur_layer[3:0], layer_op_idx};

    // Active unit done wire mux
    reg active_unit_done;
    always @(*) begin
        case (cur_opcode)
            OP_EMBED:     active_unit_done = embed_done;
            OP_RMSNORM:   active_unit_done = norm_done;
            OP_LINEAR:    active_unit_done = linear_done;
            OP_ROPE:      active_unit_done = rope_done;
            OP_ATTENTION: active_unit_done = attn_done;
            OP_ADD:       active_unit_done = add_done;
            OP_SWIGLU:    active_unit_done = swiglu_done;
            OP_ARGMAX:    active_unit_done = argmax_done;
            default:      active_unit_done = 1'b1;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            busy             <= 1'b0;
            done             <= 1'b0;
            tok_out          <= 16'd0;
            tok_out_valid    <= 1'b0;
            prog_rom_addr    <= {PROG_ADDR_W{1'b0}};
            cur_pc           <= {PROG_ADDR_W{1'b0}};
            cur_layer        <= {LAYER_W{1'b0}};
            cur_opcode       <= 8'd0;
            cur_src_buf      <= {BUF_ID_W{1'b0}};
            cur_dst_buf      <= {BUF_ID_W{1'b0}};
            cur_flags        <= 8'd0;
            cur_dim0         <= {DIM_W{1'b0}};
            cur_dim1         <= {DIM_W{1'b0}};
            cur_weight_base  <= {ROM_ADDR_W{1'b0}};
            cur_param_base   <= {ROM_ADDR_W{1'b0}};
            embed_start      <= 1'b0;
            norm_start       <= 1'b0;
            linear_start     <= 1'b0;
            rope_start       <= 1'b0;
            attn_start       <= 1'b0;
            add_start        <= 1'b0;
            swiglu_start     <= 1'b0;
            argmax_start     <= 1'b0;
        end else begin
            done          <= 1'b0;
            tok_out_valid <= 1'b0;
            embed_start   <= 1'b0;
            norm_start    <= 1'b0;
            linear_start  <= 1'b0;
            rope_start    <= 1'b0;
            attn_start    <= 1'b0;
            add_start     <= 1'b0;
            swiglu_start  <= 1'b0;
            argmax_start  <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy          <= 1'b1;
                        cur_pc        <= {PROG_ADDR_W{1'b0}};
                        cur_layer     <= {LAYER_W{1'b0}};
                        prog_rom_addr <= {PROG_ADDR_W{1'b0}};
                        state         <= S_WAIT_MEM;
                    end
                end

                S_FETCH: begin
                    prog_rom_addr <= cur_pc;
                    state         <= S_WAIT_MEM;
                end

                S_WAIT_MEM: begin
                    // Program ROM memory register cycle
                    state <= S_DECODE;
                end

                S_DECODE: begin
                    cur_opcode  <= prog_rom_data[127:120];
                    cur_src_buf <= prog_rom_data[119:112];
                    cur_dst_buf <= prog_rom_data[111:104];
                    cur_flags   <= prog_rom_data[103:96];
                    cur_dim0    <= prog_rom_data[31:16];
                    cur_dim1    <= prog_rom_data[15:0];

                    // Base address resolution: if within layer loop and base is 0, use layer table
                    if (in_layer_loop && prog_rom_data[95:64] == 32'd0) begin
                        cur_weight_base <= layer_table_weight_base;
                    end else begin
                        cur_weight_base <= prog_rom_data[95:64];
                    end

                    if (in_layer_loop && prog_rom_data[63:32] == 32'd0) begin
                        cur_param_base <= layer_table_param_base;
                    end else begin
                        cur_param_base <= prog_rom_data[63:32];
                    end

                    if (prog_rom_data[127:120] == OP_HALT) begin
                        busy  <= 1'b0;
                        done  <= 1'b1;
                        state <= S_IDLE;
                    end else begin
                        case (prog_rom_data[127:120])
                            OP_EMBED:     embed_start  <= 1'b1;
                            OP_RMSNORM:   norm_start   <= 1'b1;
                            OP_LINEAR:    linear_start <= 1'b1;
                            OP_ROPE:      rope_start   <= 1'b1;
                            OP_ATTENTION: attn_start   <= 1'b1;
                            OP_ADD:       add_start    <= 1'b1;
                            OP_SWIGLU:    swiglu_start <= 1'b1;
                            OP_ARGMAX:    argmax_start <= 1'b1;
                            default:      ;
                        endcase
                        state <= S_DISPATCH;
                    end
                end

                S_DISPATCH: begin
                    // Start pulses were 1 cycle high; now wait for execution
                    state <= S_WAIT_UNIT;
                end

                S_WAIT_UNIT: begin
                    if (active_unit_done) begin
                        if (cur_opcode == OP_ARGMAX) begin
                            tok_out       <= argmax_res_tok;
                            tok_out_valid <= 1'b1;
                        end

                        // Program Counter and Layer Loop Progression
                        if (cur_pc == layer_loop_end) begin
                            if (cur_layer + 1'b1 < num_layers) begin
                                cur_layer <= cur_layer + 1'b1;
                                cur_pc    <= layer_loop_start;
                            end else begin
                                cur_layer <= {LAYER_W{1'b0}};
                                cur_pc    <= cur_pc + 1'b1;
                            end
                        end else begin
                            cur_pc <= cur_pc + 1'b1;
                        end
                        state <= S_FETCH;
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
