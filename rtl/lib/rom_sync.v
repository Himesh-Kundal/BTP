// rom_sync.v - Parameterized Synchronous ROM Module Template
// Contract references: docs/part2/02_architecture_and_contracts.md §6
//
// Immutable weights property:
// - Module has ONLY clk, addr, and data_out ports.
// - Initialized from compiler-generated hex via $readmemh in initial block.
// - Zero write ports, zero write enable, zero runtime modification paths.

`timescale 1ns / 1ps

module rom_sync #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 16,
    parameter DEPTH      = 65536,
    parameter INIT_FILE  = ""
)(
    input  wire                  clk,
    input  wire [ADDR_WIDTH-1:0] addr,
    output reg  [DATA_WIDTH-1:0] data_out
);

    // Memory array
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, mem);
        end
    end

    always @(posedge clk) begin
        data_out <= mem[addr];
    end

endmodule
