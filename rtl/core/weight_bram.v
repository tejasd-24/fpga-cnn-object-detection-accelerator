module weight_bram(
    input clk,
    input reset,
    input [7:0] data_in,
    input [14:0] read_addr,
    input [14:0] write_addr,
    input write_en,

    output reg [7:0] data_out

    );

    parameter DEPTH = 23184;

    reg [7:0] bram_buff [DEPTH-1'b1 : 0];

    always @(posedge clk) begin
        if (write_en)
            bram_buff[write_addr] <= data_in;
    end

    always @(posedge clk) begin
        data_out <= bram_buff[read_addr];
    end

endmodule