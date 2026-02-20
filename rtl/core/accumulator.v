module accumulator(
    input clk,
    input reset,
    input [11:0] addr,
    input signed [19:0] data_in,

    input read_en,
    input write_en,
    input ow_add,

    output reg signed [23:0] data_out,
    output reg data_valid
    );

reg signed [23:0] accumulator [0:4095];


reg signed [23:0] acc_read;
always @(posedge clk) begin
    acc_read <= accumulator[addr];
end

reg [11:0] addr_d;
reg write_en_d, ow_add_d;
reg signed [19:0] data_in_d;

always @(posedge clk) begin
    addr_d <= addr;
    write_en_d <= write_en;
    ow_add_d <= ow_add;
    data_in_d <= data_in;
end

always @(posedge clk) begin
    if (write_en_d && !ow_add_d)
        accumulator[addr_d] <= acc_read + data_in_d;
    else if (write_en_d && ow_add_d)
        accumulator[addr_d] <= data_in_d;
end

always @(posedge clk or negedge reset) begin
    if (!reset) begin
        data_out <= 24'd0;
        data_valid <= 1'b0;
    end
    else if (read_en) begin
        data_out <= acc_read;
        data_valid <= 1'b1;
    end
    else
        data_valid <= 1'b0;
end

endmodule

