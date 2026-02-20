`timescale 1ns / 1ps

module accumulator_tb;

    reg clk, reset;
    reg [11:0] addr;
    reg signed [19:0] data_in;
    reg read_en, write_en, ow_add;
    wire signed [23:0] data_out;
    wire data_valid;

    accumulator UUT (
        .clk(clk), .reset(reset),
        .addr(addr), .data_in(data_in),
        .read_en(read_en), .write_en(write_en),
        .ow_add(ow_add),
        .data_out(data_out), .data_valid(data_valid)
    );

    always #10 clk = ~clk;

    initial begin
        clk = 0; reset = 0;
        addr = 0; data_in = 0;
        read_en = 0; write_en = 0; ow_add = 0;
        #25 reset = 1;

        @(posedge clk);
        addr = 12'd100;
        data_in = 20'd500;
        write_en = 1;
        ow_add = 1;
        @(posedge clk);
        write_en = 0;
        ow_add = 0;

        #40;

        @(posedge clk);
        addr = 12'd100;
        data_in = 20'd300;
        write_en = 1;
        ow_add = 0;
        @(posedge clk);
        write_en = 0;

        #40;

        @(posedge clk);
        addr = 12'd100;
        read_en = 1;
        @(posedge clk);
        @(posedge clk);
        read_en = 0;

        #20;
        $display("Accumulator[100] = %d (expected 800)", data_out);
        if (data_out == 800)
            $display("PASS");
        else
            $display("FAIL");

        #50 $finish;
    end

endmodule
