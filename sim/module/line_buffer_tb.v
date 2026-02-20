`timescale 1ns / 1ps

module line_buffer_tb;

    reg clk, reset, pixel_valid, clear;
    reg [7:0] pixel_in, img_width;
    reg [10:0] x;
    reg [9:0] y;
    wire [7:0] row0, row1, row2;
    wire [10:0] x_reg;
    wire [9:0] y_reg;

    line_buffer #(.WIDTH(8)) UUT (
        .clk(clk), .reset(reset),
        .pixel_in(pixel_in), .pixel_valid(pixel_valid),
        .clear(clear), .img_width(img_width),
        .x(x), .y(y),
        .row0(row0), .row1(row1), .row2(row2),
        .x_reg(x_reg), .y_reg(y_reg)
    );

    always #10 clk = ~clk;

    integer i;

    initial begin
        clk = 0; reset = 0; pixel_valid = 0; clear = 0;
        pixel_in = 0; img_width = 8; x = 0; y = 0;
        #25 reset = 1;

        #20 clear = 1;
        #20 clear = 0;

        for (i = 0; i < 32; i = i + 1) begin
            @(posedge clk);
            pixel_in = i + 1;
            pixel_valid = 1;
            x = i % 8;
            y = i / 8;
        end

        @(posedge clk);
        pixel_valid = 0;

        #100;

        clear = 1;
        @(posedge clk);
        clear = 0;

        #20;
        if (row0 == 0 && row1 == 0 && row2 == 0)
            $display("PASS: clear works");
        else
            $display("FAIL: clear did not reset rows");

        #100 $finish;
    end

endmodule
