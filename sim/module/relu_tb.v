`timescale 1ns / 1ps

module relu_tb;

    reg signed [23:0] conv_in;
    wire [7:0] relu_out;

    ReLU UUT (
        .conv_in(conv_in),
        .relu_out(relu_out)
    );

    integer pass_count;

    initial begin
        pass_count = 0;

        conv_in = 24'sd100;
        #10;
        if (relu_out == 8'd100) pass_count = pass_count + 1;
        $display("Input=%d Output=%d (expected 100)", conv_in, relu_out);

        conv_in = -24'sd50;
        #10;
        if (relu_out == 8'd0) pass_count = pass_count + 1;
        $display("Input=%d Output=%d (expected 0)", conv_in, relu_out);

        conv_in = 24'sd0;
        #10;
        if (relu_out == 8'd0) pass_count = pass_count + 1;
        $display("Input=%d Output=%d (expected 0)", conv_in, relu_out);

        conv_in = 24'sd255;
        #10;
        if (relu_out == 8'd255) pass_count = pass_count + 1;
        $display("Input=%d Output=%d (expected 255)", conv_in, relu_out);

        conv_in = 24'sd256;
        #10;
        if (relu_out == 8'd255) pass_count = pass_count + 1;
        $display("Input=%d Output=%d (expected 255 saturated)", conv_in, relu_out);

        conv_in = 24'sd1000;
        #10;
        if (relu_out == 8'd255) pass_count = pass_count + 1;
        $display("Input=%d Output=%d (expected 255 saturated)", conv_in, relu_out);

        conv_in = -24'sd1;
        #10;
        if (relu_out == 8'd0) pass_count = pass_count + 1;
        $display("Input=%d Output=%d (expected 0)", conv_in, relu_out);

        $display("%0d/7 tests passed", pass_count);
        if (pass_count == 7) $display("PASS");
        else $display("FAIL");

        #10 $finish;
    end

endmodule
