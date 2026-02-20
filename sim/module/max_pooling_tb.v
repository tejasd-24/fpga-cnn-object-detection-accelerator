`timescale 1ns / 1ps

module max_pooling_tb;

    reg clk, reset;
    reg [7:0] relu_in;
    reg [10:0] x;
    reg [9:0] y;
    wire [7:0] pool_out;
    wire pool_valid;

    max_pooling_engine #(.WIDTH(4)) UUT (
        .clk(clk), .reset(reset),
        .relu_in(relu_in),
        .x(x), .y(y),
        .pool_out(pool_out),
        .pool_valid(pool_valid)
    );

    always #10 clk = ~clk;

    reg [7:0] test_data [0:15];
    integer i, pool_count;

    initial begin
        clk = 0; reset = 0;
        relu_in = 0; x = 0; y = 0;
        pool_count = 0;

        test_data[0]  = 10; test_data[1]  = 20;
        test_data[2]  = 30; test_data[3]  = 40;
        test_data[4]  = 50; test_data[5]  = 60;
        test_data[6]  = 70; test_data[7]  = 80;
        test_data[8]  = 15; test_data[9]  = 25;
        test_data[10] = 35; test_data[11] = 45;
        test_data[12] = 55; test_data[13] = 65;
        test_data[14] = 75; test_data[15] = 85;

        #25 reset = 1;

        for (i = 0; i < 16; i = i + 1) begin
            @(posedge clk);
            x = i % 4;
            y = i / 4;
            relu_in = test_data[i];
        end

        repeat (10) begin
            @(posedge clk);
            if (pool_valid) begin
                $display("Pool output[%0d] = %d", pool_count, pool_out);
                pool_count = pool_count + 1;
            end
        end

        $display("Total pool outputs: %0d", pool_count);
        #50 $finish;
    end

endmodule
