`timescale 1ns / 1ps

module conv_core_tb;

    reg clk, reset, window_valid, fsm_window_valid;
    reg [10:0] x;
    reg [9:0] y;
    reg [7:0] w00, w01, w02, w10, w11, w12, w20, w21, w22;
    reg signed [7:0] K00, K01, K02, K10, K11, K12, K20, K21, K22;
    wire signed [19:0] conv_out;
    wire conv_valid;
    wire [10:0] x_regcc;
    wire [9:0] y_regcc;

    conv_core UUT (
        .clk(clk), .reset(reset),
        .window_valid(window_valid),
        .fsm_window_valid(fsm_window_valid),
        .x(x), .y(y),
        .w00(w00), .w01(w01), .w02(w02),
        .w10(w10), .w11(w11), .w12(w12),
        .w20(w20), .w21(w21), .w22(w22),
        .K00(K00), .K01(K01), .K02(K02),
        .K10(K10), .K11(K11), .K12(K12),
        .K20(K20), .K21(K21), .K22(K22),
        .conv_out(conv_out), .conv_valid(conv_valid),
        .x_regcc(x_regcc), .y_regcc(y_regcc)
    );

    always #10 clk = ~clk;

    initial begin
        clk = 0; reset = 0;
        window_valid = 0; fsm_window_valid = 0;
        w00 = 0; w01 = 0; w02 = 0;
        w10 = 0; w11 = 0; w12 = 0;
        w20 = 0; w21 = 0; w22 = 0;
        K00 = 0; K01 = 0; K02 = 0;
        K10 = 0; K11 = 0; K12 = 0;
        K20 = 0; K21 = 0; K22 = 0;
        x = 0; y = 0;
        #25 reset = 1;

        @(posedge clk);
        w00 = 10; w01 = 20; w02 = 30;
        w10 = 40; w11 = 50; w12 = 60;
        w20 = 70; w21 = 80; w22 = 90;
        K00 = 1; K01 = 1; K02 = 1;
        K10 = 1; K11 = 1; K12 = 1;
        K20 = 1; K21 = 1; K22 = 1;
        window_valid = 1;
        fsm_window_valid = 1;
        x = 5; y = 5;

        @(posedge clk);
        window_valid = 0;
        fsm_window_valid = 0;

        #100;

        wait(conv_valid);
        $display("Conv output = %d (expected 450)", conv_out);
        if (conv_out == 450)
            $display("PASS");
        else
            $display("FAIL");

        #50 $finish;
    end

endmodule
