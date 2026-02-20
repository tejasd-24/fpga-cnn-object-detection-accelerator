`timescale 1ns / 1ps

module sliding_window_tb;

    reg clk, reset, pixel_valid;
    reg [7:0] row0_in, row1_in, row2_in;
    reg [10:0] x;
    reg [9:0] y;
    wire [7:0] w00, w01, w02, w10, w11, w12, w20, w21, w22;
    wire [10:0] x_regsw;
    wire [9:0] y_regsw;
    wire window_valid;

    sliding_window UUT (
        .clk(clk), .reset(reset),
        .pixel_valid(pixel_valid),
        .row0_in(row0_in), .row1_in(row1_in), .row2_in(row2_in),
        .x(x), .y(y),
        .w00(w00), .w01(w01), .w02(w02),
        .w10(w10), .w11(w11), .w12(w12),
        .w20(w20), .w21(w21), .w22(w22),
        .x_regsw(x_regsw), .y_regsw(y_regsw),
        .window_valid(window_valid)
    );

    always #10 clk = ~clk;

    integer i;

    initial begin
        clk = 0; reset = 0; pixel_valid = 0;
        row0_in = 0; row1_in = 0; row2_in = 0;
        x = 0; y = 0;
        #25 reset = 1;

        for (i = 0; i < 16; i = i + 1) begin
            @(posedge clk);
            pixel_valid = 1;
            row0_in = i * 3 + 1;
            row1_in = i * 3 + 2;
            row2_in = i * 3 + 3;
            x = i;
            y = 2;
        end

        @(posedge clk);
        pixel_valid = 0;

        #200;

        if (window_valid)
            $display("PASS: window_valid asserted");

        $display("Window: [%d %d %d; %d %d %d; %d %d %d]",
                 w00, w01, w02, w10, w11, w12, w20, w21, w22);

        #50 $finish;
    end

endmodule
