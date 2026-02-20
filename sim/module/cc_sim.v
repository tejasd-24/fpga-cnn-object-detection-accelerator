`timescale 1ns / 1ps

module conv_core_tb();

  
    reg clk;
    reg reset;
    reg window_valid;
    reg fsm_window_valid;
    reg [10:0] x;
    reg [9:0] y;
    reg signed [7:0] w00, w01, w02, w10, w11, w12, w20, w21, w22;

    
    wire signed [19:0] conv_out;
    wire conv_valid;
    wire [10:0] x_regcc;
    wire [9:0] y_regcc;


    conv_core uut (
        .clk(clk), .reset(reset),
        .window_valid(window_valid), .fsm_window_valid(fsm_window_valid),
        .x(x), .y(y),
        .w00(w00), .w01(w01), .w02(w02),
        .w10(w10), .w11(w11), .w12(w12),
        .w20(w20), .w21(w21), .w22(w22),
        .conv_out(conv_out), .conv_valid(conv_valid),
        .x_regcc(x_regcc), .y_regcc(y_regcc)
    );

   
    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        
        reset = 1'b0; 
        window_valid = 1'b0;
        fsm_window_valid = 1'b0;
        x = 0; y = 0;
        w00=0; w01=0; w02=0; w10=0; w11=0; w12=0; w20=0; w21=0; w22=0;

       
        #20 reset = 1'b1;
        #20;


        @(posedge clk);
        window_valid = 1'b1;
        fsm_window_valid = 1'b1;
        x = 11'd100; y = 10'd100;
        w00=200; w10=200; w20=200; 
        w01=0;   w11=0;   w21=0; 
        w02=10;  w12=10;  w22=10; 


        repeat(5) @(posedge clk);
        
    
        $display("Test 1 Result: %d (Expected -570)", conv_out);
        $display("Test 1 X_coord: %d (Expected 100)", x_regcc);
        

        #100;
        $finish;
    end
endmodule