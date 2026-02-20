module conv_core(
    input clk,
    input reset,
    input window_valid, 
    input fsm_window_valid,
    input [10:0] x,
    input [9:0] y,
    
    input [7:0] w00, w01, w02,
    input [7:0] w10, w11, w12,
    input [7:0] w20, w21, w22,

    input signed [7:0] K00, K01, K02,
    input signed [7:0] K10, K11, K12,
    input signed [7:0] K20, K21, K22,
        
    output reg signed [19:0] conv_out,
    output reg conv_valid,
    output reg [10:0] x_regcc,
    output reg [9:0] y_regcc
    );
    
    reg signed [15:0] m1, m2, m3, m4, m5, m6, m7, m8, m9;
    reg signed [19:0] row_sum1, row_sum2, row_sum3;
    reg valid_1, valid_2, fsm_window_valid_stg1, fsm_window_valid_stg2;
    reg [10:0] x_stg1, x_stg2, x_buff;
    reg [9:0] y_stg1, y_stg2, y_buff;
    
    reg signed [8:0] w01_s, w00_s, w02_s, w10_s, w12_s, w11_s, w22_s, w20_s, w21_s;
    
 always@(*)
 begin   
     w00_s = {1'b0, w00};
     w01_s = {1'b0, w01};
     w02_s = {1'b0, w02};
     w10_s = {1'b0, w10};
     w11_s = {1'b0, w11};
     w12_s = {1'b0, w12};
     w20_s = {1'b0, w20};
     w21_s = {1'b0, w21};
     w22_s = {1'b0, w22};
 end 
        
    always@(posedge clk or negedge reset)
    begin
    
    if (!reset) begin
        conv_out <= 20'd0;
        valid_1  <= 1'b0;
        valid_2  <= 1'b0;
        conv_valid <= 1'b0;
        
        m1 <= 16'd0;
        m2 <= 16'd0;
        m3 <= 16'd0;
        m4 <= 16'd0;
        m5 <= 16'd0;
        m6 <= 16'd0;
        m7 <= 16'd0;
        m8 <= 16'd0;
        m9 <= 16'd0;
        
        row_sum1 <= 20'd0;
        row_sum2 <= 20'd0; 
        row_sum3 <= 20'd0;      
        
    end
    else begin
   
    if (window_valid && fsm_window_valid) begin
        m1 <= w00_s*K00;
        m2 <= w10_s*K10;
        m3 <= w20_s*K20;
        m4 <= w01_s*K01;
        m5 <= w11_s*K11;
        m6 <= w21_s*K21;
        m7 <= w02_s*K02;
        m8 <= w12_s*K12;
        m9 <= w22_s*K22;
        
        x_stg1 <= x;
        y_stg1 <= y;
        fsm_window_valid_stg1 <= fsm_window_valid;
        
    end
    
    valid_1 <= window_valid && fsm_window_valid;
    
    if(valid_1 && fsm_window_valid_stg1)
    begin       
        row_sum1 <= m1 + m4 + m7;
        row_sum2 <= m2 + m5 + m8; 
        row_sum3 <= m3 + m6 + m9;
        
        x_stg2 <= x_stg1;
        y_stg2 <= y_stg1;
        
    end
    
    valid_2 <= valid_1 && fsm_window_valid_stg1;
    fsm_window_valid_stg2 <= fsm_window_valid_stg1;
    
    if(valid_2 && fsm_window_valid_stg2)
    begin   
        conv_out <= row_sum1 + row_sum2 + row_sum3;
        x_regcc <= x_stg2;
        y_regcc <= y_stg2;
        
    end 
    
    conv_valid <= valid_2 && fsm_window_valid_stg2;
    
               
    end 
    
    end 
    
    
      
endmodule
