module sliding_window(

    input clk,
    input reset,
    input pixel_valid,
    input [10:0] x,
    input [9:0] y,

//Inputs from the line buffer

    input [7:0] row0_in,
    input [7:0] row1_in,
    input [7:0] row2_in,
       
//Outputs from sliding window    
    output [7:0] w00,
    output [7:0] w10,
    output [7:0] w20,
    
    output [7:0] w01,
    output [7:0] w11,
    output [7:0] w21,
    
    output [7:0] w02,
    output [7:0] w12,
    output [7:0] w22,
    output reg [10:0] x_regsw,
    output reg [9:0] y_regsw,
    output reg window_valid 
    );
    
    reg [7:0] row0_buff;
    reg [7:0] row1_buff;
    reg [7:0] row2_buff;
    
    reg [7:0] row0_buff_stg2;
    reg [7:0] row1_buff_stg2;
    reg [7:0] row2_buff_stg2;
   
    
    
    
 always@(posedge clk or negedge reset)
 begin
 
 if(!reset)
 begin
    row0_buff <= 1'b0;
    row1_buff <= 1'b0;
    row2_buff <= 1'b0;        
    row0_buff_stg2 <= 1'b0;
    row1_buff_stg2 <= 1'b0;
    row2_buff_stg2 <= 1'b0;
 end

 else if(pixel_valid)
 begin

    row0_buff <= row0_in;
    row1_buff <= row1_in;
    row2_buff <= row2_in;
        
    row0_buff_stg2 <= row0_buff;
    row1_buff_stg2 <= row1_buff;
    row2_buff_stg2 <= row2_buff;
    
    x_regsw <= x;
    y_regsw <= y;
   
    
    window_valid <= pixel_valid;
 end
    
 end
 
 assign w02 = (!reset) ? 0 : row0_in;
 assign w12 = (!reset) ? 0 : row1_in;
 assign w22 = (!reset) ? 0 : row2_in;
    
 assign w01 = (!reset) ? 0 : row0_buff;
 assign w11 = (!reset) ? 0 : row1_buff;
 assign w21 = (!reset) ? 0 : row2_buff;
    
 assign w00 = (!reset) ? 0 : row0_buff_stg2;
 assign w10 = (!reset) ? 0 : row1_buff_stg2;
 assign w20 = (!reset) ? 0 : row2_buff_stg2; 
    
endmodule
