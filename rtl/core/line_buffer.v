module line_buffer(

    input [7:0] pixel_in,
    input clk,
    input pixel_valid,
    input reset,
    input clear,            // pulse high to flush stale data between channels
    input [7:0] img_width,
    input [10:0] x,
    input [9:0] y,
    
    output reg [7:0] row0,
    output reg [7:0] row1,
    output reg [7:0] row2,
    output reg [10:0] x_reg,
    output reg [9:0] y_reg

);


parameter WIDTH = 128;

reg [7:0] pix_buffer0 [0:WIDTH-1'b1];
reg [7:0] pix_buffer1 [0:WIDTH-1'b1]; 

reg [$clog2(WIDTH) - 1'b1 : 0] ctr;

integer ci;

always@(posedge clk or negedge reset)
begin

    if(!reset)
    begin
        ctr <= 1'b0;
        row0 <= 8'd0;
        row1 <= 8'd0;
        row2 <= 8'd0;
    end
    else if (clear) begin
        // Reset counter and clear stale row data between channels/layers
        ctr <= 1'b0;
        row0 <= 8'd0;
        row1 <= 8'd0;
        row2 <= 8'd0;
        for (ci = 0; ci < WIDTH; ci = ci + 1) begin
            pix_buffer0[ci] <= 8'd0;
            pix_buffer1[ci] <= 8'd0;
        end
    end
    else if (pixel_valid) begin    
        row0 <= pix_buffer0[ctr];
        row1 <= pix_buffer1[ctr];
        row2 <= pixel_in;
        
        x_reg <= x;
        y_reg <= y;
           
        pix_buffer0[ctr] <= pixel_in;
        pix_buffer1[ctr] <= pix_buffer0[ctr];
    
        if (ctr == img_width-1'b1)
            ctr <= 0;
        else
            ctr <= ctr + 1'b1;
        end
end

endmodule