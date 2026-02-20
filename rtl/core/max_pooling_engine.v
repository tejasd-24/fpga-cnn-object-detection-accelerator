module max_pooling_engine(
    input clk,
    input reset,
    input [7:0] relu_in,
    input [10:0] x,
    input [9:0] y,

    output reg [7:0] pool_out,
    output reg pool_valid

    );

    parameter WIDTH = 128;

    reg [7:0] col_max_buffer [WIDTH - 1 : 0];
    reg [7:0] odd_row_buffer;

always@(posedge clk or negedge reset)
begin
    if(!reset)
    begin
        odd_row_buffer <= 1'b0;
        pool_valid <= 1'b0;
    end

    else if(!x[0] && !y[0])
    begin
        col_max_buffer[x] <= relu_in;
        pool_valid <= 1'b0;
    end

    else if(x[0] && !y[0])
    begin
        if(relu_in > col_max_buffer[x-1'b1])
        begin
            col_max_buffer[x >> 1] <= relu_in;
        end

        else col_max_buffer[x >> 1] <= col_max_buffer[x-1'b1];
        pool_valid <= 1'b0;
    end

    else if(!x[0] && y[0]) 
    begin
        odd_row_buffer <= relu_in;
        pool_valid <= 1'b0;
    end
    
    else if(x[0] && y[0])
    begin
        if(relu_in > odd_row_buffer && relu_in > col_max_buffer[x>>1])
            pool_out <= relu_in;

            else if(odd_row_buffer > relu_in && odd_row_buffer > col_max_buffer[x>>1])
                 pool_out <= odd_row_buffer;

            else pool_out <= col_max_buffer[x>>1];

        pool_valid <= 1'b1;
    end
    
    end

endmodule
