`timescale 1ns / 1ps

module weight_bram_tb;

    reg clk, reset, write_en;
    reg [7:0] data_in;
    reg [14:0] read_addr, write_addr;
    wire [7:0] data_out;

    weight_bram #(.DEPTH(256)) UUT (
        .clk(clk), .reset(reset),
        .data_in(data_in),
        .read_addr(read_addr),
        .write_addr(write_addr),
        .write_en(write_en),
        .data_out(data_out)
    );

    always #10 clk = ~clk;

    integer i, pass_count;

    initial begin
        clk = 0; reset = 0; write_en = 0;
        data_in = 0; read_addr = 0; write_addr = 0;
        pass_count = 0;
        #25 reset = 1;

        for (i = 0; i < 16; i = i + 1) begin
            @(posedge clk);
            write_addr = i;
            data_in = (i * 17 + 3) & 8'hFF;
            write_en = 1;
        end
        @(posedge clk);
        write_en = 0;

        #20;

        for (i = 0; i < 16; i = i + 1) begin
            @(posedge clk);
            read_addr = i;
            @(posedge clk);
            #1;
            if (data_out == ((i * 17 + 3) & 8'hFF))
                pass_count = pass_count + 1;
            else
                $display("FAIL: addr=%0d expected=%0d got=%0d",
                         i, (i * 17 + 3) & 8'hFF, data_out);
        end

        $display("%0d/16 read-back tests passed", pass_count);
        if (pass_count == 16) $display("PASS");
        else $display("FAIL");

        #50 $finish;
    end

endmodule
