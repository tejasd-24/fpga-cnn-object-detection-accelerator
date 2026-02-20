module layer_fsm(
    input clock,
    input reset,
    input conv_valid,
    input pool_valid,
    input start,
    input signed [7:0] fm_read_pixel,
    input signed [7:0] pool_data_in,
    input signed [7:0]  weight_data_in,


    output reg [7:0] pool_data_out,

    output reg busy,
    output reg done,
    output [1:0] current_layer,

    output reg [13:0] read_addr_fsm,
    output reg [11:0] write_addr_fsm,
    output reg write_en_fsm,

    output reg [5:0] fm_read_channel,
    output reg [5:0] fm_write_channel,
    output signed [7:0] fm_pixel_out,


    output reg [14:0] wbram_read,

    output signed [7:0]  weight_data_out,
    output reg [3:0]  weight_core_sel,
    output reg [3:0]  weight_idx,
    output reg        weight_load_en,
    output reg pixel_valid,

    output reg [11:0] accumulator_addr,
    output reg read_en,
    output reg write_en,
    output reg ow_add,
    output reg drain_start

    );

    localparam IDLE           =     4'b0000;
    localparam LOAD_WEIGHTS   =     4'b0001;
    localparam PROCESS_PASS   =     4'b0010;
    localparam CHECK_CHANNEL  =     4'b0011;
    localparam DRAIN_ACCUM    =     4'b0100;
    localparam CHECK_BATCH    =     4'b0101;
    localparam CHECK_LAYER    =     4'b0110;
    localparam DONE           =     4'b0111;
    localparam DRAIN_FINISH   =     4'b1000;


    reg [3:0]  state;
    reg [1:0]  layer;          
    reg [2:0]  batch;         
    reg [5:0]  channel;         
    reg [13:0] pixel_addr;      
    reg [7:0]  weight_count;    
    reg [11:0] drain_addr; 
    reg [14:0] weight_base_address;
    reg [3:0]  weight_core_sel_d;
    reg [3:0]  weight_idx_d;
    reg        weight_load_en_d;
    reg [11:0] conv_out_addr;
    reg [1:0]  tile_count;          // which 4096-pixel tile (0-3) for layer 0
 
    wire [14:0] image_size = (layer == 0) ? 15'd16384 : (layer == 1) ? 14'd4096  : 14'd1024;
    wire [5:0] max_channels = (layer == 0) ? 6'd1 : (layer == 1) ? 6'd16 : 6'd32;
    wire [12:0] output_size = (layer == 0) ? 13'd4096 : (layer == 1) ? 13'd1024 : 13'd256;
    wire [2:0] max_batches = (layer == 0) ? 3'd1 : (layer == 1) ? 3'd2 : 3'd4;
    wire [1:0] max_layers = 2'd2;
    // Accumulator tile size: 4096 for all layers (accum depth = 4096)
    wire [11:0] tile_size = (layer == 0) ? 12'd4095 :
                            (layer == 1) ? 12'd4095 : 12'd1023;
                             
    always@(posedge clock or negedge reset)
    begin
        if(!reset)
        begin
            state <= IDLE;
            layer <= 2'd0;
            batch <= 3'd0;
            channel <= 6'd0;
            pixel_addr <= 14'd0;
            weight_count <= 8'd0;
            drain_addr <= 12'd0;
            read_en <= 1'b0;
            write_en <= 1'b0;
            ow_add <= 1'b0;
            pixel_valid <= 1'b0;
            wbram_read <= 15'd0;
            accumulator_addr <= 12'd0;
            weight_load_en <= 1'b0;
            weight_idx <= 4'd0;
            weight_core_sel <= 4'd0;
            fm_write_channel <= 6'd0;
            fm_read_channel <= 6'd0;
            write_en_fsm <= 1'b0;
            write_addr_fsm <= 12'd0;
            read_addr_fsm <= 12'd0;
            weight_base_address <= 15'd0;
            busy <= 1'b0;
            done <= 1'b0;
            pool_data_out <= 8'd0;
            weight_core_sel_d <= 4'd0;
            weight_idx_d <= 4'd0;
            weight_load_en_d <= 1'b0;
            conv_out_addr <= 12'd0;
            tile_count <= 2'd0;
            drain_start <= 1'b0;
        end

        else
        begin
            case(state)

            IDLE            :           begin
                                            busy <= 1'b0;

                                            if(start)
                                            begin
                                                done <= 1'b0;
                                                layer <= 2'b0;
                                                batch <= 3'b0;
                                                channel <= 6'b0;
                                                pixel_addr <= 14'd0;      
                                                weight_count <= 8'd0;    
                                                drain_addr <= 12'd0;
                                                read_en <= 1'b0;
                                                write_en <= 1'b0;
                                                ow_add <= 1'b0;
                                                pixel_valid <= 1'b0;
                                                wbram_read <= 15'd0;
                                                accumulator_addr <= 12'd0;
                                                weight_load_en <= 1'b0;
                                                weight_idx <= 4'd0;
                                                weight_core_sel <= 4'd0;
                                                fm_write_channel <= 6'd0;
                                                fm_read_channel <= 6'd0;
                                                write_en_fsm <= 1'b0;
                                                write_addr_fsm <= 12'd0;
                                                read_addr_fsm <= 12'd0;
                                                busy <= 1'b1;
                                                weight_base_address <= 15'd0;
                                                state <= LOAD_WEIGHTS;
                                                conv_out_addr <= 12'd0;
                                                tile_count <= 2'd0;
                                                drain_start <= 1'b0;
                                            end

                                            else state <= IDLE;
                
                                        end

            LOAD_WEIGHTS    :           begin 
                                            weight_count <= weight_count + 1'd1;
                                            wbram_read <= weight_base_address + weight_count;

                                            weight_core_sel_d <= weight_count / 4'd9;
                                            weight_idx_d <= weight_count % 4'd9;
                                            weight_load_en_d <= (weight_count <= 8'd143) ? 1'b1 : 1'b0;

                                            weight_core_sel <= weight_core_sel_d;
                                            weight_idx <= weight_idx_d;
                                            weight_load_en <= weight_load_en_d;

                                            pixel_valid <= 0;
                                            write_en <= 0;
                                            read_en <= 0;
                                            write_en_fsm <= 0;

                                                if(weight_count == 8'd145)
                                                begin
                                                   pixel_addr <= 14'd0;
                                                   state <= PROCESS_PASS;
                                                   weight_count <= 8'd0;
                                                end

                                                else state <= LOAD_WEIGHTS;
                
                                        end

            PROCESS_PASS    :           begin
                                            busy <= 1'b1;
                                            pixel_valid <= 1'b1;
                                            
                                            fm_read_channel <= channel;
                                            read_addr_fsm <= pixel_addr;
                                            write_en <= conv_valid;
                                            weight_load_en <= 1'b0;
                                            read_en <= 1'b0;
                                            write_en_fsm <= 1'b0;
                                            drain_start <= 1'b0;

                                            if(channel == 6'd0)
                                            begin
                                                ow_add <= 1'b1;
                                            end

                                            else ow_add <= 1'b0;

                                            pixel_addr <= pixel_addr + 1'b1;

                                            // Layer 0 tiling: drain every 4096 pixels
                                            if (layer == 2'd0 && conv_out_addr == tile_size && conv_valid) begin
                                                // Tile boundary reached — go drain
                                                pixel_valid <= 1'b0;
                                                drain_addr <= 12'd0;
                                                conv_out_addr <= 12'd0;
                                                drain_start <= (tile_count == 2'd0) ? 1'b1 : 1'b0;
                                                state <= DRAIN_ACCUM;
                                            end
                                            else if(pixel_addr == image_size - 1'b1)
                                            begin
                                                pixel_addr <= 14'd0;
                                                pixel_valid <= 1'b0;
                                                state <= CHECK_CHANNEL;                                               
                                            end
                                            else state <= PROCESS_PASS;

                                            if (conv_valid)
                                                conv_out_addr <= conv_out_addr + 1;
                                            accumulator_addr <= conv_out_addr;
                                        end

            CHECK_CHANNEL   :           begin
                                            busy <= 1'b1;

                                            read_en <= 1'b0;
                                            write_en <= 1'b0;
                                            weight_load_en <= 1'b0;
                                            write_en_fsm <= 1'b0;
                                            pixel_valid <= 1'b0;

                                            if(channel < max_channels - 1'b1)
                                            begin
                                                channel <= channel + 1'b1;
                                                pixel_addr <= 14'd0;
                                                conv_out_addr <= 12'd0;
                                                weight_count <= 8'd0;
                                                weight_base_address <= weight_base_address + 8'd144;
                                                state <= LOAD_WEIGHTS;

                                            end

                                            else begin state <= DRAIN_ACCUM;
                                                       drain_addr <= 12'd0;
                                                       conv_out_addr <= 12'd0;
                                            end
                
                                        end

            DRAIN_ACCUM     :           begin
                                            read_en <= 1'b1;
                                            accumulator_addr <= drain_addr;
                                            drain_addr <= drain_addr + 1'b1;
                                            pixel_valid <= 1'b0;
                                            weight_load_en <= 1'b0;
                                            write_en <= 1'b0;
                                            write_en_fsm <= pool_valid;
                                            drain_start <= 1'b0;

                                            if (drain_addr == tile_size)
                                            begin
                                                // Layer 0 tiling: if more tiles remain, go back to PROCESS.
                                                // Don't override read_en here — let DRAIN_ACCUM's read_en=1
                                                // carry over for 1 cycle into PROCESS_PASS, providing the
                                                // exit gain that balances the entry loss (exactly 4096 data_valid).
                                                if (layer == 2'd0 && tile_count < 2'd3) begin
                                                    tile_count <= tile_count + 1'b1;
                                                    drain_addr <= 12'd0;
                                                    conv_out_addr <= 12'd0;
                                                    state <= PROCESS_PASS;
                                                end
                                                else begin
                                                    state <= CHECK_BATCH;
                                                    tile_count <= 2'd0;
                                                end
                                            end
                                            else state <= DRAIN_ACCUM;

                                            pool_data_out <= pool_data_in;
                                            fm_write_channel <= batch * 16;
                                            
                                        end

            CHECK_BATCH     :           begin
                                            busy <= 1'b1;

                                            read_en <= 1'b0;
                                            write_en <= 1'b0;
                                            weight_load_en <= 1'b0;
                                            write_en_fsm <= 1'b0;

                                            if(batch < max_batches - 1'b1)
                                            begin
                                                batch <= batch + 1'b1;
                                                channel <= 6'd0;
                                                weight_count <= 8'd0;
                                                weight_base_address <= weight_base_address + 8'd144;
                                                state <= LOAD_WEIGHTS;
                                            end

                                            else state <= CHECK_LAYER;



                                        end

            CHECK_LAYER     :           begin
                                            busy <= 1'b1;

                                            read_en <= 1'b0;
                                            write_en <= 1'b0;
                                            weight_load_en <= 1'b0;
                                            write_en_fsm <= 1'b0;

                                            if(layer < max_layers)
                                            begin
                                                layer <= layer + 1;
                                                batch <= 3'd0;
                                                channel <= 6'd0;
                                                weight_base_address <= weight_base_address + 8'd144;

                                                state <= LOAD_WEIGHTS;
                                            end

                                            else state <= DONE;                                            
                                        end

            DONE            :           begin
                                            busy <= 1'b0;
                                            done <= 1'b1;

                                            if (!start) state <= IDLE;               
                                        end
        endcase
    end
    end

    assign weight_data_out = weight_data_in;
    assign fm_pixel_out = fm_read_pixel;
    assign current_layer = layer;

endmodule
