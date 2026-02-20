module cnn_acc_top(
    input clk,
    input reset,
    input start,

    input [7:0] axi_weight_data,
    input [14:0] axi_weight_addr,
    input axi_weight_en,

    input [7:0] input_pixel,
    input [13:0] input_addr,
    input input_wen,

    input [6:0] output_ch,
    input [11:0] output_addr,
    output [7:0] output_data,

    input [4:0] relu_shift_l0,
    input [4:0] relu_shift_l1,
    input [4:0] relu_shift_l2,

    output busy,
    output done,
    output [1:0] current_layer
);

    wire [13:0] read_addr_fsm;
    wire [11:0] write_addr_fsm;
    wire write_en_fsm;
    wire [5:0] fm_read_channel;
    wire [5:0] fm_write_channel;
    wire signed [7:0] fm_pixel_out;
    wire [14:0] wbram_read;
    wire signed [7:0] weight_data_out;
    wire [3:0] weight_core_sel, weight_idx;
    wire weight_load_en;
    wire pixel_valid;
    wire [11:0] accumulator_addr;
    wire read_en, write_en, ow_add;
    wire [7:0] pool_data_out_fsm;
    wire conv_valid_w;
    wire pool_valid_w;
    wire drain_start;

    wire [7:0] img_width = (current_layer == 0) ? 8'd128 :
                           (current_layer == 1) ? 7'd64  : 7'd32;

    wire [6:0] read_offset  = (current_layer == 0) ? 7'd0  :
                              (current_layer == 1) ? 7'd0  : 7'd16;
    wire [6:0] write_offset = (current_layer == 0) ? 7'd0  :
                              (current_layer == 1) ? 7'd16 : 7'd48;

    wire [6:0] actual_read_ch  = read_offset + fm_read_channel;
    wire [6:0] actual_write_base = write_offset + fm_write_channel;

    wire [7:0] weight_bram_dout;
    weight_bram WBRAM (
        .clk(clk), .reset(reset),
        .data_in(axi_weight_data),
        .write_addr(axi_weight_addr),
        .write_en(axi_weight_en),
        .read_addr(wbram_read),
        .data_out(weight_bram_dout)
    );

    reg signed [7:0] kernel_w [0:15][0:8];
    reg read_en_d1, read_en_d2;
    // Initialize for simulation
    integer ki, kj;
    initial begin
        for (ki = 0; ki < 16; ki = ki + 1)
            for (kj = 0; kj < 9; kj = kj + 1)
                kernel_w[ki][kj] = 8'sd0;
    end
    always @(posedge clk) begin
        if (weight_load_en)
            kernel_w[weight_core_sel][weight_idx] <= weight_data_out;
    end

    reg [7:0] input_bram [0:16383];
    reg [7:0] input_bram_dout;
    always @(posedge clk) begin
        if (input_wen)
            input_bram[input_addr] <= input_pixel;
        input_bram_dout <= input_bram[read_addr_fsm];
    end

    wire [7:0] fm_dout [0:111];
    wire [7:0] fm_read_pixel_mux = (current_layer == 0) ? input_bram_dout :
                                                           fm_dout[actual_read_ch];

    reg [10:0] pixel_x;
    reg [9:0] pixel_y;
    reg pixel_valid_prev;
    always @(posedge clk or negedge reset) begin
        if (!reset) begin
            pixel_x <= 0;
            pixel_y <= 0;
            pixel_valid_prev <= 0;
        end else begin
            pixel_valid_prev <= pixel_valid;
            if (pixel_valid && !pixel_valid_prev) begin
                // Only reset coords when starting a genuinely new pixel stream
                // For layer 0 tiling: don't reset between tiles (pixel resumes mid-image)
                if (current_layer != 0 || (pixel_x == 0 && pixel_y == 0)) begin
                    pixel_x <= 0;
                    pixel_y <= 0;
                end
            end else if (pixel_valid) begin
                if (pixel_x == img_width - 1) begin
                    pixel_x <= 0;
                    pixel_y <= pixel_y + 1;
                end else
                    pixel_x <= pixel_x + 1;
            end
        end
    end

    reg [10:0] drain_x;
    reg [9:0] drain_y;
    reg read_en_prev;
    always @(posedge clk or negedge reset) begin
        if (!reset) begin
            drain_x <= 0;
            drain_y <= 0;
            read_en_prev <= 0;
        end else begin
            read_en_prev <= read_en;
            if (drain_start) begin
                drain_x <= 0;
                drain_y <= 0;
            end else if (read_en && !read_en_prev) begin
                // Non-tiled layers: reset on rising edge of read_en
                if (current_layer != 0) begin
                    drain_x <= 0;
                    drain_y <= 0;
                end
            end else if (read_en) begin
                if (drain_x == img_width - 1) begin
                    drain_x <= 0;
                    drain_y <= drain_y + 1;
                end else
                    drain_x <= drain_x + 1;
            end
        end
    end

    reg [11:0] pool_write_addr;
    always @(posedge clk or negedge reset) begin
        if (!reset)
            pool_write_addr <= 0;
        else if (drain_start)
            pool_write_addr <= 0;
        else if (read_en && !read_en_prev && current_layer != 0)
            pool_write_addr <= 0;
        else if (pool_valid_w && read_en_d2)
            pool_write_addr <= pool_write_addr + 1;
    end

    // Delay drain coordinates by 1 cycle to align with accumulator read latency
    reg [10:0] drain_x_d;
    reg [9:0]  drain_y_d;
    always @(posedge clk or negedge reset) begin
        if (!reset) begin
            drain_x_d <= 0;
            drain_y_d <= 0;
        end else begin
            drain_x_d <= drain_x;
            drain_y_d <= drain_y;
        end
    end

    // Delay read_en by 2 cycles to match accumulator→ReLU→pool pipeline latency.
    // This gates pool writes so they only occur for legitimate drain data,
    // preventing spurious pool_valid during PROCESS from overwriting BRAM.
    always @(posedge clk or negedge reset) begin
        if (!reset) begin
            read_en_d1 <= 0;
            read_en_d2 <= 0;
        end else begin
            read_en_d1 <= read_en;
            read_en_d2 <= read_en_d1;
        end
    end

    wire [7:0] row0, row1, row2;
    wire [10:0] x_lb;
    wire [9:0] y_lb;
    // Clear line buffer when loading weights (start of each new channel)
    wire lb_clear = weight_load_en;

    line_buffer LB (
        .pixel_in(fm_read_pixel_mux), .clk(clk),
        .pixel_valid(pixel_valid), .reset(reset),
        .clear(lb_clear),
        .img_width(img_width),
        .x(pixel_x), .y(pixel_y),
        .row0(row0), .row1(row1), .row2(row2),
        .x_reg(x_lb), .y_reg(y_lb)
    );

    wire [7:0] w00, w01, w02, w10, w11, w12, w20, w21, w22;
    wire window_valid;
    wire [10:0] x_sw;
    wire [9:0] y_sw;
    sliding_window SW (
        .clk(clk), .reset(reset),
        .pixel_valid(pixel_valid),
        .row0_in(row0), .row1_in(row1), .row2_in(row2),
        .x(x_lb), .y(y_lb),
        .x_regsw(x_sw), .y_regsw(y_sw),
        .window_valid(window_valid),
        .w00(w00), .w01(w01), .w02(w02),
        .w10(w10), .w11(w11), .w12(w12),
        .w20(w20), .w21(w21), .w22(w22)
    );

    wire signed [19:0] conv_out_arr [0:15];
    wire conv_valid_arr [0:15];

    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : conv_gen
            conv_core CORE (
                .clk(clk), .reset(reset),
                .window_valid(window_valid),
                .fsm_window_valid(pixel_valid),
                .x(x_sw), .y(y_sw),
                .w00(w00), .w01(w01), .w02(w02),
                .w10(w10), .w11(w11), .w12(w12),
                .w20(w20), .w21(w21), .w22(w22),
                .K00(kernel_w[i][0]), .K01(kernel_w[i][1]), .K02(kernel_w[i][2]),
                .K10(kernel_w[i][3]), .K11(kernel_w[i][4]), .K12(kernel_w[i][5]),
                .K20(kernel_w[i][6]), .K21(kernel_w[i][7]), .K22(kernel_w[i][8]),
                .conv_out(conv_out_arr[i]),
                .conv_valid(conv_valid_arr[i]),
                .x_regcc(), .y_regcc()
            );
        end
    endgenerate

    assign conv_valid_w = conv_valid_arr[0];

    wire signed [23:0] accum_out [0:15];
    wire accum_dv [0:15];

    generate
        for (i = 0; i < 16; i = i + 1) begin : accum_gen
            accumulator ACCUM (
                .clk(clk), .reset(reset),
                .addr(accumulator_addr),
                .data_in(conv_out_arr[i]),
                .read_en(read_en),
                .write_en(write_en),
                .ow_add(ow_add),
                .data_out(accum_out[i]),
                .data_valid(accum_dv[i])
            );
        end
    endgenerate

    // Per-layer right-shift to prevent ReLU saturation
    // Shift values set via AXI registers from software
    wire [4:0] relu_shift = (current_layer == 0) ? relu_shift_l0 :
                            (current_layer == 1) ? relu_shift_l1 :
                                                   relu_shift_l2;

    wire [7:0] relu_out [0:15];
    generate
        for (i = 0; i < 16; i = i + 1) begin : relu_gen
            ReLU RELU_UNIT (
                .conv_in(accum_out[i] >>> relu_shift),
                .relu_out(relu_out[i])
            );
        end
    endgenerate

    wire [7:0] pool_out_arr [0:15];
    wire pool_valid_arr [0:15];

    generate
        for (i = 0; i < 16; i = i + 1) begin : pool_gen
            max_pooling_engine #(.WIDTH(128)) POOL (
                .clk(clk), .reset(reset),
                .relu_in(relu_out[i]),
                .x(drain_x_d), .y(drain_y_d),
                .pool_out(pool_out_arr[i]),
                .pool_valid(pool_valid_arr[i])
            );
        end
    endgenerate

    assign pool_valid_w = pool_valid_arr[0];

    reg [111:0] fm_wen_pool;
    reg [7:0] fm_din_pool [0:111];
    integer n;
    always @(*) begin
        for (n = 0; n < 112; n = n + 1) begin
            fm_wen_pool[n] = 1'b0;
            fm_din_pool[n] = 8'd0;
        end
        for (n = 0; n < 16; n = n + 1) begin
            if (actual_write_base + n < 112) begin
                fm_wen_pool[actual_write_base + n] = pool_valid_arr[n] && read_en_d2;
                fm_din_pool[actual_write_base + n] = pool_out_arr[n];
            end
        end
    end

    genvar ch;
    generate
        for (ch = 0; ch < 112; ch = ch + 1) begin : fm_gen
            feature_bram #(
                .DEPTH((ch < 16) ? 4096 : (ch < 48) ? 1024 : 256)
            ) FM (
                .clk(clk), .reset(reset),
                .data_in(fm_din_pool[ch]),
                .read_addr(busy ? read_addr_fsm[11:0] : output_addr),
                .write_addr(pool_write_addr),
                .write_en(fm_wen_pool[ch]),
                .data_out(fm_dout[ch])
            );
        end
    endgenerate

    reg [7:0] output_data_r;
    always @(posedge clk) begin
        output_data_r <= fm_dout[output_ch];
    end
    assign output_data = output_data_r;

    layer_fsm FSM (
        .clock(clk), .reset(reset),
        .start(start),
        .conv_valid(conv_valid_w),
        .pool_valid(pool_valid_w),
        .fm_read_pixel(fm_read_pixel_mux),
        .pool_data_in(pool_out_arr[0]),
        .weight_data_in(weight_bram_dout),
        .pool_data_out(pool_data_out_fsm),
        .busy(busy), .done(done),
        .current_layer(current_layer),
        .read_addr_fsm(read_addr_fsm),
        .write_addr_fsm(write_addr_fsm),
        .write_en_fsm(write_en_fsm),
        .fm_read_channel(fm_read_channel),
        .fm_write_channel(fm_write_channel),
        .fm_pixel_out(fm_pixel_out),
        .wbram_read(wbram_read),
        .weight_data_out(weight_data_out),
        .weight_core_sel(weight_core_sel),
        .weight_idx(weight_idx),
        .weight_load_en(weight_load_en),
        .pixel_valid(pixel_valid),
        .accumulator_addr(accumulator_addr),
        .read_en(read_en),
        .write_en(write_en),
        .ow_add(ow_add),
        .drain_start(drain_start)
    );

endmodule
