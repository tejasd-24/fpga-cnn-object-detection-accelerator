`timescale 1ns / 1ps

module layer_fsm_tb;

    // =========================================================
    // Clock, reset, control
    // =========================================================
    reg clock;
    reg reset;
    reg start;
    reg conv_valid;
    reg pool_valid;
    reg signed [7:0] fm_read_pixel;
    reg signed [7:0] pool_data_in;
    reg signed [7:0] weight_data_in;

    // =========================================================
    // Outputs
    // =========================================================
    wire [7:0] pool_data_out;
    wire busy, done;
    wire [1:0] current_layer;
    wire [11:0] read_addr_fsm, write_addr_fsm;
    wire write_en_fsm;
    wire [5:0] fm_read_channel, fm_write_channel;
    wire signed [7:0] fm_pixel_out;
    wire [14:0] wbram_read;
    wire signed [7:0] weight_data_out;
    wire [3:0] weight_core_sel, weight_idx;
    wire weight_load_en;
    wire pixel_valid;
    wire [11:0] accumulator_addr;
    wire read_en, write_en, ow_add;

    // =========================================================
    // Internal state observation (reach into DUT)
    // =========================================================
    wire [2:0] fsm_state = uut.state;
    wire [1:0] fsm_layer = uut.layer;
    wire [2:0] fsm_batch = uut.batch;
    wire [5:0] fsm_channel = uut.channel;

    // State name decoder for display
    reg [12*8-1:0] state_name;
    always @(*) begin
        case (fsm_state)
            3'b000: state_name = "IDLE        ";
            3'b001: state_name = "LOAD_WEIGHT ";
            3'b010: state_name = "PROCESS_PASS";
            3'b011: state_name = "CHECK_CHAN  ";
            3'b100: state_name = "DRAIN_ACCUM ";
            3'b101: state_name = "CHECK_BATCH ";
            3'b110: state_name = "CHECK_LAYER ";
            3'b111: state_name = "DONE        ";
            default: state_name = "UNKNOWN     ";
        endcase
    end

    // =========================================================
    // DUT instantiation
    // =========================================================
    layer_fsm uut (
        .clock(clock),
        .reset(reset),
        .conv_valid(conv_valid),
        .pool_valid(pool_valid),
        .start(start),
        .fm_read_pixel(fm_read_pixel),
        .pool_data_in(pool_data_in),
        .weight_data_in(weight_data_in),
        .pool_data_out(pool_data_out),
        .busy(busy),
        .done(done),
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
        .ow_add(ow_add)
    );

    // =========================================================
    // Simulated Weight BRAM (1-clock read latency)
    // =========================================================
    reg [7:0] fake_weight_bram [0:23183];
    integer i;
    initial begin
        for (i = 0; i < 23184; i = i + 1)
            fake_weight_bram[i] = i[7:0];
    end
    always @(posedge clock) begin
        weight_data_in <= fake_weight_bram[wbram_read];
    end

    // =========================================================
    // Simulated conv_valid: goes high after pipeline delay
    // In real hardware, line_buffer+sliding_window+conv = ~260 clocks
    // For testing we use a simple delay of pixel_valid
    // =========================================================
    reg [9:0] conv_delay_counter;
    always @(posedge clock or negedge reset) begin
        if (!reset) begin
            conv_valid <= 0;
            conv_delay_counter <= 0;
        end else begin
            if (pixel_valid && !conv_valid) begin
                conv_delay_counter <= conv_delay_counter + 1;
                if (conv_delay_counter >= 10'd260)
                    conv_valid <= 1;
            end
            if (!pixel_valid) begin
                // Pipeline drains: keep conv_valid high for ~260 more clocks
                if (conv_valid) begin
                    conv_delay_counter <= conv_delay_counter - 1;
                    if (conv_delay_counter == 0)
                        conv_valid <= 0;
                end
            end
        end
    end

    // =========================================================
    // Simulated pool_valid: high every 4 clocks during drain
    // (MaxPool 2x2 stride 2 produces 1 output per 4 inputs)
    // =========================================================
    reg [1:0] pool_counter;
    always @(posedge clock or negedge reset) begin
        if (!reset) begin
            pool_valid <= 0;
            pool_counter <= 0;
            pool_data_in <= 0;
        end else begin
            if (read_en) begin
                pool_counter <= pool_counter + 1;
                pool_valid <= (pool_counter == 2'd3);
                pool_data_in <= accumulator_addr[7:0]; // fake data
            end else begin
                pool_valid <= 0;
                pool_counter <= 0;
            end
        end
    end

    // =========================================================
    // Clock: 100 MHz
    // =========================================================
    initial clock = 0;
    always #5 clock = ~clock;

    // =========================================================
    // State transition monitor
    // =========================================================
    reg [2:0] prev_state;
    always @(posedge clock) begin
        prev_state <= fsm_state;
        if (fsm_state !== prev_state) begin
            $display("[%0t ns] STATE: %0d -> %0d | layer=%0d batch=%0d channel=%0d",
                     $time, prev_state, fsm_state,
                     fsm_layer, fsm_batch, fsm_channel);
        end
    end

    // =========================================================
    // Test counters for verification
    // =========================================================
    integer load_weight_count;
    integer process_pass_count;
    integer drain_count;
    integer total_clocks;

    always @(posedge clock) begin
        total_clocks <= total_clocks + 1;
        if (fsm_state == 3'b001) load_weight_count <= load_weight_count + 1;
        if (fsm_state == 3'b010) process_pass_count <= process_pass_count + 1;
        if (fsm_state == 3'b100) drain_count <= drain_count + 1;
    end

    // =========================================================
    // Main test
    // =========================================================
    initial begin
        // Init
        reset = 0;
        start = 0;
        fm_read_pixel = 8'd42; // constant test pixel
        load_weight_count = 0;
        process_pass_count = 0;
        drain_count = 0;
        total_clocks = 0;

        $display("============================================");
        $display("  LAYER FSM — FULL STATE MACHINE TEST");
        $display("============================================");
        $display("");

        // Reset
        #100;
        reset = 1;
        #20;

        // =====================================================
        // TEST: Verify IDLE
        // =====================================================
        @(posedge clock);
        @(posedge clock);
        if (fsm_state !== 3'b000) $display("FAIL: Not in IDLE after reset");
        else $display("PASS: In IDLE after reset");

        if (busy !== 0) $display("FAIL: busy not 0 in IDLE");
        else $display("PASS: busy = 0 in IDLE");

        // =====================================================
        // START the FSM
        // =====================================================
        $display("");
        $display(">>> Starting FSM — processing all 3 layers");
        $display(">>> Layer 1: 1 ch, 1 batch, 16384 pixels, 4096 drain");
        $display(">>> Layer 2: 16 ch, 2 batches, 4096 pixels, 1024 drain");
        $display(">>> Layer 3: 32 ch, 4 batches, 1024 pixels, 256 drain");
        $display("");

        @(posedge clock);
        start = 1;
        @(posedge clock);
        @(posedge clock);
        start = 0;

        // =====================================================
        // Wait for DONE (polling loop with timeout)
        // =====================================================
        begin : wait_done_block
            integer timeout;
            timeout = 0;
            while (done !== 1 && timeout < 20_000_000) begin
                @(posedge clock);
                timeout = timeout + 1;
            end

            if (done == 1) begin
                $display("");
                $display("============================================");
                $display("  FSM REACHED DONE STATE!");
                $display("============================================");
            end else begin
                $display("");
                $display("FAIL: TIMEOUT after %0d clocks", timeout);
                $display("  Stuck in state: %0d", fsm_state);
                $display("  Layer=%0d Batch=%0d Channel=%0d",
                         fsm_layer, fsm_batch, fsm_channel);
            end
        end

        // =====================================================
        // Final statistics
        // =====================================================
        $display("");
        $display("--- Statistics ---");
        $display("Total clocks:         %0d", total_clocks);
        $display("LOAD_WEIGHTS clocks:  %0d", load_weight_count);
        $display("PROCESS_PASS clocks:  %0d", process_pass_count);
        $display("DRAIN_ACCUM clocks:   %0d", drain_count);
        $display("Current layer:        %0d", current_layer);
        $display("Done signal:          %0b", done);
        $display("Busy signal:          %0b", busy);

        // Verify final state
        if (done == 1 && busy == 0)
            $display("\nPASS: FSM completed all 3 layers successfully!");
        else
            $display("\nFAIL: FSM did not complete correctly");

        #100;
        $finish;
    end

    // =========================================================
    // Waveform dump
    // =========================================================
    initial begin
        $dumpfile("layer_fsm_full_tb.vcd");
        $dumpvars(0, layer_fsm_tb);
    end

endmodule
