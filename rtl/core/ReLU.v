
module ReLU(
    input signed [23:0] conv_in,
    output [7:0] relu_out

    );

// Signed comparison: negative → 0, >255 → 255, else passthrough
assign relu_out = (conv_in[23])          ? 8'd0   :   // negative
                  (|conv_in[23:8])       ? 8'd255 :   // > 255
                                           conv_in[7:0];

endmodule
