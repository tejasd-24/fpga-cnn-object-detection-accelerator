/*
 * arm_cnn.c — Optimized 3-layer CNN inference for ARM Cortex-A9
 * Matches FPGA accelerator math bit-for-bit.
 *
 * Architecture:
 *   Layer 0: 1→16ch,  128×128, conv3×3, ReLU>>2, pool2×2 → 16×64×64
 *   Layer 1: 16→32ch, 64×64,   conv3×3, ReLU>>4, pool2×2 → 32×32×32
 *   Layer 2: 32→64ch, 32×32,   conv3×3, ReLU>>6, pool2×2 → 64×16×16
 *
 * Compile: gcc -shared -fPIC -O3 -mcpu=cortex-a9 -mfpu=neon -o arm_cnn.so arm_cnn.c
 * Fallback: gcc -shared -fPIC -O3 -o arm_cnn.so arm_cnn.c
 */

#include <stdint.h>
#include <string.h>

/* Layer 0 is the largest: 128×128 input, 130×130 padded, 16 output channels.
 * Layer 1: 64×64 input, 66×66 padded, 32 output channels.
 * Layer 2: 32×32 input, 34×34 padded, 64 output channels.
 * 
 * Use flat 1D arrays with explicit stride calculations to avoid
 * 3D array stride mismatches.
 */

/* Max padded size: 130×130 = 16900, ×64 channels = 1,081,600 */
#define MAX_PAD_SIZE (130 * 130)   /* per channel */
#define MAX_CONV_SIZE (128 * 128)  /* per channel, output before pool */
#define MAX_POOL_SIZE (64 * 64)    /* per channel, output after pool */

static uint8_t pad_buf[64 * MAX_PAD_SIZE];    /* zero-padded input */
static int32_t conv_buf[64 * MAX_CONV_SIZE];  /* convolution accumulator */
static uint8_t inter_buf[64 * MAX_POOL_SIZE]; /* intermediate between layers */

/*
 * Weight layout (matching FPGA/training export):
 *   For each layer: grouped by output-channel blocks of 16
 *     for ob in 0..oc/16:
 *       for ic in 0..ic_count:
 *         for c in 0..16:
 *           for w in 0..9:
 *             kernel[ob*16+c][ic][w/3][w%3] = weights[idx++]
 */
static void parse_kernels(const uint8_t *raw, int8_t *kern_out,
                          int oc, int ic)
{
    int idx = 0;
    for (int ob = 0; ob < oc / 16; ob++) {
        for (int i = 0; i < ic; i++) {
            for (int c = 0; c < 16; c++) {
                int o = ob * 16 + c;
                for (int w = 0; w < 9; w++) {
                    /* kern_out[o][i][dy][dx] stored flat as [o*ic*9 + i*9 + dy*3+dx] */
                    kern_out[(o * ic + i) * 9 + (w / 3) * 3 + w % 3] =
                        (int8_t)raw[idx++];
                }
            }
        }
    }
}

/*
 * Run one CNN layer: conv3×3 + ReLU>>shift + maxpool2×2
 *
 *   input:  flat uint8, layout [ic][H][W] with stride H*W per channel
 *   output: flat uint8, layout [oc][H/2][W/2]
 *   kern:   flat int8,  layout [oc][ic][3][3] = [oc*ic*9]
 */
static void run_layer(const uint8_t *input, int ic, int H, int W,
                      const int8_t *kern, int oc, int shift,
                      uint8_t *output)
{
    int pW = W + 2;

    /* Zero-pad input into pad_buf: each channel is pW wide, H+2 tall */
    int pad_stride = pW;  /* row stride in padded buffer */
    int pad_ch_size = (H + 2) * pW;  /* per-channel size in padded buffer */

    memset(pad_buf, 0, (size_t)ic * pad_ch_size);
    for (int ch = 0; ch < ic; ch++) {
        for (int r = 0; r < H; r++) {
            /* Copy row r of channel ch to padded row r+1, col offset 1 */
            memcpy(&pad_buf[ch * pad_ch_size + (r + 1) * pad_stride + 1],
                   &input[ch * H * W + r * W],
                   (size_t)W);
        }
    }

    /* Clear conv accumulator: oc channels × H × W */
    int conv_ch_size = H * W;
    memset(conv_buf, 0, sizeof(int32_t) * (size_t)oc * conv_ch_size);

    /* Convolution: accumulate all input channels */
    for (int o = 0; o < oc; o++) {
        int32_t *out_ch = &conv_buf[o * conv_ch_size];
        for (int i = 0; i < ic; i++) {
            const int8_t *k = &kern[(o * ic + i) * 9];
            const uint8_t *pad_ch = &pad_buf[i * pad_ch_size];
            for (int dy = 0; dy < 3; dy++) {
                for (int dx = 0; dx < 3; dx++) {
                    int8_t kv = k[dy * 3 + dx];
                    if (kv == 0) continue;
                    for (int r = 0; r < H; r++) {
                        const uint8_t *src_row = &pad_ch[(r + dy) * pad_stride + dx];
                        int32_t *dst_row = &out_ch[r * W];
                        for (int c = 0; c < W; c++) {
                            dst_row[c] += (int32_t)kv * (int32_t)src_row[c];
                        }
                    }
                }
            }
        }
    }

    /* ReLU >> shift + MaxPool 2×2 */
    int oH = H / 2, oW = W / 2;
    for (int o = 0; o < oc; o++) {
        const int32_t *conv_ch = &conv_buf[o * conv_ch_size];
        uint8_t *out_ch = &output[o * oH * oW];
        for (int pr = 0; pr < oH; pr++) {
            for (int pc = 0; pc < oW; pc++) {
                int r = pr * 2, c = pc * 2;
                int32_t v00 = conv_ch[r * W + c];
                int32_t v01 = conv_ch[r * W + c + 1];
                int32_t v10 = conv_ch[(r + 1) * W + c];
                int32_t v11 = conv_ch[(r + 1) * W + c + 1];

                v00 = v00 > 0 ? (v00 >> shift) : 0;
                v01 = v01 > 0 ? (v01 >> shift) : 0;
                v10 = v10 > 0 ? (v10 >> shift) : 0;
                v11 = v11 > 0 ? (v11 >> shift) : 0;

                if (v00 > 255) v00 = 255;
                if (v01 > 255) v01 = 255;
                if (v10 > 255) v10 = 255;
                if (v11 > 255) v11 = 255;

                int32_t mx = v00;
                if (v01 > mx) mx = v01;
                if (v10 > mx) mx = v10;
                if (v11 > mx) mx = v11;

                out_ch[pr * oW + pc] = (uint8_t)mx;
            }
        }
    }
}

/*
 * Full 3-layer CNN inference.
 *
 * Args:
 *   input_img:   128×128 uint8 grayscale (row-major, 16384 bytes)
 *   weights_bin: raw weight binary (23184 bytes)
 *   shifts:      3 ints [shift_l0, shift_l1, shift_l2]
 *   output:      64×16×16 = 16384 uint8 feature map
 *
 * Returns: 0 on success
 */
int cnn_infer(const uint8_t *input_img,
              const uint8_t *weights_bin,
              const int *shifts,
              uint8_t *output)
{
    static const int cfg[][4] = {
        {1, 16, 128, 128},
        {16, 32, 64, 64},
        {32, 64, 32, 32}
    };
    static const int wt_sizes[] = {
        16 * 1 * 9,    /* 144 */
        32 * 16 * 9,   /* 4608 */
        64 * 32 * 9    /* 18432 */
    };

    static int8_t kern[64 * 32 * 9]; /* max: layer 2 = 64*32*9 = 18432 */

    const uint8_t *cur_input = input_img;
    int wt_off = 0;

    for (int layer = 0; layer < 3; layer++) {
        int ic = cfg[layer][0];
        int oc = cfg[layer][1];
        int H  = cfg[layer][2];
        int W  = cfg[layer][3];

        parse_kernels(weights_bin + wt_off, kern, oc, ic);
        wt_off += wt_sizes[layer];

        /* Last layer outputs to caller's buffer; others to inter_buf */
        uint8_t *out = (layer == 2) ? output : inter_buf;

        run_layer(cur_input, ic, H, W, kern, oc, shifts[layer], out);

        cur_input = out;
    }

    return 0;
}
