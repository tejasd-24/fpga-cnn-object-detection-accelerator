#!/usr/bin/env python3
"""
ARM-only CNN inference benchmark.

Runs the exact same 3-layer CNN (Conv3x3 → ReLU → MaxPool2x2) entirely on
the ARM Cortex-A9 using NumPy, with the same int8 weights and uint8 math
as the FPGA accelerator. This establishes the ARM baseline for speedup comparison.

Usage (on PYNQ):
    python3 arm_benchmark.py --weights fpga_weights/weights.bin \
        --image fpga_weights/test_image_0_class0.bin

    python3 arm_benchmark.py --weights fpga_weights/weights.bin \
        --image fpga_weights/test_image_200_class2.bin --runs 5
"""

import numpy as np
import time
import argparse
import os
import json


# ─── Constants (match FPGA exactly) ─────────────────────────────
NUM_WEIGHT_BYTES = 23184
IMG_SIZE = 128
SHIFT_L0 = 2
SHIFT_L1 = 4
SHIFT_L2 = 6

# Layer configs: (in_ch, out_ch, in_size)
LAYERS = [
    (1,  16, 128),   # Layer 0: 128×128×1  → 64×64×16
    (16, 32,  64),   # Layer 1:  64×64×16 → 32×32×32
    (32, 64,  32),   # Layer 2:  32×32×32 → 16×16×64
]
SHIFTS = [SHIFT_L0, SHIFT_L1, SHIFT_L2]


def parse_weights(weights_bin):
    """Parse the flat weights.bin into per-layer 3×3 kernel arrays."""
    kernels = []
    offset = 0
    for in_ch, out_ch, _ in LAYERS:
        num = out_ch * in_ch * 9
        raw = weights_bin[offset:offset + num].astype(np.int8)  # signed!
        # Reshape: [out_ch, in_ch, 3, 3]
        # Weight layout in .bin: for each group of 16 output channels,
        # for each input channel, 9 weights per core (16 cores)
        # Actually the layout is: for each (input_ch_batch),
        #   for each output group (16 cores), 9 weights
        # Let me match the FPGA weight loading order:
        #   The FSM loads 144 bytes per pass (16 cores × 9 weights)
        #   For in_ch input channels, that's in_ch passes.
        # So layout: [in_ch][16_cores][9_weights], repeated for out_ch/16 batches

        # Number of output batches (each batch = 16 output channels)
        out_batches = out_ch // 16
        k = np.zeros((out_ch, in_ch, 3, 3), dtype=np.int8)

        idx = 0
        for ob in range(out_batches):
            for ic in range(in_ch):
                for core in range(16):
                    oc = ob * 16 + core
                    for wi in range(9):
                        k[oc, ic, wi // 3, wi % 3] = raw[idx]
                        idx += 1

        kernels.append(k)
        offset += num

    return kernels


def arm_conv_layer(x, kernels, shift):
    """
    One CNN layer on ARM: Conv3×3 → ReLU → MaxPool2×2.
    Matches FPGA integer arithmetic exactly:
      - int8 weights × uint8 activations → int32 accumulator
      - Arithmetic right shift
      - ReLU: clamp to [0, 255]
      - 2×2 max pooling
    
    Args:
        x: input, shape (in_ch, H, W), dtype uint8
        kernels: shape (out_ch, in_ch, 3, 3), dtype int8
        shift: ReLU right-shift amount
    
    Returns:
        output, shape (out_ch, H//2, W//2), dtype uint8
    """
    in_ch, H, W = x.shape
    out_ch = kernels.shape[0]

    # Pad input (zero padding for 3×3 same convolution)
    x_pad = np.zeros((in_ch, H + 2, W + 2), dtype=np.int32)
    x_pad[:, 1:H+1, 1:W+1] = x.astype(np.int32)

    # Convolution: accumulate across input channels
    conv_out = np.zeros((out_ch, H, W), dtype=np.int32)

    # Vectorized convolution using NumPy
    k = kernels.astype(np.int32)  # (out_ch, in_ch, 3, 3)

    for dy in range(3):
        for dx in range(3):
            patch = x_pad[:, dy:dy+H, dx:dx+W]  # (in_ch, H, W)
            # Einstein summation: out[oc, h, w] += k[oc, ic, dy, dx] * patch[ic, h, w]
            conv_out += np.einsum('oi,ihw->ohw', k[:, :, dy, dx], patch)

    # Arithmetic right shift (signed)
    conv_shifted = conv_out >> shift

    # ReLU: clamp to [0, 255]
    relu_out = np.clip(conv_shifted, 0, 255).astype(np.uint8)

    # MaxPool 2×2
    pool_out = relu_out.reshape(out_ch, H // 2, 2, W // 2, 2).max(axis=(2, 4))

    return pool_out


def arm_inference(image, kernels, shifts):
    """Run full 3-layer CNN on ARM. Returns features and per-layer times."""
    x = image.reshape(1, IMG_SIZE, IMG_SIZE).astype(np.uint8)

    layer_times = []
    for i, (kernel, shift) in enumerate(zip(kernels, shifts)):
        t0 = time.time()
        x = arm_conv_layer(x, kernel, shift)
        dt = time.time() - t0
        layer_times.append(dt)

    return x, layer_times


def classify(features, fc_weight, fc_bias, class_names):
    """Spatial bin pooling → linear classifier (same as FPGA pipeline)."""
    feat_maps = features.astype(np.float32)  # (64, 16, 16)
    ch, h, w = feat_maps.shape

    # Normalize to [0, 1]
    feat_maps = feat_maps / 255.0

    # 4×4 spatial bin pooling
    grid = 4
    pooled = np.zeros(ch * grid * grid, dtype=np.float32)
    for c in range(ch):
        for r in range(grid):
            for col in range(grid):
                bin_val = feat_maps[c,
                    r * (h // grid):(r + 1) * (h // grid),
                    col * (w // grid):(col + 1) * (w // grid)
                ].mean()
                pooled[c * grid * grid + r * grid + col] = bin_val

    # Linear layer
    logits = pooled @ fc_weight.T + fc_bias

    # Softmax
    exp_l = np.exp(logits - logits.max())
    probs = exp_l / exp_l.sum()

    idx = np.argmax(probs)
    return idx, class_names[idx], probs[idx], probs


def main():
    parser = argparse.ArgumentParser(description='ARM-only CNN benchmark')
    parser.add_argument('--weights', required=True, help='Path to weights.bin')
    parser.add_argument('--image', required=True, help='Path to test image (.bin)')
    parser.add_argument('--runs', type=int, default=3, help='Number of benchmark runs')
    parser.add_argument('--fc-weight', default=None, help='Path to fc_weight.npy')
    parser.add_argument('--fc-bias', default=None, help='Path to fc_bias.npy')
    parser.add_argument('--classes', default=None, help='Path to classes.json')
    args = parser.parse_args()

    # Find weight dir
    wdir = os.path.dirname(args.weights)
    fc_w_path = args.fc_weight or os.path.join(wdir, 'fc_weight.npy')
    fc_b_path = args.fc_bias or os.path.join(wdir, 'fc_bias.npy')
    cls_path = args.classes or os.path.join(wdir, 'classes.json')

    print("=" * 60)
    print("  ARM-ONLY CNN BENCHMARK (NumPy on Cortex-A9)")
    print("=" * 60)

    # Load weights
    weights_bin = np.fromfile(args.weights, dtype=np.uint8)
    assert len(weights_bin) == NUM_WEIGHT_BYTES
    print(f"Parsing {NUM_WEIGHT_BYTES} weight bytes...")
    kernels = parse_weights(weights_bin)
    for i, k in enumerate(kernels):
        print(f"  Layer {i}: kernel shape {k.shape}")

    # Load classifier
    fc_weight = np.load(fc_w_path)
    fc_bias = np.load(fc_b_path)
    class_names = ['airplane', 'cat', 'zebra', 'bus', 'bicycle', 'donut']
    if os.path.exists(cls_path):
        with open(cls_path) as f:
            class_names = json.load(f)
    print(f"Classifier: {fc_weight.shape[0]} classes — {class_names}")

    # Load test image
    image = np.fromfile(args.image, dtype=np.uint8)
    assert len(image) == IMG_SIZE * IMG_SIZE, f"Expected {IMG_SIZE*IMG_SIZE}, got {len(image)}"

    # Extract true class from filename
    basename = os.path.basename(args.image)
    true_class = None
    if '_class' in basename:
        true_class = int(basename.split('_class')[1].split('.')[0])

    print(f"\nImage: {basename}")
    if true_class is not None:
        print(f"True class: {class_names[true_class]} (class {true_class})")

    # ─── Benchmark ──────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Running {args.runs} inference(s)...")
    print(f"{'─' * 60}")

    all_times = []
    for run in range(args.runs):
        t_total = time.time()
        features, layer_times = arm_inference(image, kernels, SHIFTS)
        dt_total = time.time() - t_total

        all_times.append(dt_total)

        print(f"\n  Run {run + 1}/{args.runs}:")
        for i, lt in enumerate(layer_times):
            in_ch, out_ch, in_sz = LAYERS[i]
            out_sz = in_sz // 2
            macs = in_sz * in_sz * out_ch * in_ch * 9
            print(f"    Layer {i} ({in_ch}→{out_ch}ch, {in_sz}×{in_sz}→{out_sz}×{out_sz}): "
                  f"{lt*1000:.1f} ms  ({macs/1e6:.1f}M MACs)")
        print(f"    ────────────────────────────────────")
        print(f"    TOTAL CONV TIME: {dt_total*1000:.1f} ms")

    # Classify using last run's features
    features_flat = features.reshape(64, 256)
    features_spatial = features.reshape(64, 16, 16)
    cls_idx, cls_name, conf, probs = classify(features_spatial, fc_weight, fc_bias, class_names)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Prediction:  {cls_name} (class {cls_idx})")
    print(f"  Confidence:  {conf*100:.1f}%")
    if true_class is not None:
        print(f"  Correct:     {'✓' if cls_idx == true_class else '✗'}")

    # Timing summary
    avg_time = np.mean(all_times)
    min_time = np.min(all_times)
    print(f"\n  ARM conv time (avg): {avg_time*1000:.1f} ms")
    print(f"  ARM conv time (min): {min_time*1000:.1f} ms")
    print(f"  ARM FPS:             {1.0/avg_time:.2f}")
    print(f"\n  Compare to FPGA:     6.8 ms")
    print(f"  FPGA speedup:        {avg_time*1000/6.8:.1f}×")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
