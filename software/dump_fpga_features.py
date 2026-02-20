"""
Dump FPGA Features — runs on the PYNQ board
=============================================
Loads each test image, runs FPGA inference, reads back layer-2
features (64 channels × 16×16), and saves everything to a .npz file.

The .npz is then copied to your laptop for classifier retraining.

Usage (on PYNQ):
  python3 dump_fpga_features.py
  # → produces fpga_features.npz

Then copy to laptop:
  scp xilinx@192.168.2.99:/home/xilinx/jupyter_notebooks/CNN/fpga_features.npz .
"""

import numpy as np
import os
import glob
import time
import argparse
import sys

# ── Reuse accelerator class from pynq_inference.py ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pynq_inference import CNNAccelerator, SHIFT_L0, SHIFT_L1, SHIFT_L2

BITSTREAM_PATH = "/home/xilinx/jupyter_notebooks/CNN/lyr3_cnn.bit"

# Layer 2 output: 64 channels, 16×16 = 256 values each
L2_CHANNELS = 64
L2_SIZE = 256       # 16×16
L2_CH_OFFSET = 48   # feature BRAM channels 48-111


def dump_features(args):
    # ── Initialize FPGA ──
    print("=" * 60)
    print("  FPGA FEATURE DUMP")
    print("=" * 60)

    acc = CNNAccelerator(bitstream_path=args.bitstream)
    acc.set_shifts(args.s0, args.s1, args.s2)

    # Load weights once
    acc.load_weights(args.weights)

    # ── Find test images ──
    pattern = os.path.join(args.image_dir, "test_image_*.bin")
    images = sorted(glob.glob(pattern))
    if not images:
        print(f"No test images found: {pattern}")
        return

    print(f"Found {len(images)} test images")
    print("-" * 60)

    all_features = []
    all_labels = []
    all_names = []

    skipped = 0
    for idx, img_path in enumerate(images):
        basename = os.path.basename(img_path)

        # Extract label from filename: test_image_N_classL.bin
        label = -1
        if "_class" in basename:
            label = int(basename.split("_class")[1].split(".")[0])

        try:
            # Load image into FPGA
            image = np.fromfile(img_path, dtype=np.uint8)
            acc.load_image(image)

            # Run inference
            acc.start_inference()
            elapsed = acc.wait_done(timeout=15.0)

            # Read all 64 layer-2 feature maps
            features = np.zeros((L2_CHANNELS, L2_SIZE), dtype=np.uint8)
            for ch in range(L2_CHANNELS):
                bram_ch = L2_CH_OFFSET + ch
                acc.write_reg(0x20, bram_ch)  # REG_OUTPUT_CH
                for addr in range(L2_SIZE):
                    acc.write_reg(0x24, addr)  # REG_OUTPUT_ADDR
                    _ = acc.read_reg(0x28)     # trigger read
                    features[ch, addr] = acc.read_reg(0x28) & 0xFF

            all_features.append(features)
            all_labels.append(label)
            all_names.append(basename)

            if (idx + 1) % 10 == 0 or idx == 0:
                nonzero = (features > 0).sum()
                fmax = features.max()
                fmean = features.mean()
                print(f"  [{idx+1:3d}/{len(images)}] {basename}: "
                      f"label={label}, max={fmax}, mean={fmean:.1f}, "
                      f"nonzero={nonzero}/{L2_CHANNELS*L2_SIZE}, "
                      f"fpga={elapsed*1000:.1f}ms")

        except (OSError, IOError) as e:
            skipped += 1
            print(f"  [{idx+1:3d}/{len(images)}] SKIP {basename}: {e}")
            continue

    if skipped:
        print(f"\n  Skipped {skipped} files due to I/O errors")

    # ── Save ──
    features_array = np.stack(all_features)    # (N, 64, 256)
    labels_array = np.array(all_labels)        # (N,)

    out_path = os.path.join(os.path.dirname(args.image_dir), args.output)
    np.savez(out_path,
             features=features_array,
             labels=labels_array,
             names=all_names,
             shifts=np.array([args.s0, args.s1, args.s2]))

    print(f"\n{'=' * 60}")
    print(f"  Saved {len(all_features)} feature maps to: {out_path}")
    print(f"  Shape: {features_array.shape}")
    print(f"  Labels: {np.unique(labels_array)}")
    print(f"  Feature range: [{features_array.min()}, {features_array.max()}]")
    print(f"  Mean: {features_array.astype(float).mean():.2f}")
    print(f"  Nonzero: {(features_array > 0).mean()*100:.1f}%")
    print(f"{'=' * 60}")
    print(f"\nNow copy to laptop and run retrain_classifier.py:")
    print(f"  scp xilinx@<pynq_ip>:{out_path} /home/tejas/Desktop/cnn_accn/training/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump FPGA features")
    parser.add_argument("--bitstream", default=BITSTREAM_PATH)
    parser.add_argument("--weights", default="fpga_weights/weights.bin")
    parser.add_argument("--image-dir", default="fpga_weights")
    parser.add_argument("--output", default="fpga_features.npz")
    parser.add_argument("--s0", type=int, default=SHIFT_L0)
    parser.add_argument("--s1", type=int, default=SHIFT_L1)
    parser.add_argument("--s2", type=int, default=SHIFT_L2)
    args = parser.parse_args()
    dump_features(args)
