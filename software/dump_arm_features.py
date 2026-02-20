#!/usr/bin/env python3
"""
Dump ARM Features — runs on the PYNQ board
============================================
Runs the same 3-layer CNN on ARM (using optimized C or numpy fallback)
for all test images. Saves features as .npz in the same format as
dump_fpga_features.py for classifier retraining.

Usage (on PYNQ):
  python3 dump_arm_features.py
  python3 retrain_classifier.py --features arm_features.npz --output-dir fpga_weights --prefix arm_

Then use with realtime_detect.py --mode arm
"""

import numpy as np
import os
import glob
import time
import argparse
import subprocess
import ctypes

# ── Constants (match FPGA exactly) ──
IMG = 128
SH0, SH1, SH2 = 2, 4, 6
NUM_WEIGHT_BYTES = 23184
N_CH, FM = 64, 256  # 64 channels × 16×16

DIR = os.path.dirname(os.path.abspath(__file__))
ARM_C_SRC = os.path.join(DIR, "arm_cnn.c")
ARM_C_LIB = os.path.join(DIR, "arm_cnn.so")


def compile_c_lib():
    """Compile the C CNN library if needed."""
    if not os.path.exists(ARM_C_SRC):
        return None
    if not os.path.exists(ARM_C_LIB) or os.path.getmtime(ARM_C_SRC) > os.path.getmtime(ARM_C_LIB):
        print("Compiling ARM CNN C library...", end=" ", flush=True)
        for flags in [
            ['-O3', '-mcpu=cortex-a9', '-mfpu=neon', '-mfloat-abi=hard'],
            ['-O3'],
        ]:
            r = subprocess.run(
                ['gcc', '-shared', '-fPIC'] + flags + ['-o', ARM_C_LIB, ARM_C_SRC],
                capture_output=True, text=True)
            if r.returncode == 0:
                print(f"done.")
                break
        else:
            print(f"FAILED")
            return None
    lib = ctypes.CDLL(ARM_C_LIB)
    lib.cnn_infer.argtypes = [ctypes.c_void_p]*4
    lib.cnn_infer.restype = ctypes.c_int
    return lib


def parse_kernels(wt):
    """Parse weights.bin into numpy kernels for fallback."""
    kernels, off = [], 0
    for ic,oc,_ in [(1,16,128),(16,32,64),(32,64,32)]:
        n = oc*ic*9; raw = wt[off:off+n].astype(np.int8)
        k = np.zeros((oc,ic,3,3), dtype=np.int8); idx=0
        for ob in range(oc//16):
            for i in range(ic):
                for c in range(16):
                    for w in range(9):
                        k[ob*16+c,i,w//3,w%3]=raw[idx]; idx+=1
        kernels.append(k); off+=n
    return kernels


def numpy_infer(image, kernels):
    """Numpy CNN inference (fallback if C not available)."""
    x = image.reshape(1,IMG,IMG).astype(np.uint8)
    for kern,sh in zip(kernels, [SH0,SH1,SH2]):
        ic,H,W = x.shape; oc = kern.shape[0]
        xp = np.zeros((ic,H+2,W+2),dtype=np.int32)
        xp[:,1:H+1,1:W+1] = x.astype(np.int32)
        out = np.zeros((oc,H,W),dtype=np.int32)
        ki = kern.astype(np.int32)
        for dy in range(3):
            for dx in range(3):
                out += np.einsum('oi,ihw->ohw',ki[:,:,dy,dx],xp[:,dy:dy+H,dx:dx+W])
        x = np.clip(out>>sh,0,255).astype(np.uint8)
        x = x.reshape(oc,H//2,2,W//2,2).max(axis=(2,4))
    return x.reshape(N_CH, FM)


def main():
    parser = argparse.ArgumentParser(description="Dump ARM CNN features")
    parser.add_argument("--weights", default="fpga_weights/weights.bin")
    parser.add_argument("--image-dir", default="fpga_weights")
    parser.add_argument("--output", default="arm_features.npz")
    args = parser.parse_args()

    print("=" * 60)
    print("  ARM FEATURE DUMP (C-optimized CNN)")
    print("=" * 60)

    # Load weights
    wt = np.fromfile(args.weights, dtype=np.uint8)
    assert len(wt) == NUM_WEIGHT_BYTES
    shifts_arr = np.array([SH0, SH1, SH2], dtype=np.int32)

    # Try C library first
    clib = compile_c_lib()
    kernels = parse_kernels(wt) if clib is None else None

    if clib:
        print("Using C-optimized inference")
    else:
        print("Using numpy fallback (slower)")

    # Find test images
    pattern = os.path.join(args.image_dir, "test_image_*.bin")
    images = sorted(glob.glob(pattern))
    if not images:
        print(f"No test images found: {pattern}")
        return
    print(f"Found {len(images)} test images\n")

    all_features = []
    all_labels = []
    all_names = []
    output_buf = np.zeros(N_CH * FM, dtype=np.uint8)

    t_start = time.time()
    for idx, img_path in enumerate(images):
        basename = os.path.basename(img_path)

        # Extract label from filename
        label = -1
        if "_class" in basename:
            label = int(basename.split("_class")[1].split(".")[0])

        image = np.fromfile(img_path, dtype=np.uint8)
        if len(image) != IMG * IMG:
            continue

        if clib:
            clib.cnn_infer(
                image.ctypes.data_as(ctypes.c_void_p),
                wt.ctypes.data_as(ctypes.c_void_p),
                shifts_arr.ctypes.data_as(ctypes.c_void_p),
                output_buf.ctypes.data_as(ctypes.c_void_p))
            features = output_buf.reshape(N_CH, FM).copy()
        else:
            features = numpy_infer(image, kernels)

        all_features.append(features)
        all_labels.append(label)
        all_names.append(basename)

        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            print(f"  [{idx+1:3d}/{len(images)}] {rate:.1f} img/s | {basename} label={label}")

    # Save
    features_array = np.stack(all_features)  # (N, 64, 256)
    labels_array = np.array(all_labels)

    np.savez(args.output,
             features=features_array,
             labels=labels_array,
             names=all_names,
             shifts=np.array([SH0, SH1, SH2]))

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Saved {len(all_features)} feature maps to: {args.output}")
    print(f"  Shape: {features_array.shape}")
    print(f"  Labels: {np.unique(labels_array)}")
    print(f"  Total time: {elapsed:.1f}s ({len(all_features)/elapsed:.1f} img/s)")
    print(f"{'=' * 60}")
    print(f"\nNext: retrain classifier on these features:")
    print(f"  python3 retrain_classifier.py --features {args.output} --prefix arm_")


if __name__ == '__main__':
    main()
