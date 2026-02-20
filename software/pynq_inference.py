"""
PYNQ Z2 CNN Accelerator — Inference Script
===========================================
Runs on the PYNQ board. Loads trained weights + image into the FPGA via
AXI DMA, runs inference, reads back layer-2 feature maps via AXI-Lite,
and classifies using a linear layer on the ARM CPU.

Files needed on the board (copy from training/fpga_weights/):
  - weights.bin           (23,184 bytes — FPGA weight BRAM)
  - fc_weight.npy         (num_classes × 64 float32)
  - fc_bias.npy           (num_classes float32)
  - classes.json          (class name list)
  - test_image_*.bin      (128×128 grayscale test images)

Hardware register map (from lyr3_cnn_axi_slave_lite_v1_0_S00_AXI.v):
  ┌────────────────────────────────────────────────────────────────┐
  │  Reg   Offset   Name              Description                 │
  ├────────────────────────────────────────────────────────────────┤
  │  R0    0x00     control           [0]=start (auto-clear)      │
  │                                   [1]=stream_phase (0=wt,1=img│
  │                                   [2]=addr_reset (auto-clear) │
  │  R1    0x04     status (RO)       [0]=busy, [1]=done,         │
  │                                   [3:2]=current_layer         │
  │  R2    0x08     (unused)                                      │
  │  R3    0x0C     (unused)                                      │
  │  R4    0x10     (unused)                                      │
  │  R5    0x14     (unused)                                      │
  │  R6    0x18     (unused)                                      │
  │  R7    0x1C     (unused)                                      │
  │  R8    0x20     output_ch         [6:0] = channel to read     │
  │  R9    0x24     output_addr       [11:0] = address to read    │
  │  R10   0x28     output_data (RO)  [7:0] = data readback       │
  │  R10   0x28     relu_shifts (WR)  [4:0]=L0, [9:5]=L1,         │
  │                                   [14:10]=L2                  │
  │                                                                │
  │  Data loading uses AXI-Stream via DMA (not register writes)   │
  │  stream_phase=0: DMA sends weights (23,184 bytes)             │
  │  stream_phase=1: DMA sends image   (16,384 bytes)             │
  └────────────────────────────────────────────────────────────────┘

Usage:
  python3 pynq_inference.py                           # classify all test images
  python3 pynq_inference.py --image test_image_0.bin  # single image
"""

import numpy as np
from PIL import Image, ImageDraw
import os
import sys
import json
import time
import glob
import argparse


# ============================================================
#  Configuration — UPDATE THESE FOR YOUR BOARD
# ============================================================
BITSTREAM_PATH = "/home/xilinx/jupyter_notebooks/CNN/lyr3_cnn.bit"

# AXI-Lite register offsets (from AXI wrapper RTL)
# ADDR_LSB = 2, OPT_MEM_ADDR_BITS = 3
# Register N is at byte offset N * 4
REG_CONTROL      = 0x00   # R0: [0]=start, [1]=stream_phase, [2]=addr_reset
REG_STATUS       = 0x04   # R1: [0]=busy, [1]=done, [3:2]=current_layer
REG_OUTPUT_CH    = 0x20   # R8: [6:0] = output channel select
REG_OUTPUT_ADDR  = 0x24   # R9: [11:0] = output address select
REG_OUTPUT_DATA  = 0x28   # R10 read: [7:0] = output data
REG_RELU_SHIFTS  = 0x28   # R10 write: [4:0]=L0, [9:5]=L1, [14:10]=L2

# File paths (relative to script location)
WEIGHTS_FILE   = "fpga_weights/weights.bin"
FC_WEIGHT_FILE = "fpga_weights/fc_weight.npy"
FC_BIAS_FILE   = "fpga_weights/fc_bias.npy"
CLASSES_FILE   = "fpga_weights/classes.json"

# Layer 2 output geometry
L2_NUM_CHANNELS = 64
L2_SIZE         = 256     # 16×16 after 2×2 maxpool of 32×32
L2_CH_OFFSET    = 48      # feature BRAM channels 48-111

# Default ReLU shift values (must match training)
SHIFT_L0 = 2
SHIFT_L1 = 4
SHIFT_L2 = 6

# Number of bytes
NUM_WEIGHT_BYTES = 23184
NUM_IMAGE_BYTES  = 16384  # 128 × 128


# ============================================================
#  PYNQ Hardware Interface
# ============================================================
class CNNAccelerator:
    """Interface to the CNN accelerator on PYNQ via AXI-Lite + DMA."""

    def __init__(self, bitstream_path=BITSTREAM_PATH):
        """Program FPGA and get handles to AXI-Lite IP and DMA."""
        try:
            from pynq import Overlay, allocate
            self.allocate = allocate

            print(f"Loading bitstream: {bitstream_path}")
            self.overlay = Overlay(bitstream_path)
            print("Bitstream loaded successfully.")

            # ── Find the AXI-Lite control IP ──
            self.ip = None
            for name in dir(self.overlay):
                if name.startswith('_'):
                    continue
                obj = getattr(self.overlay, name)
                # Look for our custom IP (has register width 6-bit addr = 64 bytes)
                if hasattr(obj, 'register_map') or hasattr(obj, 'mmio'):
                    if 'lyr3_cnn' in name or 'cnn' in name.lower():
                        self.ip = obj
                        print(f"Found accelerator IP: {name}")
                        break

            if self.ip is None:
                # Fallback: list all IPs
                print("\nAvailable IPs:")
                for name in dir(self.overlay):
                    if not name.startswith('_'):
                        obj = getattr(self.overlay, name)
                        print(f"  {name}: {type(obj).__name__}")
                raise RuntimeError(
                    "Could not find CNN accelerator IP. "
                    "Check the block design IP names."
                )

            # Get MMIO handle
            if hasattr(self.ip, 'mmio'):
                self.mmio = self.ip.mmio
            else:
                from pynq import MMIO
                self.mmio = MMIO(self.ip.base_addr, 0x100)

            print(f"Control MMIO base: 0x{self.mmio.base_addr:08X}")

            # ── Find the AXI DMA ──
            self.dma = None
            for name in dir(self.overlay):
                if 'dma' in name.lower():
                    obj = getattr(self.overlay, name)
                    if hasattr(obj, 'sendchannel'):
                        self.dma = obj
                        print(f"Found DMA: {name}")
                        break

            if self.dma is None:
                print("WARNING: No DMA found. Will use slow register-based loading.")

            self._pynq = True

        except ImportError:
            print("=" * 50)
            print("  PYNQ not available — SIMULATION MODE")
            print("  (Testing script logic without hardware)")
            print("=" * 50)
            self._pynq = False

    def write_reg(self, offset, value):
        if self._pynq:
            self.mmio.write(offset, int(value))

    def read_reg(self, offset):
        if self._pynq:
            return self.mmio.read(offset)
        return 0x2  # simulate done=1, busy=0

    def _reset_stream_addr(self):
        """Pulse bit[2] of control register to reset stream address counter."""
        self.write_reg(REG_CONTROL, 0x04)  # bit 2 = addr_reset (auto-clears)
        time.sleep(0.0001)

    def _send_via_dma(self, data_bytes):
        """Send a byte array to the accelerator via AXI DMA."""
        buf = self.allocate(shape=(len(data_bytes),), dtype=np.uint8)
        buf[:] = data_bytes
        self.dma.sendchannel.transfer(buf)
        self.dma.sendchannel.wait()
        del buf

    def load_weights(self, weights_path):
        """Load quantized weights into FPGA weight BRAM via DMA."""
        weights = np.fromfile(weights_path, dtype=np.uint8)
        assert len(weights) == NUM_WEIGHT_BYTES, (
            f"Expected {NUM_WEIGHT_BYTES} weights, got {len(weights)}"
        )

        print(f"Loading {len(weights)} weights...", end=" ", flush=True)
        t0 = time.time()

        if self._pynq and self.dma:
            # Set stream_phase = 0 (weights) and reset stream address
            self.write_reg(REG_CONTROL, 0x04)  # addr_reset
            time.sleep(0.0001)
            self.write_reg(REG_CONTROL, 0x00)  # stream_phase=0 (weights)

            self._send_via_dma(weights)
        elif self._pynq:
            print("(no DMA, skipping)", end=" ")

        dt = time.time() - t0
        print(f"done ({dt:.2f}s)")

    def load_image(self, image):
        """Load a 128×128 grayscale image into input BRAM via DMA."""
        if isinstance(image, str):
            image = np.fromfile(image, dtype=np.uint8)

        assert len(image) == NUM_IMAGE_BYTES, (
            f"Expected {NUM_IMAGE_BYTES} pixels, got {len(image)}"
        )

        if self._pynq and self.dma:
            # Set stream_phase = 1 (image) and reset stream address
            self.write_reg(REG_CONTROL, 0x06)  # bit2=addr_reset, bit1=image_phase
            time.sleep(0.0001)
            self.write_reg(REG_CONTROL, 0x02)  # bit1=image_phase, addr_reset auto-cleared

            self._send_via_dma(image)

    def set_shifts(self, s0=SHIFT_L0, s1=SHIFT_L1, s2=SHIFT_L2):
        """Set ReLU right-shift values (all packed into register 10)."""
        packed = (s0 & 0x1F) | ((s1 & 0x1F) << 5) | ((s2 & 0x1F) << 10)
        self.write_reg(REG_RELU_SHIFTS, packed)

    def start_inference(self):
        """Pulse start bit to begin inference."""
        self.write_reg(REG_CONTROL, 0x01)  # bit 0 = start (auto-clears)
        time.sleep(0.0001)

    def wait_done(self, timeout=10.0):
        """Poll done signal. Returns elapsed time in seconds."""
        t0 = time.time()
        while True:
            status = self.read_reg(REG_STATUS)
            done = (status >> 1) & 1
            busy = status & 1
            if done:
                return time.time() - t0
            if time.time() - t0 > timeout:
                layer = (status >> 2) & 3
                raise TimeoutError(
                    f"Timed out after {timeout}s "
                    f"(busy={busy}, done={done}, layer={layer})"
                )
            time.sleep(0.001)

    def read_feature_map(self, channel, num_values):
        """Read one feature map channel from FPGA BRAMs."""
        self.write_reg(REG_OUTPUT_CH, channel)
        values = np.zeros(num_values, dtype=np.uint8)

        for addr in range(num_values):
            self.write_reg(REG_OUTPUT_ADDR, addr)
            # Two reads: first triggers the registered BRAM read,
            # second gets the actual data
            _ = self.read_reg(REG_OUTPUT_DATA)
            values[addr] = self.read_reg(REG_OUTPUT_DATA) & 0xFF

        return values

    def read_layer2_output(self):
        """
        Read all 64 channels of layer 2 output (16×16=256 values each).
        Feature BRAM channels 48-111.

        Returns: numpy array shape (64, 256) uint8
        """
        features = np.zeros((L2_NUM_CHANNELS, L2_SIZE), dtype=np.uint8)

        print("Reading layer 2 features...", end=" ", flush=True)
        t0 = time.time()

        for ch in range(L2_NUM_CHANNELS):
            bram_ch = L2_CH_OFFSET + ch   # channels 48..111
            features[ch] = self.read_feature_map(bram_ch, L2_SIZE)

        dt = time.time() - t0
        print(f"done ({dt:.2f}s)")

        return features


# ============================================================
#  ARM-side Classifier
# ============================================================
class Classifier:
    """
    Final classification layer — runs on ARM CPU.
    AdaptiveAvgPool2d(1) → Linear(64, num_classes)
    """

    def __init__(self, fc_weight_path, fc_bias_path, classes_path=None):
        self.weight = np.load(fc_weight_path)   # (num_classes, 64)
        self.bias = np.load(fc_bias_path)       # (num_classes,)
        self.num_classes = self.weight.shape[0]

        self.class_names = None
        if classes_path and os.path.exists(classes_path):
            with open(classes_path) as f:
                self.class_names = json.load(f)

        print(f"Classifier: {self.num_classes} classes", end="")
        if self.class_names:
            print(f" — {self.class_names}")
        else:
            print()

    def classify(self, features):
        """
        Classify from layer 2 feature maps.

        Args:
            features: (64, 256) uint8 — raw FPGA output

        Returns:
            (class_index, class_name, confidence, probabilities)
        """
        # Spatial bin pooling (4×4 grid) — preserves WHERE activations are
        feat_maps = features.astype(np.float32).reshape(64, 16, 16)
        grid = 4
        bin_size = 16 // grid
        pooled = np.zeros(64 * grid * grid, dtype=np.float32)
        for ch in range(64):
            for r in range(grid):
                for c in range(grid):
                    patch = feat_maps[ch, r*bin_size:(r+1)*bin_size, c*bin_size:(c+1)*bin_size]
                    pooled[ch * grid * grid + r * grid + c] = patch.mean()
        pooled = pooled / 255.0  # normalize to [0,1]

        # Linear: scores = W @ pooled + b
        scores = self.weight @ pooled + self.bias

        # Softmax
        exp_s = np.exp(scores - scores.max())
        probs = exp_s / exp_s.sum()

        idx = int(np.argmax(scores))
        conf = float(probs[idx])
        name = self.class_names[idx] if self.class_names else str(idx)

        return idx, name, conf, probs

    def get_cam_bbox(self, features, class_idx, img_size=128):
        """
        Generate Class Activation Map and bounding box.
        Excludes saturated channels (mean > 250) that cause uniform activation.
        Uses percentile-based threshold for tighter bounding boxes.
        """
        feat_maps = features.astype(np.float32).reshape(64, 16, 16)

        # Compute per-channel means to find saturated channels
        ch_means = feat_maps.mean(axis=(1, 2))  # (64,)

        # Class weights for the predicted class
        w = self.weight[class_idx]  # (1024,) now

        # Build CAM from spatial bin weights
        grid = 4
        cam = np.zeros((16, 16), dtype=np.float32)
        for ch in range(64):
            # Skip saturated channels — they provide no spatial discrimination
            if ch_means[ch] > 250:
                continue
            for r in range(grid):
                for c in range(grid):
                    bin_idx = ch * grid * grid + r * grid + c
                    bin_weight = w[bin_idx]
                    bin_size = 16 // grid
                    cam[r*bin_size:(r+1)*bin_size, c*bin_size:(c+1)*bin_size] += (
                        bin_weight * feat_maps[ch, r*bin_size:(r+1)*bin_size, c*bin_size:(c+1)*bin_size]
                    )

        # ReLU
        cam = np.maximum(cam, 0)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upscale to img_size × img_size
        cam_img = Image.fromarray((cam * 255).astype(np.uint8))
        cam_img = cam_img.resize((img_size, img_size), Image.BILINEAR)
        cam_full = np.array(cam_img).astype(np.float32) / 255.0

        # Percentile-based threshold — keep top 30% of activation
        threshold = np.percentile(cam_full, 70)
        threshold = max(threshold, 0.2)  # floor to avoid too-small boxes
        mask = cam_full > threshold
        if mask.any():
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            pad = 3
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img_size - 1, x2 + pad)
            y2 = min(img_size - 1, y2 + pad)
        else:
            x1, y1, x2, y2 = 0, 0, img_size - 1, img_size - 1

        return cam_full, (x1, y1, x2, y2)


# ============================================================
#  Inference Pipeline
# ============================================================
def load_image_any(image_path):
    """Load image from .bin, .jpg, .png, or any PIL-supported format."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext == '.bin':
        image = np.fromfile(image_path, dtype=np.uint8)
        if len(image) != 128 * 128:
            raise ValueError(f"Expected 16384 bytes, got {len(image)}")
        return image
    else:
        # Load any image format, convert to 128×128 grayscale
        img = Image.open(image_path).convert('L').resize((128, 128))
        return np.array(img, dtype=np.uint8).flatten()


def run_inference(acc, classifier, image_path, verbose=True, save_output=True):
    """Full pipeline: load image → FPGA → classify → CAM bbox → save JPEG."""
    # 1. Load image (supports .bin, .jpg, .png)
    image = load_image_any(image_path)
    acc.load_image(image)

    # 2. Run accelerator
    acc.start_inference()
    elapsed = acc.wait_done(timeout=10.0)

    # 3. Read features
    features = acc.read_layer2_output()

    # 4. Classify
    idx, name, conf, probs = classifier.classify(features)

    # 5. CAM bounding box
    cam, bbox = classifier.get_cam_bbox(features, idx)
    x1, y1, x2, y2 = bbox

    if verbose:
        basename = os.path.basename(image_path)
        true_label = ""
        if "_class" in basename:
            true_label = f" (true: class {basename.split('_class')[1].split('.')[0]})"

        print(f"\n  Image:      {basename}{true_label}")
        print(f"  Prediction: {name} (class {idx})")
        print(f"  Confidence: {conf:.1%}")
        print(f"  BBox:       ({x1}, {y1}) → ({x2}, {y2})")
        print(f"  FPGA time:  {elapsed*1000:.1f} ms")

        # Top-3
        top3 = np.argsort(probs)[::-1][:3]
        print("  Top-3:")
        for rank, i in enumerate(top3):
            cn = classifier.class_names[i] if classifier.class_names else str(i)
            print(f"    {rank+1}. {cn}: {probs[i]:.1%}")

    # 6. Save output JPEG with bounding box
    if save_output:
        # Reconstruct grayscale image
        gray = image.reshape(128, 128)
        # Convert to RGB for colored bounding box
        rgb = np.stack([gray, gray, gray], axis=2)
        out_img = Image.fromarray(rgb, 'RGB')
        draw = ImageDraw.Draw(out_img)

        # Draw bounding box in green
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

        # Draw label background + text
        label = f"{name} {conf:.0%}"
        text_y = max(0, y1 - 14)
        draw.rectangle([x1, text_y, x1 + len(label) * 7, text_y + 13],
                       fill=(0, 255, 0))
        draw.text((x1 + 2, text_y + 1), label, fill=(0, 0, 0))

        # Save
        basename = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(os.path.dirname(image_path), f"{basename}_result.jpg")
        out_img.save(out_path, quality=95)
        if verbose:
            print(f"  Output:     {out_path}")

    return idx, name, conf


def main():
    parser = argparse.ArgumentParser(description="PYNQ CNN Inference")
    parser.add_argument("--bitstream", default=BITSTREAM_PATH)
    parser.add_argument("--weights", default=WEIGHTS_FILE)
    parser.add_argument("--image", default=None,
                        help="Single image (.bin, .jpg, .png)")
    parser.add_argument("--image-dir", default="fpga_weights")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save output JPEG")
    parser.add_argument("--shifts", default=f"{SHIFT_L0},{SHIFT_L1},{SHIFT_L2}",
                        help="ReLU shifts: l0,l1,l2")
    parser.add_argument("--dump-features", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve(p):
        return os.path.join(script_dir, p) if not os.path.isabs(p) else p

    weights_path   = resolve(args.weights)
    fc_weight_path = resolve(FC_WEIGHT_FILE)
    fc_bias_path   = resolve(FC_BIAS_FILE)
    classes_path   = resolve(CLASSES_FILE)

    shifts = [int(s) for s in args.shifts.split(",")]
    assert len(shifts) == 3

    # ── Init ─────────────────────────────────────────────────
    print("=" * 60)
    print("  CNN ACCELERATOR — PYNQ INFERENCE")
    print("=" * 60)

    acc = CNNAccelerator(bitstream_path=args.bitstream)
    acc.set_shifts(*shifts)
    acc.load_weights(weights_path)

    classifier = Classifier(fc_weight_path, fc_bias_path, classes_path)

    # ── Run ──────────────────────────────────────────────────
    if args.image:
        img_path = resolve(args.image)
        run_inference(acc, classifier, img_path,
                      save_output=not args.no_save)

        if args.dump_features:
            features = acc.read_layer2_output()
            out = img_path.replace(".bin", "_features.npy")
            np.save(out, features)
            print(f"  Features saved: {out}")
    else:
        pattern = os.path.join(resolve(args.image_dir), "test_image_*.bin")
        images = sorted(glob.glob(pattern))

        if not images:
            print(f"\nNo test images found: {pattern}")
            print("Run first:  python train_cnn.py --dataset cifar10")
            return

        print(f"\nClassifying {len(images)} images...")
        print("-" * 60)

        correct = total = 0
        for img_path in images:
            idx, name, conf = run_inference(acc, classifier, img_path,
                                            save_output=not args.no_save)

            basename = os.path.basename(img_path)
            if "_class" in basename:
                true_class = int(basename.split("_class")[1].split(".")[0])
                if idx == true_class:
                    correct += 1
                total += 1

        print("\n" + "=" * 60)
        print("  RESULTS")
        print("=" * 60)
        print(f"  Images: {len(images)}")
        if total > 0:
            print(f"  Accuracy: {correct}/{total} = {100*correct/total:.1f}%")
        print("=" * 60)


if __name__ == "__main__":
    main()
