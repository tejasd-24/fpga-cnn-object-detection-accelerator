# FPGA Based CNN Accelerator

A fully custom **3-layer Convolutional Neural Network accelerator** implemented in Verilog and deployed on a **PYNQ Z2 (Zynq-7020)** FPGA board. Performs real-time object detection via USB webcam at **18 FPS** — all convolution, ReLU, and pooling computed entirely in FPGA hardware at **~7ms per image**.

---

## Highlights

| Metric | Value |
|--------|-------|
| **Real-time FPS** | **22 FPS** (FPGA) vs 3.7 FPS (ARM C-optimized) |
| **FPGA inference time** | **6.8 ms** per image |
| **FPGA speedup vs ARM** | **~5×** faster (vs optimized C on Cortex-A9) |
| **Classification accuracy** | **56.1%** across 6 classes (3.4× random chance) |
| **Parallel compute cores** | 16 convolution cores running simultaneously |
| **On-chip storage** | All feature maps in BRAM — zero external memory access |
| **Input resolution** | 128 × 128 grayscale |
| **Classes** | airplane, cat, zebra, bus, bicycle, donut |

## Architecture

### What Runs on FPGA (Programmable Logic)
All compute-intensive CNN operations run entirely on the FPGA fabric:

```
                                                                                        Input Image (128×128×1)
                                                                                                   │
                                                                                                   ▼
                                                                             ┌─────────────────────────────────────────────┐
                                                                             │         FPGA ACCELERATOR (PL)               │
                                                                             │                                             │
                                                                             │  Layer 1: Conv3×3 (1→16ch) → ReLU → Pool2   │
                                                                             │           128×128 → 64×64, 16 channels      │
                                                                             │                                             │
                                                                             │  Layer 2: Conv3×3 (16→32ch) → ReLU → Pool2  │
                                                                             │           64×64 → 32×32, 32 channels        │
                                                                             │                                             │
                                                                             │  Layer 3: Conv3×3 (32→64ch) → ReLU → Pool2  │
                                                                             │           32×32 → 16×16, 64 channels        │
                                                                             │                                             │
                                                                             │  16 parallel conv cores × accumulator       │
                                                                             │  All feature maps stored in on-chip BRAM    │
                                                                             └─────────────────────────────────────────────┘
                                                                                                   │
                                                                                                   ▼    64 × 16 × 16 feature maps (read via AXI-Lite)
                                                                                                   │
                                                                             ┌─────────────────────────────────────────────┐
                                                                             │         ARM CPU (PS)                        │
                                                                             │                                             │
                                                                             │  Spatial bin pooling (4×4 grid) → 1024 feat │
                                                                             │  Linear classifier → 6-class softmax        │
                                                                             │  CAM-based bounding box generation          │
                                                                             └─────────────────────────────────────────────┘
                                                                                                  │
                                                                                                  ▼
                                                                               Prediction + Bounding Box + Annotated JPEG
```

### Hardware Pipeline Detail
Each of the 3 CNN layers executes this pipeline in hardware:

1. **Line Buffer** — Caches 3 rows for streaming 3×3 convolution
2. **Sliding Window** — Extracts 3×3 patches from line buffer output
3. **16× Conv Cores** — Parallel signed 8-bit multiply-accumulate (MAC)
4. **Accumulator** — Accumulates partial sums across input channels (tiled)
5. **ReLU** — Clamps negative values to zero with configurable right-shift
6. **Max Pooling** — 2×2 stride-2 spatial downsampling

The FSM (`layer_fsm.v`) orchestrates all 3 layers sequentially, managing weight loading, channel tiling, and feature BRAM read/write addressing.

## Project Structure

```
fpga-cnn-object-detection-accelerator/
│
├── rtl/                                  # Verilog RTL source
│   ├── core/                            # CNN datapath modules
│   │   ├── cnn_acc_top.v                # Top-level: datapath + instantiations
│   │   ├── layer_fsm.v                  # FSM: orchestrates 3 layers
│   │   ├── conv_core.v                  # 3×3 signed convolution MAC
│   │   ├── accumulator.v               # Multi-channel partial sum accumulator
│   │   ├── ReLU.v                       # Rectified linear unit
│   │   ├── max_pooling_engine.v         # 2×2 max pooling
│   │   ├── line_buffer.v               # 3-row line buffer for streaming
│   │   ├── sliding_window.v            # 3×3 window extractor
│   │   ├── weight_bram.v               # Weight storage BRAM
│   │   └── feature_bram.v              # Feature map storage BRAM
│   └── axi_wrapper/                     # AXI-Lite interface
│       ├── lyr3_cnn_axi.v              # AXI top wrapper
│       └── lyr3_cnn_axi_slave_lite_v1_0_S00_AXI.v
│
├── fpga/hw/                             # Pre-built FPGA binaries
│   ├── lyr3_cnn.bit                    # Bitstream for PYNQ Z2
│   └── lyr3_cnn.hwh                    # Hardware handoff file
│
├── sim/                                  # Simulation & testbenches
│   ├── module/cc_sim.v                  # Conv core unit test
│   └── top/tb.v                         # Full-system testbench
│
├── software/                             # PYNQ deployment scripts
│   ├── realtime_detect.py              # ★ Real-time webcam detection (FPGA + ARM)
│   ├── pynq_inference.py               # Single-image FPGA inference + visualization
│   ├── arm_cnn.c                       # Optimized C CNN for ARM inference
│   ├── arm_benchmark.py                # ARM-only speed benchmark
│   ├── fast_readout.c                  # C-accelerated FPGA feature readout
│   ├── dump_fpga_features.py           # Dump FPGA features for classifier training
│   ├── dump_arm_features.py            # Dump ARM features for classifier training
│   └── retrain_classifier.py           # Train linear classifier on dumped features
│
├── training/                             # Model training (laptop/GPU)
│   └── train_cnn.py                    # PyTorch quantization-aware CNN training
│
├── weights/                              # Pre-trained model weights
│   ├── weights.bin                     # Quantized CNN weights (23,184 bytes)
│   ├── fc_weight.npy                   # Classifier weights (6×1024)
│   ├── fc_bias.npy                     # Classifier biases (6,)
│   └── classes.json                    # Class name mapping
│
├── Vivado_block_design/                  # Vivado block design screenshot
│   └── Vivado_bd.png
│
└── docs/                                 # Documentation & diagrams
```

## How It Works

### 1. Train the CNN (on laptop/GPU)
```bash
python3 training/train_cnn.py \
    --train-dir ~/coco/train2017 \
    --train-ann ~/coco/annotations/instances_train2017.json \
    --val-dir ~/coco/val2017 \
    --val-ann ~/coco/annotations/instances_val2017.json \
    --output-dir weights
```

This trains a quantization-aware 3-layer CNN in PyTorch, then exports:
- `weights.bin` — 8-bit quantized conv weights (23,184 bytes)
- Test images as 128×128 raw `.bin` files

### 2. Deploy to FPGA
```bash
# Copy bitstream + weights to PYNQ
scp fpga/hw/lyr3_cnn.bit \
    fpga/hw/lyr3_cnn.hwh \
    weights/weights.bin \
    xilinx@<pynq-ip>:/home/xilinx/jupyter_notebooks/CNN/
```

### 3. Dump FPGA Features & Retrain Classifier
```bash
# On PYNQ — extract features from all test images through FPGA
python3 dump_fpga_features.py

# Retrain the linear classifier on actual FPGA features
python3 retrain_classifier.py --features fpga_features.npz
```

### 4. Run Inference
```bash
# Single image classification
python3 pynq_inference.py --weights fpga_weights/weights.bin \
    --image fpga_weights/test_image_200_class2.bin
```

### 5. Retrain Classifier for ARM Mode
```bash
# On PYNQ — dump ARM CNN features from test images
python3 dump_arm_features.py

# Retrain classifier on ARM features
python3 retrain_classifier.py --features arm_features.npz --prefix arm_
```

### 6. Real-Time Webcam Detection
```bash
# FPGA mode — 18 FPS, custom CNN in hardware
python3 realtime_detect.py --mode fpga --res 320x240

# ARM mode — 3.7 FPS, same CNN in optimized C
python3 realtime_detect.py --mode arm --res 320x240
```

Open `http://<pynq-ip>:5000` in a browser for the live MJPEG stream with detection overlay.

## Results

### Per-Class Accuracy
| Class | Accuracy | Samples |
|-------|----------|---------|
| Airplane | 73.2% | 97 |
| Zebra | 71.8% | 85 |
| Donut | 58.1% | 62 |
| Bus | 49.0% | 100 |
| Cat | 45.0% | 100 |
| Bicycle | 43.0% | 100 |
| **Overall** | **56.1%** | **544** |

### Real-Time Performance (USB Webcam)
| Mode | FPS | Inference | Method |
|------|-----|-----------|--------|
| **FPGA** | **18** | 7ms conv + 18ms read | Custom 3-layer CNN in hardware |
| **ARM (C-opt)** | **3.7** | 248ms | Same CNN, optimized C on Cortex-A9 |
| ARM (numpy) | 0.3 | 3300ms | Same CNN, pure numpy (baseline) |

**FPGA speedup: ~5× vs optimized ARM C**

## FPGA Resource Usage

- **Target:** Xilinx Zynq-7020 (PYNQ Z2)
- **Convolution cores:** 16 parallel (processes 16 output channels simultaneously)
- **Accumulator BRAM:** 16 × 4096 × 24-bit (tiled for BRAM efficiency)
- **Feature BRAMs:** 112 channels × 4096 × 8-bit
- **Weight BRAM:** 23,184 × 8-bit
- **Clock:** 50 MHz

## Proof of FPGA Acceleration

Everything below the classifier runs on FPGA hardware — here's the evidence:

1. **RTL modules** — `cnn_acc_top.v` instantiates `conv_core` (MAC), `accumulator`, `ReLU`, and `max_pooling_engine` in a `generate` block — 16 parallel instances
2. **FSM control** — `layer_fsm.v` (500+ lines) orchestrates weight loading, 3-layer processing, tiling, and drain cycles — all in Verilog state machine
3. **DMA data path** — Python sends weights and image pixels to FPGA via AXI DMA, never processes them on ARM
4. **Start/done protocol** — Python pulses a start bit, then polls the FPGA's done flag — the ARM is idle during convolution
5. **Feature readout** — Python reads results from FPGA BRAMs via AXI-Lite registers (`REG_OUTPUT_CH`, `REG_OUTPUT_ADDR`, `REG_OUTPUT_DATA`)
6. **Timing** — 6.8 ms for 3 conv layers is consistent with a 50 MHz hardware pipeline, far too fast for ARM-side convolution

The **only** ARM-side computation is the final classifier: spatial bin pooling → linear layer → softmax → CAM bounding box.

## Requirements

### Hardware
- PYNQ Z2 board (Zynq-7020)
- MicroSD card with PYNQ v2.x image
- Ethernet connection to host

### Software (Training — laptop)
- Python 3.8+
- PyTorch
- pycocotools
- COCO dataset (train2017 + val2017 + annotations)

### Software (Inference — PYNQ)
- Python 3 (pre-installed on PYNQ)
- NumPy, Pillow (pre-installed on PYNQ)
- pynq library (pre-installed on PYNQ)

## Key Design Decisions

- **8-bit quantization** — Weights stored as signed int8, activations as unsigned uint8, matching FPGA's integer arithmetic
- **Tiled accumulation** — Accumulator BRAM reduced from 16K to 4K entries by processing input channels in tiles of 16
- **Spatial bin pooling** — 4×4 grid pooling (1024 features) preserves spatial layout for better shape discrimination vs. global average pooling (64 features)
- **CAM-based localization** — Class Activation Maps with saturated channel filtering and percentile-based thresholding for bounding box generation
- **Class-balanced training** — Inverse-frequency loss weighting handles imbalanced class distribution (62-100 samples per class)

## License

This project is for educational and research purposes.

## Acknowledgments

- [PYNQ](http://www.pynq.io/) framework for Zynq FPGA development
- [MS COCO](https://cocodataset.org/) dataset for training images
- Xilinx Vivado for FPGA synthesis and implementation
