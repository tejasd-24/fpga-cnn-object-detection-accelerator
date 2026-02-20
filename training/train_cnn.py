"""
Train a 3-layer CNN matching the FPGA accelerator.

Uses COCO dataset with 8 categories:
  person, car, airplane, cat, dog, bottle, laptop, phone

Hardware architecture (128×128 grayscale input):
  Layer 0: 1→16 ch, 3×3 conv, ReLU, 2×2 MaxPool  → 64×64×16
  Layer 1: 16→32 ch, 3×3 conv, ReLU, 2×2 MaxPool  → 32×32×32
  Layer 2: 32→64 ch, 3×3 conv, ReLU, 2×2 MaxPool  → 16×16×64

Output files (in fpga_weights/):
  weights.bin       - FPGA weight BRAM (23,184 bytes)
  fc_weight.npy     - Classifier weight matrix (8 × 64)
  fc_bias.npy       - Classifier bias vector (8,)
  classes.json      - Class name list
  test_image_*.bin  - Test images for FPGA inference

Usage:
  python train_cnn.py                           # default: 5000/class, 30 epochs
  python train_cnn.py --max-per-class 0         # use ALL images per class
  python train_cnn.py --epochs 50               # longer training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
import argparse
import json
import random
from PIL import Image


# ── Hardware constants ──────────────────────────────────────────
SHIFTS = (2, 4, 6)                # per-layer ReLU right-shift
QUANT_MAX = 127                   # symmetric int8 range
ACCUM_BITS = 24                   # accumulator width
IMG_SIZE = 128                    # input image size
NUM_CLASSES = 6

# COCO category ID mapping — 6 visually distinct classes
COCO_CATS = {
    'airplane': 5,
    'cat':      17,
    'zebra':    24,
    'bus':      6,
    'bicycle':  2,
    'donut':    60,
}
CLASS_NAMES = list(COCO_CATS.keys())


# ── COCO Classification Dataset ────────────────────────────────
class COCOClassification(Dataset):
    """Single-label classification from COCO detection annotations."""

    def __init__(self, img_dir, ann_file, transform, max_per_class=0):
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transform = transform

        self.samples = []   # list of (img_id, class_idx)

        for class_idx, (name, cat_id) in enumerate(COCO_CATS.items()):
            img_ids = self.coco.getImgIds(catIds=[cat_id])
            random.shuffle(img_ids)
            if max_per_class > 0:
                img_ids = img_ids[:max_per_class]
            for img_id in img_ids:
                self.samples.append((img_id, class_idx))

        random.shuffle(self.samples)

        # Print stats
        counts = {}
        for _, c in self.samples:
            counts[c] = counts.get(c, 0) + 1
        print(f"  Dataset: {len(self.samples)} images from {img_dir}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"    {i}: {name:10s} → {counts.get(i, 0)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, label = self.samples[idx]
        info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, info['file_name'])
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Bit-accurate FPGA simulation ────────────────────────────────
def fpga_conv_layer(x, conv_weight, shift, scale):
    """
    Simulates one FPGA conv layer exactly matching the fixed RTL.
    int8_weights × uint8_pixels → 24-bit accumulator
    → arithmetic right-shift → 24-bit ReLU [0,255] → 2×2 MaxPool
    """
    w_q = (conv_weight * scale).round().clamp(-QUANT_MAX, QUANT_MAX)
    out = torch.nn.functional.conv2d(x, w_q, padding=1)

    M = 2 ** (ACCUM_BITS - 1)
    out = ((out + M) % (2 * M)) - M

    out = torch.div(out, 2.0 ** shift, rounding_mode='floor')
    out = out.clamp(0, 255)
    out = torch.nn.functional.max_pool2d(out, 2)
    return out


# ── CNN Model ───────────────────────────────────────────────────
class FPGA_CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  16, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
        self.qat = False

    def _quant_scale(self):
        all_w = torch.cat([self.conv1.weight.data.flatten(),
                           self.conv2.weight.data.flatten(),
                           self.conv3.weight.data.flatten()])
        return QUANT_MAX / all_w.abs().max().clamp(min=1e-8)

    def forward(self, x):
        if self.qat:
            s = self._quant_scale()
            x = x * 255.0
            x = fpga_conv_layer(x, self.conv1.weight, SHIFTS[0], s)
            x = fpga_conv_layer(x, self.conv2.weight, SHIFTS[1], s)
            x = fpga_conv_layer(x, self.conv3.weight, SHIFTS[2], s)
        else:
            x = torch.nn.functional.max_pool2d(torch.relu(self.conv1(x)), 2)
            x = torch.nn.functional.max_pool2d(torch.relu(self.conv2(x)), 2)
            x = torch.nn.functional.max_pool2d(torch.relu(self.conv3(x)), 2)
        return self.classifier(x)


# ── Extract features ────────────────────────────────────────────
@torch.no_grad()
def extract_features(model, loader, device, verbose=False):
    model.eval()
    s = model._quant_scale()
    all_f, all_l = [], []
    for i, (imgs, labels) in enumerate(loader):
        x = imgs.to(device) * 255.0
        x = fpga_conv_layer(x, model.conv1.weight, SHIFTS[0], s)
        x = fpga_conv_layer(x, model.conv2.weight, SHIFTS[1], s)
        x = fpga_conv_layer(x, model.conv3.weight, SHIFTS[2], s)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        all_f.append(x.cpu().numpy())
        all_l.append(labels.numpy())
        if verbose and i == 0:
            print(f"  Feature range: [{x.min():.1f}, {x.max():.1f}], "
                  f"mean={x.mean():.2f}, nonzero={(x > 0).float().mean()*100:.0f}%")
    return np.vstack(all_f), np.concatenate(all_l)


# ── Export weights ──────────────────────────────────────────────
def export_weights(model, path):
    layers = [
        (model.conv1.weight.data, 16,  1),
        (model.conv2.weight.data, 32, 16),
        (model.conv3.weight.data, 64, 32),
    ]
    all_w = torch.cat([w.flatten() for w, _, _ in layers])
    scale = QUANT_MAX / all_w.abs().max().clamp(min=1e-8).item()

    buf = []
    for w, out_ch, in_ch in layers:
        for batch in range(out_ch // 16):
            for ic in range(in_ch):
                for core in range(16):
                    oc = batch * 16 + core
                    k = (w[oc, ic].flatten() * scale).round().clamp(-127, 127)
                    buf.extend(k.to(torch.int8).cpu().numpy().view(np.uint8).tolist())

    data = np.array(buf, dtype=np.uint8)
    assert len(data) == 23184, f"Expected 23184 weights, got {len(data)}"
    data.tofile(path)
    print(f"  weights.bin: {len(data)} bytes (scale={scale:.2f})")


# ── Export test images ──────────────────────────────────────────
def export_test_images(val_ds, out_dir, num_per_class=10):
    """Export test images as 128×128 raw .bin files for FPGA."""
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    exported = 0
    for i in range(len(val_ds)):
        img, label = val_ds[i]
        c = int(label)
        if counts.get(c, 0) >= num_per_class:
            continue
        counts[c] = counts.get(c, 0) + 1
        raw = (img.squeeze() * 255).byte().numpy()
        fname = f"test_image_{exported}_class{c}.bin"
        raw.tofile(os.path.join(out_dir, fname))
        exported += 1
        if all(counts.get(j, 0) >= num_per_class for j in range(NUM_CLASSES)):
            break
    print(f"  Exported {exported} test images to {out_dir}/")


# ── Training ────────────────────────────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    tx = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_img_dir = os.path.expanduser(args.train_images)
    train_ann     = os.path.expanduser(args.train_ann)
    val_img_dir   = os.path.expanduser(args.val_images)
    val_ann       = os.path.expanduser(args.val_ann)

    # Validate paths
    for path, desc in [(train_img_dir, "Train images"), (train_ann, "Train annotations"),
                        (val_img_dir, "Val images"), (val_ann, "Val annotations")]:
        if not os.path.exists(path):
            print(f"ERROR: {desc} not found at {path}")
            exit(1)

    print("\n── Training set ──")
    train_ds = COCOClassification(train_img_dir, train_ann, tx,
                                   max_per_class=args.max_per_class)

    print("\n── Validation set ──")
    val_ds = COCOClassification(val_img_dir, val_ann, tx,
                                 max_per_class=62)   # balanced (smallest = donut=62)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)

    print(f"\nTrain: {len(train_ds)}  Val: {len(val_ds)}  Classes: {NUM_CLASSES}")

    model = FPGA_CNN(NUM_CLASSES).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit  = nn.CrossEntropyLoss()

    # ── Phase 1: Float training ──
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Float Training ({args.epochs} epochs)")
    print(f"{'='*60}")
    best_acc, best_state = 0.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for imgs, labels in train_ld:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = crit(model(imgs), labels)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            pred = model(imgs).argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        sched.step()

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_ld:
                imgs, labels = imgs.to(device), labels.to(device)
                v_correct += (model(imgs).argmax(1) == labels).sum().item()
                v_total += labels.size(0)
        v_acc = 100.0 * v_correct / v_total
        print(f"  Epoch {epoch:3d}: loss={loss_sum/len(train_ld):.3f}  "
              f"train={100*correct/total:.1f}%  val={v_acc:.1f}%")

        if v_acc > best_acc:
            best_acc = v_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"  Best val accuracy: {best_acc:.1f}%")
    model.load_state_dict(best_state)

    # ── Phase 2: Classifier on FPGA-simulated features ──
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Classifier on Quantized Features")
    print(f"{'='*60}")
    model.qat = True

    train_ld2 = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_ld2   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_feats, train_labels = extract_features(model, train_ld2, device, verbose=True)
    val_feats,   val_labels   = extract_features(model, val_ld2,   device)

    ft = torch.tensor(train_feats, dtype=torch.float32, device=device)
    lt = torch.tensor(train_labels, dtype=torch.long,   device=device)
    vf = torch.tensor(val_feats,   dtype=torch.float32, device=device)
    vl = torch.tensor(val_labels,  dtype=torch.long,    device=device)

    fc = nn.Linear(64, NUM_CLASSES).to(device)
    fc_opt = optim.Adam(fc.parameters(), lr=0.01)
    fc_sched = optim.lr_scheduler.StepLR(fc_opt, step_size=200, gamma=0.3)

    for ep in range(1, 601):
        fc_opt.zero_grad()
        loss = crit(fc(ft), lt)
        loss.backward()
        fc_opt.step()
        fc_sched.step()
        if ep % 100 == 0:
            with torch.no_grad():
                ta = (fc(ft).argmax(1) == lt).float().mean().item() * 100
                va = (fc(vf).argmax(1) == vl).float().mean().item() * 100
            print(f"  FC epoch {ep:4d}: train={ta:.1f}%  val={va:.1f}%  loss={loss.item():.3f}")

    # ── Export ──
    print(f"\n{'='*60}")
    print(f"  EXPORTING FILES → {out}/")
    print(f"{'='*60}")

    export_weights(model, os.path.join(out, 'weights.bin'))

    w = fc.weight.detach().cpu().numpy().astype(np.float32)
    b = fc.bias.detach().cpu().numpy().astype(np.float32)
    np.save(os.path.join(out, 'fc_weight.npy'), w)
    np.save(os.path.join(out, 'fc_bias.npy'), b)
    print(f"  fc_weight.npy: {w.shape}  fc_bias.npy: {b.shape}")

    with open(os.path.join(out, 'classes.json'), 'w') as f:
        json.dump(CLASS_NAMES, f)
    print(f"  classes.json: {CLASS_NAMES}")

    # Export test images for FPGA
    export_test_images(val_ds, out, num_per_class=30)

    print(f"\n  Done! Copy {out}/ to PYNQ and run inference.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train CNN for FPGA accelerator (COCO)')
    p.add_argument('--epochs',     type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr',         type=float, default=0.001)
    p.add_argument('--max-per-class', type=int, default=5000,
                   help='Max images per class for training (0 = all)')
    p.add_argument('--train-images', default='/media/tejas/New Volume/coco/train2017',
                   help='Path to COCO train2017 images')
    p.add_argument('--train-ann',    default='~/coco/annotations/instances_train2017.json',
                   help='Path to COCO train2017 annotations')
    p.add_argument('--val-images',   default='~/coco/val2017',
                   help='Path to COCO val2017 images')
    p.add_argument('--val-ann',      default='~/coco/annotations/instances_val2017.json',
                   help='Path to COCO val2017 annotations')
    p.add_argument('--output-dir', default='./fpga_weights')
    train(p.parse_args())
