"""
Retrain Classifier on Real FPGA Features
==========================================
Runs on your laptop. Loads the .npz file of features dumped from the
FPGA by dump_fpga_features.py, trains a linear classifier on them,
and exports the new fc_weight.npy + fc_bias.npy for the PYNQ.

Usage:
  python3 retrain_classifier.py                             # default
  python3 retrain_classifier.py --features fpga_features.npz
  python3 retrain_classifier.py --lr 0.01 --epochs 1000

After retraining, copy the new weights to the PYNQ:
  scp fpga_weights/fc_weight.npy fpga_weights/fc_bias.npy \\
    xilinx@192.168.2.99:/home/xilinx/jupyter_notebooks/CNN/fpga_weights/
"""

import numpy as np
import os
import json
import argparse


def train_linear_classifier(features, labels, num_classes,
                            lr=0.01, epochs=1000, val_split=0.2,
                            verbose=True):
    """
    Train a linear classifier using vanilla numpy (no PyTorch needed).
    Implements: softmax cross-entropy loss + SGD with momentum.

    Args:
        features:  (N, D) float32 — pooled features
        labels:    (N,)   int     — class labels
        num_classes: int
        lr: learning rate
        epochs: number of training epochs
        val_split: fraction held out for validation

    Returns:
        (weight, bias) — numpy arrays
    """
    N, D = features.shape

    # Shuffle and split
    rng = np.random.RandomState(42)
    indices = rng.permutation(N)
    n_val = max(1, int(N * val_split))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]

    # Class-balanced loss weights (inverse frequency)
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_counts = np.maximum(class_counts, 1)  # avoid div by zero
    class_weights = (1.0 / class_counts)
    class_weights = class_weights / class_weights.sum() * num_classes  # normalize
    sample_weights = class_weights[y_train]  # per-sample weight

    # Initialize weights
    W = rng.randn(D, num_classes).astype(np.float32) * 0.01
    b = np.zeros(num_classes, dtype=np.float32)

    # Momentum
    vW = np.zeros_like(W)
    vb = np.zeros_like(b)
    momentum = 0.9

    best_val_acc = 0
    best_W, best_b = W.copy(), b.copy()

    for epoch in range(epochs):
        # Forward
        logits = X_train @ W + b              # (N_train, C)
        logits -= logits.max(axis=1, keepdims=True)  # numerical stability
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Cross-entropy loss (class-balanced) + L2 regularization
        N_t = len(y_train)
        reg_lambda = 0.001  # L2 regularization strength
        per_sample_loss = -np.log(probs[np.arange(N_t), y_train] + 1e-10)
        loss = (per_sample_loss * sample_weights).mean()
        loss += 0.5 * reg_lambda * (W * W).sum()

        # Backward
        dlogits = probs.copy()
        dlogits[np.arange(N_t), y_train] -= 1
        dlogits *= sample_weights[:, None]  # apply class weights to gradients
        dlogits /= N_t

        dW = X_train.T @ dlogits + reg_lambda * W  # + L2 grad
        db = dlogits.sum(axis=0)              # (C,)

        # SGD with momentum
        vW = momentum * vW - lr * dW
        vb = momentum * vb - lr * db
        W += vW
        b += vb

        # Validate every 100 epochs
        if (epoch + 1) % 100 == 0 or epoch == 0:
            # Training accuracy
            train_pred = (X_train @ W + b).argmax(axis=1)
            train_acc = (train_pred == y_train).mean() * 100

            # Validation accuracy
            val_pred = (X_val @ W + b).argmax(axis=1)
            val_acc = (val_pred == y_val).mean() * 100

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_W, best_b = W.copy(), b.copy()

            if verbose:
                print(f"  Epoch {epoch+1:4d}: loss={loss:.4f} "
                      f"train={train_acc:.1f}% val={val_acc:.1f}%")

        # Learning rate decay
        if (epoch + 1) % 300 == 0:
            lr *= 0.5

    return best_W.T, best_b  # (C, D) to match PyTorch convention


def main():
    parser = argparse.ArgumentParser(
        description="Retrain classifier on FPGA features"
    )
    parser.add_argument("--features", default="fpga_features.npz",
                        help="Path to dumped FPGA features (.npz)")
    parser.add_argument("--output-dir", default="fpga_weights",
                        help="Directory to save new classifier weights")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--classes", default="fpga_weights/classes.json",
                        help="Path to classes.json")
    parser.add_argument("--prefix", default="",
                        help="Prefix for output files, e.g. 'arm_' → arm_fc_weight.npy")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve(p):
        return os.path.join(script_dir, p) if not os.path.isabs(p) else p

    # ── Load features ──
    feat_path = resolve(args.features)
    print("=" * 60)
    print("  RETRAIN CLASSIFIER ON FPGA FEATURES")
    print("=" * 60)
    print(f"Loading: {feat_path}")

    data = np.load(feat_path)
    features = data['features']    # (N, 64, 256) uint8
    labels = data['labels']        # (N,)
    shifts = data.get('shifts', None)

    print(f"  Samples:  {features.shape[0]}")
    print(f"  Channels: {features.shape[1]}")
    print(f"  Spatial:  {features.shape[2]} (16×16)")
    if shifts is not None:
        print(f"  Shifts:   {shifts}")

    # Filter out unlabeled images (label == -1)
    valid = labels >= 0
    features = features[valid]
    labels = labels[valid]
    print(f"  Labeled:  {features.shape[0]}")

    # ── Feature statistics ──
    print(f"\nFeature statistics:")
    print(f"  Range:    [{features.min()}, {features.max()}]")
    print(f"  Mean:     {features.astype(float).mean():.2f}")
    print(f"  Nonzero:  {(features > 0).mean()*100:.1f}%")

    # Per-channel statistics
    channel_means = features.astype(float).mean(axis=(0, 2))  # (64,)
    active = (channel_means > 1.0).sum()
    print(f"  Active channels (mean > 1.0): {active}/{features.shape[1]}")

    # Show top-5 most active channels
    top5 = np.argsort(channel_means)[::-1][:5]
    print(f"  Top-5 channels: {list(top5)} "
          f"(means: {[f'{channel_means[ch]:.1f}' for ch in top5]})")

    # ── Spatial Bin Pooling (4×4 grid) instead of Global Avg Pool ──
    # Global avg pool loses spatial info (donut ring vs bus rectangle look same!)
    # Split each 16×16 feature into 4×4 grid → 16 bins per channel → 64×16 = 1024 features
    N_samples = features.shape[0]
    n_ch = features.shape[1]
    feat_maps = features.astype(np.float32).reshape(N_samples, n_ch, 16, 16)

    grid = 4  # 4×4 grid of 4×4 bins
    bin_size = 16 // grid
    pooled_bins = np.zeros((N_samples, n_ch, grid * grid), dtype=np.float32)
    for r in range(grid):
        for c in range(grid):
            patch = feat_maps[:, :, r*bin_size:(r+1)*bin_size, c*bin_size:(c+1)*bin_size]
            pooled_bins[:, :, r * grid + c] = patch.mean(axis=(2, 3))

    # Flatten: (N, 64, 16) → (N, 1024)
    pooled = pooled_bins.reshape(N_samples, -1)

    # Normalize to [0, 1] — critical to prevent softmax saturation
    pooled = pooled / 255.0

    print(f"\nSpatial bin pooled features ({grid}×{grid} grid):")
    print(f"  Shape:    {pooled.shape}")
    print(f"  Range:    [{pooled.min():.4f}, {pooled.max():.4f}]")
    print(f"  Mean:     {pooled.mean():.4f}")
    print(f"  Std:      {pooled.std():.4f}")

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    num_classes = len(unique)
    print(f"\nClass distribution ({num_classes} classes):")
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} samples")

    # ── Load class names ──
    classes_path = resolve(args.classes)
    class_names = None
    if os.path.exists(classes_path):
        with open(classes_path) as f:
            class_names = json.load(f)
        print(f"  Names: {class_names}")

    # ── Train ──
    print(f"\n{'─' * 60}")
    print(f"Training linear classifier (lr={args.lr}, epochs={args.epochs})")
    print(f"{'─' * 60}")

    W, bias = train_linear_classifier(
        pooled, labels, num_classes,
        lr=args.lr, epochs=args.epochs, verbose=True
    )

    # ── Final evaluation ──
    logits = pooled @ W.T + bias
    preds = logits.argmax(axis=1)
    overall_acc = (preds == labels).mean() * 100
    print(f"\nOverall accuracy: {overall_acc:.1f}%")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for cls in unique:
        mask = labels == cls
        cls_acc = (preds[mask] == labels[mask]).mean() * 100
        name = class_names[cls] if class_names else str(cls)
        print(f"  {name:12s}: {cls_acc:.1f}% ({mask.sum()} samples)")

    # Confusion matrix (compact)
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    print(f"{'':>12s}", end="")
    for cls in unique:
        name = class_names[cls][:4] if class_names else str(cls)
        print(f" {name:>5s}", end="")
    print()
    for true_cls in unique:
        name = class_names[true_cls][:8] if class_names else str(true_cls)
        print(f"  {name:>10s}", end="")
        mask = labels == true_cls
        for pred_cls in unique:
            cnt = ((preds[mask]) == pred_cls).sum()
            print(f" {cnt:5d}", end="")
        print()

    # ── Save ──
    output_dir = resolve(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    fc_weight_path = os.path.join(output_dir, f"{args.prefix}fc_weight.npy")
    fc_bias_path = os.path.join(output_dir, f"{args.prefix}fc_bias.npy")

    np.save(fc_weight_path, W.astype(np.float32))   # (num_classes, 64)
    np.save(fc_bias_path, bias.astype(np.float32))   # (num_classes,)

    print(f"\n{'=' * 60}")
    print(f"  SAVED NEW CLASSIFIER WEIGHTS")
    print(f"{'=' * 60}")
    print(f"  {fc_weight_path}  shape={W.shape}")
    print(f"  {fc_bias_path}  shape={bias.shape}")
    print(f"\nNow copy to PYNQ and re-run inference:")
    print(f"  scp {fc_weight_path} {fc_bias_path} \\")
    print(f"    xilinx@192.168.2.99:/home/xilinx/jupyter_notebooks/CNN/fpga_weights/")
    print(f"  # Then on PYNQ:")
    print(f"  sudo /usr/local/share/pynq-venv/bin/python3 pynq_inference.py")


if __name__ == "__main__":
    main()
