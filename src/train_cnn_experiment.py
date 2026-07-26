"""
train_cnn_experiment.py

Configurable version of the CNN training script for hyperparameter
optimization. Run this multiple times with different --config arguments;
each run appends its result to outputs/hyperparam_results/results.csv,
building up a comparison table for the final presentation.

Example usage:
    python train_cnn_experiment.py --name baseline_frozen --epochs 6 --lr 1e-3 --unfreeze_layers 0
    python train_cnn_experiment.py --name partial_unfreeze --epochs 6 --lr 1e-4 --unfreeze_layers 20
    python train_cnn_experiment.py --name full_unfreeze --epochs 6 --lr 1e-5 --unfreeze_layers 999
    python train_cnn_experiment.py --name bigger_batch --epochs 6 --lr 1e-4 --unfreeze_layers 20 --batch_size 64
    python train_cnn_experiment.py --name more_epochs --epochs 12 --lr 1e-4 --unfreeze_layers 20
"""

import os
import csv
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ---- EDIT THESE TWO PATHS FOR YOUR MACHINE ----
DATA_ROOT = r"C:\Users\chere\Downloads\archive (8)\casting_data\casting_data"
OUT_DIR = r"C:\Users\chere\OneDrive\Documents\GitHub\SricharanCherepally-C986X996-CS898BA-Project\outputs\hyperparam_results"
# ------------------------------------------------

CLASS_TO_LABEL = {"ok_front": 0, "def_front": 1}
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
IMG_SIZE = 160


class CastingDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]


def list_split(split, subset_per_class=None):
    paths, labels = [], []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = os.path.join(DATA_ROOT, split, class_name)
        files = sorted(os.listdir(class_dir))
        if subset_per_class is not None:
            random.Random(SEED).shuffle(files)
            files = files[:subset_per_class]
        for f in files:
            paths.append(os.path.join(class_dir, f))
            labels.append(label)
    return paths, labels


def build_model(unfreeze_layers: int):
    """
    unfreeze_layers: how many of the LAST feature-extractor children to
    leave trainable. 0 = fully frozen backbone (classifier head only).
    999 (or any large number) = fully unfrozen (full fine-tuning).
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    feature_children = list(model.features.children())
    n_total = len(feature_children)
    n_unfreeze = min(unfreeze_layers, n_total)

    for i, child in enumerate(feature_children):
        requires_grad = i >= (n_total - n_unfreeze)
        for p in child.parameters():
            p.requires_grad = requires_grad

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Experiment name, used in the results table")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--unfreeze_layers", type=int, default=0,
                         help="0 = frozen backbone, 999 = fully unfrozen")
    parser.add_argument("--train_subset_per_class", type=int, default=None,
                         help="Leave unset to use the FULL training set")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cpu")
    num_workers = min(8, os.cpu_count() or 1)

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_paths, train_labels = list_split("train", subset_per_class=args.train_subset_per_class)
    test_paths, test_labels = list_split("test")  # always full test set

    print(f"[{args.name}] Train: {len(train_paths)} images | Test: {len(test_paths)} images | "
          f"epochs={args.epochs} lr={args.lr} batch_size={args.batch_size} "
          f"unfreeze_layers={args.unfreeze_layers}")

    train_ds = CastingDataset(train_paths, train_labels, train_tf)
    test_ds = CastingDataset(test_paths, test_labels, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(args.unfreeze_layers).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{args.name}] Trainable params: {trainable:,} / {total:,}")

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total_n = 0.0, 0, 0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_n += labels.size(0)

        print(f"[{args.name}] Epoch {epoch}/{args.epochs}  loss={running_loss/total_n:.4f}  "
              f"train_acc={correct/total_n:.4f}  time={time.time()-t0:.1f}s")

    total_train_time = time.time() - t_start

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            outputs = model(imgs.to(device))
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
    }
    print(f"[{args.name}] RESULT: accuracy={metrics['accuracy']:.4f} "
          f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
          f"train_time={total_train_time:.1f}s")

    # Append to shared results CSV
    results_path = os.path.join(OUT_DIR, "results.csv")
    file_exists = os.path.isfile(results_path)
    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["name", "epochs", "lr", "batch_size", "unfreeze_layers",
                              "train_images", "trainable_params", "train_time_sec",
                              "accuracy", "precision", "recall", "f1"])
        writer.writerow([args.name, args.epochs, args.lr, args.batch_size, args.unfreeze_layers,
                          len(train_paths), trainable, round(total_train_time, 1),
                          round(metrics["accuracy"], 4), round(metrics["precision"], 4),
                          round(metrics["recall"], 4), round(metrics["f1"], 4)])

    print(f"[{args.name}] Appended result to {results_path}")


if __name__ == "__main__":
    main()
