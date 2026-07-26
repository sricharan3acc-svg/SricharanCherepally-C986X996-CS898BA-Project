"""
train_cnn_clean.py

Final, leakage-free training run. Uses the winning hyperparameters from
the optimization sweep (20 layers unfrozen, lr=1e-4, batch_size=32,
12 epochs) but first removes any training image that is an exact
duplicate of a test image (found via MD5 file hash), so the reported
test accuracy reflects genuinely unseen data.
"""

import os
import csv
import time
import random
import hashlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ---- EDIT THESE TWO PATHS FOR YOUR MACHINE (same as train_cnn_experiment.py) ----
DATA_ROOT = r"C:\Users\chere\Downloads\archive (8)\casting_data\casting_data"
OUT_DIR = r"C:\Users\chere\OneDrive\Documents\GitHub\SricharanCherepally-C986X996-CS898BA-Project\outputs\cnn_clean_results"
# ----------------------------------------------------------------------------------

CLASS_TO_LABEL = {"ok_front": 0, "def_front": 1}
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
IMG_SIZE = 160

# Winning config from the hyperparameter sweep
EPOCHS = 12
LR = 1e-4
BATCH_SIZE = 32
UNFREEZE_LAYERS = 20


def hash_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


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


def list_split(split):
    paths, labels = [], []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = os.path.join(DATA_ROOT, split, class_name)
        for f in sorted(os.listdir(class_dir)):
            paths.append(os.path.join(class_dir, f))
            labels.append(label)
    return paths, labels


def remove_leaked_duplicates(train_paths, train_labels, test_paths):
    print("Checking for train/test duplicate images (by file hash)...")
    test_hashes = {hash_file(p) for p in test_paths}

    clean_paths, clean_labels = [], []
    removed = 0
    for p, l in zip(train_paths, train_labels):
        if hash_file(p) in test_hashes:
            removed += 1
            continue
        clean_paths.append(p)
        clean_labels.append(l)

    print(f"Removed {removed} training images that were exact duplicates of test images.")
    print(f"Clean training set: {len(clean_paths)} images (was {len(train_paths)})")
    return clean_paths, clean_labels


def build_model(unfreeze_layers: int):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    feature_children = list(model.features.children())
    n_total = len(feature_children)
    n_unfreeze = min(unfreeze_layers, n_total)
    for i, child in enumerate(feature_children):
        for p in child.parameters():
            p.requires_grad = i >= (n_total - n_unfreeze)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model


def main():
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

    train_paths, train_labels = list_split("train")
    test_paths, test_labels = list_split("test")

    train_paths, train_labels = remove_leaked_duplicates(train_paths, train_labels, test_paths)

    print(f"Final training set: {len(train_paths)} images | Test set (untouched): {len(test_paths)} images")
    print(f"Config: epochs={EPOCHS} lr={LR} batch_size={BATCH_SIZE} unfreeze_layers={UNFREEZE_LAYERS}")

    train_ds = CastingDataset(train_paths, train_labels, train_tf)
    test_ds = CastingDataset(test_paths, test_labels, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    model = build_model(UNFREEZE_LAYERS).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LR)

    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
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
        print(f"Epoch {epoch}/{EPOCHS}  loss={running_loss/total_n:.4f}  "
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
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n=== CLEAN RESULT (no train/test leakage) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  confusion matrix [normal, defective]:\n{cm}")
    print(f"  train_time={total_train_time:.1f}s")

    with open(os.path.join(OUT_DIR, "clean_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["accuracy", "precision", "recall", "f1", "train_time_sec",
                          "train_images_after_dedup", "test_images"])
        writer.writerow([round(metrics["accuracy"], 4), round(metrics["precision"], 4),
                          round(metrics["recall"], 4), round(metrics["f1"], 4),
                          round(total_train_time, 1), len(train_paths), len(test_paths)])

    print(f"\nSaved to {OUT_DIR}\\clean_metrics.csv")


if __name__ == "__main__":
    main()
