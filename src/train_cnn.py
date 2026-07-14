"""
train_cnn.py

Approach B from the pitch: CNN transfer learning.
Fine-tunes a pretrained MobileNetV2 on the casting images.

Kept intentionally lightweight (small subset, few epochs, CPU-only) for
a first-pass midterm number. Full hyperparameter optimization on the
complete dataset is planned for the final presentation.
"""

import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

DATA_ROOT = r"C:\Users\chere\Downloads\archive (8)\casting_data\casting_data"
OUT_DIR = r"C:\Users\chere\OneDrive\Documents\GitHub\SricharanCherepally-C986X996-CS898BA-Project\outputs\cnn_results"
CLASS_TO_LABEL = {"ok_front": 0, "def_front": 1}

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Keep this a genuine "first pass": subsample train set for CPU-feasible
# runtime, use the FULL test set for a trustworthy evaluation number.
TRAIN_SUBSET_PER_CLASS = 600
BATCH_SIZE = 32
EPOCHS = 4
LR = 1e-3
IMG_SIZE = 160  # smaller than 224 to keep CPU epochs fast


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


def build_model():
    try:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        pretrained = True
    except Exception as e:
        # Some sandboxed/offline environments can't reach the weights host.
        # Fall back to random init so the script still runs end-to-end, but
        # this is NOT a true transfer-learning result -- re-run with
        # internet access for the real number.
        print(f"[WARNING] Could not download pretrained weights ({e}). "
              f"Falling back to RANDOM INITIALIZATION. This run is a sanity "
              f"check only, not the transfer-learning result for the slide.")
        model = models.mobilenet_v2(weights=None)
        pretrained = False

    if pretrained:
        # freeze the pretrained feature extractor, only train the classifier head
        for param in model.features.parameters():
            param.requires_grad = False
    # else: leave everything trainable -- frozen RANDOM conv weights would
    # produce meaningless features, so a from-scratch run needs full training

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model._pretrained_flag = pretrained
    return model


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cpu")

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

    train_paths, train_labels = list_split("train", subset_per_class=TRAIN_SUBSET_PER_CLASS)
    test_paths, test_labels = list_split("test")  # full test set

    print(f"Train subset: {len(train_paths)} images "
          f"({sum(train_labels)} defective, {len(train_labels)-sum(train_labels)} normal)")
    print(f"Test (full):  {len(test_paths)} images "
          f"({sum(test_labels)} defective, {len(test_labels)-sum(test_labels)} normal)")

    train_ds = CastingDataset(train_paths, train_labels, train_tf)
    test_ds = CastingDataset(test_paths, test_labels, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LR)
    print(f"Pretrained ImageNet weights loaded: {model._pretrained_flag}")

    history = []
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
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
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        t1 = time.time()
        print(f"Epoch {epoch}/{EPOCHS}  loss={epoch_loss:.4f}  train_acc={epoch_acc:.4f}  "
              f"time={t1-t0:.1f}s")
        history.append({"epoch": epoch, "loss": epoch_loss, "train_acc": epoch_acc})

    print(f"Total training time: {time.time()-t_start:.1f}s")

    # --- Evaluate on full test set ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
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
    print("\n=== CNN (MobileNetV2 transfer learning, first pass) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  confusion matrix [normal, defective]:\n{cm}")

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "confusion_matrix": cm.tolist(),
                    "history": history, "pretrained_imagenet_weights": model._pretrained_flag,
                    "config": {"train_subset_per_class": TRAIN_SUBSET_PER_CLASS,
                               "epochs": EPOCHS, "img_size": IMG_SIZE,
                               "batch_size": BATCH_SIZE, "lr": LR}}, f, indent=2)

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "mobilenetv2_first_pass.pt"))
    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
