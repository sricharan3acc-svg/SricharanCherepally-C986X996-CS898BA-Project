"""
extract_dataset_features.py

Runs feature extraction across the full train/ and test/ splits and
saves the resulting feature matrices + labels to disk, so the
(slow-ish) extraction step only needs to run once.
"""

import os
import time
import numpy as np

from preprocessing import load_grayscale
from features import extract_features, FEATURE_NAMES

DATA_ROOT = r"C:\Users\chere\Downloads\archive (8)\casting_data\casting_data"
OUT_DIR = r"C:\Users\chere\OneDrive\Documents\GitHub\SricharanCherepally-C986X996-CS898BA-Project\outputs\features"

# label convention: 1 = defective, 0 = normal
CLASS_TO_LABEL = {"def_front": 1, "ok_front": 0}


def process_split(split: str):
    X, y, paths = [], [], []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = os.path.join(DATA_ROOT, split, class_name)
        files = sorted(os.listdir(class_dir))
        for fname in files:
            fpath = os.path.join(class_dir, fname)
            try:
                gray = load_grayscale(fpath)
                feat = extract_features(gray)
            except Exception as e:
                print(f"  skipping {fpath}: {e}")
                continue
            X.append(feat)
            y.append(label)
            paths.append(fpath)
    return np.array(X), np.array(y), paths


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for split in ["train", "test"]:
        print(f"Processing split: {split}")
        t0 = time.time()
        X, y, paths = process_split(split)
        t1 = time.time()
        print(f"  {split}: X={X.shape}, y={y.shape}, "
              f"defective={int(y.sum())}, normal={int((y==0).sum())}, "
              f"time={t1-t0:.1f}s")

        np.save(os.path.join(OUT_DIR, f"X_{split}.npy"), X)
        np.save(os.path.join(OUT_DIR, f"y_{split}.npy"), y)
        with open(os.path.join(OUT_DIR, f"paths_{split}.txt"), "w") as f:
            f.write("\n".join(paths))

    with open(os.path.join(OUT_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(FEATURE_NAMES))

    print("Done. Feature names saved to feature_names.txt")


if __name__ == "__main__":
    main()
