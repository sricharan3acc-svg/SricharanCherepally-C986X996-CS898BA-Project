"""
run_clahe_ablation.py

Image Analysis Evaluation: tests whether CLAHE actually improves
classifier performance, by extracting features WITHOUT CLAHE and
training the same SVM/Random Forest setup for direct comparison
against the main (with-CLAHE) results from train_baseline.py.
"""

import os
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocessing import load_grayscale
from features_no_clahe import extract_features_no_clahe

# ---- EDIT THESE PATHS FOR YOUR MACHINE ----
DATA_ROOT = r"C:\Users\chere\Downloads\archive (8)\casting_data\casting_data"
OUT_DIR = r"C:\Users\chere\OneDrive\Documents\GitHub\SricharanCherepally-C986X996-CS898BA-Project\outputs\clahe_ablation"
# ---------------------------------------------

CLASS_TO_LABEL = {"def_front": 1, "ok_front": 0}


def process_split(split):
    X, y = [], []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = os.path.join(DATA_ROOT, split, class_name)
        for fname in sorted(os.listdir(class_dir)):
            gray = load_grayscale(os.path.join(class_dir, fname))
            X.append(extract_features_no_clahe(gray))
            y.append(label)
    return np.array(X), np.array(y)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Extracting features WITHOUT CLAHE (for comparison against the main pipeline)...")
    t0 = time.time()
    X_train, y_train = process_split("train")
    X_test, y_test = process_split("test")
    print(f"Done in {time.time()-t0:.1f}s. Train: {X_train.shape}, Test: {X_test.shape}")

    np.save(os.path.join(OUT_DIR, "X_train_noclahe.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "y_train_noclahe.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "X_test_noclahe.npy"), X_test)
    np.save(os.path.join(OUT_DIR, "y_test_noclahe.npy"), y_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []

    svm = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
    svm.fit(X_train_s, y_train)
    y_pred = svm.predict(X_test_s)
    svm_metrics = {
        "model": "SVM (no CLAHE)",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    print("SVM (no CLAHE):", svm_metrics)
    results.append(svm_metrics)

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_metrics = {
        "model": "Random Forest (no CLAHE)",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    print("Random Forest (no CLAHE):", rf_metrics)
    results.append(rf_metrics)

    import csv
    with open(os.path.join(OUT_DIR, "ablation_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "precision", "recall", "f1"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nSaved results to {OUT_DIR}\\ablation_results.csv")
    print("\nCompare these against your main train_baseline.py results (with CLAHE):")
    print("  SVM (with CLAHE):            acc=0.9105  f1=0.9306")
    print("  Random Forest (with CLAHE):  acc=0.9287  f1=0.9439")


if __name__ == "__main__":
    main()
