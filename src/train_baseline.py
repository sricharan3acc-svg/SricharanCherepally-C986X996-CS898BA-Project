"""
train_baseline.py

Trains the classical baseline (Approach A from the pitch) on the
extracted feature vectors: LBP + GLCM texture descriptors + edge
density + blob/void stats -> SVM and Random Forest classifiers,
evaluated on the held-out test split.
"""

import os
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

FEAT_DIR = r"C:\Users\chere\OneDrive\Documents\GitHub\SricharanCherepally-C986X996-CS898BA-Project\outputs\features"
OUT_DIR = r"C:\Users\chere\OneDrive\Documents\GitHub\SricharanCherepally-C986X996-CS898BA-Project\outputs\baseline_results"


def load_split(split):
    X = np.load(os.path.join(FEAT_DIR, f"X_{split}.npy"))
    y = np.load(os.path.join(FEAT_DIR, f"y_{split}.npy"))
    return X, y


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["normal", "defective"])
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  confusion matrix (rows=true, cols=pred) [normal, defective]:\n{cm}")
    print(report)
    return metrics, cm, report


def plot_confusion(cm, name, out_path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["normal", "defective"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{name} — Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # --- SVM (RBF kernel) ---
    svm = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
    svm.fit(X_train_s, y_train)
    metrics_svm, cm_svm, report_svm = evaluate("SVM (RBF)", svm, X_test_s, y_test)
    plot_confusion(cm_svm, "SVM (RBF)", os.path.join(OUT_DIR, "confusion_svm.png"))
    results["svm_rbf"] = metrics_svm

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)  # RF doesn't need scaling
    metrics_rf, cm_rf, report_rf = evaluate("Random Forest", rf, X_test, y_test)
    plot_confusion(cm_rf, "Random Forest", os.path.join(OUT_DIR, "confusion_rf.png"))
    results["random_forest"] = metrics_rf

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(OUT_DIR, "classification_reports.txt"), "w") as f:
        f.write("=== SVM (RBF) ===\n" + report_svm + "\n\n")
        f.write("=== Random Forest ===\n" + report_rf + "\n")

    print(f"\nSaved metrics + confusion matrices to {OUT_DIR}")


if __name__ == "__main__":
    main()
