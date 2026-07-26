"""
live_demo.py

VIRTUAL DEMONSTRATION SCRIPT
Loads the trained Random Forest model and runs it live on a handful of
real, held-out test images -- one at a time. Each image is displayed
on screen with its prediction and true label printed to the console.
This script is meant to be screen-recorded for the "Virtual
Demonstration" requirement of the final presentation.

Run with: python live_demo.py
"""

import os
import time
import random
import numpy as np
import joblib
import matplotlib.pyplot as plt

from preprocessing import load_grayscale
from features import extract_features

# ---- EDIT THESE PATHS FOR YOUR MACHINE ----
DATA_ROOT = r"C:\Users\chere\Downloads\archive (8)\casting_data\casting_data"
MODEL_PATH = r"C:\Users\chere\OneDrive\Documents\GitHub\SricharanCherepally-C986X996-CS898BA-Project\outputs\demo_model.joblib"
# ---------------------------------------------

LABEL_NAMES = {0: "NORMAL", 1: "DEFECTIVE"}
SECONDS_PER_IMAGE = 4  # how long each image stays on screen before auto-advancing


def pick_demo_images(n_per_class=3, seed=7):
    rng = random.Random(seed)
    images = []
    for class_name, label in [("ok_front", 0), ("def_front", 1)]:
        class_dir = os.path.join(DATA_ROOT, "test", class_name)
        files = sorted(os.listdir(class_dir))
        chosen = rng.sample(files, n_per_class)
        for f in chosen:
            images.append((os.path.join(class_dir, f), label))
    rng.shuffle(images)
    return images


def main():
    print("=" * 60)
    print("  CASTING DEFECT DETECTION -- LIVE DEMONSTRATION")
    print("=" * 60)
    print(f"\nLoading trained model from:\n  {MODEL_PATH}\n")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.\n")

    demo_images = pick_demo_images()
    print(f"Running live predictions on {len(demo_images)} held-out test images "
          f"(never seen during training)...\n")
    time.sleep(2)

    correct = 0

    # Single persistent window that updates in place -- much more reliable
    # for screen recording than opening a new window per image.
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title("Casting Defect Detection - Live Demo")
    plt.show(block=False)
    plt.pause(0.5)

    for i, (path, true_label) in enumerate(demo_images, 1):
        fname = os.path.basename(path)
        print("-" * 60)
        print(f"[{i}/{len(demo_images)}] Loading image: {fname}")

        gray = load_grayscale(path)
        feat = extract_features(gray).reshape(1, -1)

        pred = model.predict(feat)[0]
        proba = model.predict_proba(feat)[0]
        confidence = proba[pred] * 100

        pred_name = LABEL_NAMES[pred]
        true_name = LABEL_NAMES[true_label]
        is_correct = (pred == true_label)
        correct += int(is_correct)
        mark = "CORRECT" if is_correct else "WRONG"

        print(f"  True label:      {true_name}")
        print(f"  Predicted label: {pred_name}  (confidence: {confidence:.1f}%)")
        print(f"  Result: {mark}")

        # Update the SAME window/axes instead of creating a new figure
        ax.clear()
        ax.imshow(gray, cmap="gray")
        ax.axis("off")
        color = "green" if is_correct else "red"
        ax.set_title(
            f"[{i}/{len(demo_images)}]  True: {true_name}   |   Predicted: {pred_name} ({confidence:.1f}%)\n{mark}",
            fontsize=13, color=color, fontweight="bold"
        )
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(SECONDS_PER_IMAGE)

    print("-" * 60)
    print(f"\nDemo complete: {correct}/{len(demo_images)} correct "
          f"({100*correct/len(demo_images):.1f}%) on images never seen during training.")

    plt.ioff()
    print("\nClose the image window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
