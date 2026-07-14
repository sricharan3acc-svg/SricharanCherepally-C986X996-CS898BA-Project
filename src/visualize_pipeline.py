"""
visualize_pipeline.py

Runs the preprocessing pipeline on sample defective + normal images and
saves a labeled before/after grid image for each, so the pipeline's
effect is visually demonstrable on the midterm slides.
"""

import os
import cv2
import matplotlib.pyplot as plt
from preprocessing import run_pipeline

DATA_DIR = DATA_DIR = r"C:\Users\chere\Downloads\archive (8)\casting_data\casting_data\train"
OUT_DIR = r"C:\Users\chere\OneDrive\Documents\GitHub\SricharanCherepally-C986X996-CS898BA-Project\outputs\preprocessing_examples"


def visualize_one(image_path: str, title: str, out_path: str):
    result = run_pipeline(image_path)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    panels = [
        ("1. Original (grayscale)", result["original"]),
        ("2. CLAHE enhanced", result["clahe"]),
        ("3. Threshold + morph clean", result["binary_clean"]),
        ("4. Canny edges", result["edges"]),
        ("5. Candidate voids boxed", cv2.cvtColor(result["blob_viz"], cv2.COLOR_BGR2RGB)),
    ]

    for ax, (label, img) in zip(axes, panels):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    stats = result["blob_stats"]
    fig.suptitle(
        f"{title}  |  candidate voids: {stats['num_candidate_voids']}, "
        f"total void area: {stats['total_void_area_px']}px",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}  |  stats: {stats}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    def_dir = os.path.join(DATA_DIR, "def_front")
    ok_dir = os.path.join(DATA_DIR, "ok_front")

    def_samples = sorted(os.listdir(def_dir))[:2]
    ok_samples = sorted(os.listdir(ok_dir))[:2]

    for i, fname in enumerate(def_samples):
        visualize_one(
            os.path.join(def_dir, fname),
            title=f"DEFECTIVE sample {i+1} ({fname})",
            out_path=os.path.join(OUT_DIR, f"defective_{i+1}.png"),
        )

    for i, fname in enumerate(ok_samples):
        visualize_one(
            os.path.join(ok_dir, fname),
            title=f"NORMAL sample {i+1} ({fname})",
            out_path=os.path.join(OUT_DIR, f"normal_{i+1}.png"),
        )


if __name__ == "__main__":
    main()
