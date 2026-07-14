"""
preprocessing.py

Image processing pipeline for casting defect detection.
Implements the 4-step strategy from the project pitch (beyond basic augmentation):
  1. Contrast enhancement (CLAHE)
  2. Thresholding + morphological operations
  3. Edge detection (Canny)
  4. Connected-component / blob analysis

Each function takes and returns a numpy array (grayscale or BGR as noted).
"""

import cv2
import numpy as np


def load_grayscale(path: str) -> np.ndarray:
    """Load an image from disk as grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.5, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    Step 1: Contrast Limited Adaptive Histogram Equalization.
    Brings out subtle blow-holes and pinholes that are barely visible
    against the uniform gray metal surface by boosting local contrast.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def threshold_and_clean(gray_enhanced: np.ndarray):
    """
    Step 2: Thresholding + morphological operations.
    Isolates the rim and inner ring from the background and removes noise
    before feature extraction.

    Returns:
        binary_mask: cleaned binary mask (0/255) of the foreground part
        thresh_raw: the raw Otsu threshold output before cleanup (for comparison)
    """
    # Otsu's method auto-picks a threshold given the image's histogram
    _, thresh_raw = cv2.threshold(
        gray_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological opening (erode then dilate) removes small noise specks
    # Morphological closing (dilate then erode) fills small holes in the part
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thresh_raw, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed, thresh_raw


def canny_edges(gray_enhanced: np.ndarray, low_thresh: int = 40, high_thresh: int = 120) -> np.ndarray:
    """
    Step 3: Canny edge detection.
    Highlights irregular, jagged rim boundaries characteristic of flash
    and burr defects. Run on the CLAHE-enhanced image so subtle edges
    aren't lost to low contrast.
    """
    blurred = cv2.GaussianBlur(gray_enhanced, (3, 3), 0)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    return edges


def blob_analysis(binary_mask: np.ndarray, min_area: int = 4, max_area: int = 2000):
    """
    Step 4: Connected-component / blob analysis.
    Counts and sizes void/pit regions inside the part, turning visual
    defects (blow-holes, pinholes) into quantitative engineered features.

    We look for small dark regions INSIDE the part's own silhouette
    (candidate voids). The image background (outside the circular part)
    is explicitly excluded, otherwise dark corners around the part get
    misread as "voids" and pollute the count for every image, defective
    or not.

    Returns:
        stats: dict of engineered features
        labeled_viz: mask image with candidate void blobs highlighted (BGR)
    """
    part_num_labels, part_labels, part_stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    if part_num_labels <= 1:
        part_silhouette = np.zeros_like(binary_mask)
    else:
        largest_label = 1 + int(np.argmax(part_stats[1:, cv2.CC_STAT_AREA]))
        part_silhouette = np.where(part_labels == largest_label, 255, 0).astype(np.uint8)
        part_silhouette = cv2.morphologyEx(
            part_silhouette, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2
        )

    inverted = cv2.bitwise_not(binary_mask)
    inside_dark = cv2.bitwise_and(inverted, part_silhouette)

    num_labels, labels, cv_stats, centroids = cv2.connectedComponentsWithStats(
        inside_dark, connectivity=8
    )

    candidate_blobs = []
    for i in range(1, num_labels):
        area = cv_stats[i, cv2.CC_STAT_AREA]
        x, y, w, h, _ = cv_stats[i]
        touches_border = (
            x <= 1 or y <= 1
            or x + w >= part_silhouette.shape[1] - 1
            or y + h >= part_silhouette.shape[0] - 1
        )
        if min_area <= area <= max_area and not touches_border:
            candidate_blobs.append({
                "area": int(area),
                "centroid": tuple(centroids[i].round(1)),
                "bbox": tuple(int(v) for v in cv_stats[i, :4]),
            })

    total_void_area = sum(b["area"] for b in candidate_blobs)

    stats = {
        "num_candidate_voids": len(candidate_blobs),
        "total_void_area_px": total_void_area,
        "largest_void_area_px": max((b["area"] for b in candidate_blobs), default=0),
        "mean_void_area_px": (total_void_area / len(candidate_blobs)) if candidate_blobs else 0.0,
    }

    # Build a visualization
    viz = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    for b in candidate_blobs:
        x, y, w, h = b["bbox"]
        cv2.rectangle(viz, (x, y), (x + w, y + h), (0, 0, 255), 1)

    return stats, viz


def run_pipeline(path: str):
    """
    Run all 4 steps in sequence on a single image and return every
    intermediate result, useful for visualization / QA.
    """
    gray = load_grayscale(path)
    enhanced = apply_clahe(gray)
    binary_mask, thresh_raw = threshold_and_clean(enhanced)
    edges = canny_edges(enhanced)
    blob_stats, blob_viz = blob_analysis(binary_mask)

    return {
        "original": gray,
        "clahe": enhanced,
        "thresh_raw": thresh_raw,
        "binary_clean": binary_mask,
        "edges": edges,
        "blob_stats": blob_stats,
        "blob_viz": blob_viz,
    }


if __name__ == "__main__":
    import sys
    result = run_pipeline(sys.argv[1])
    print(result["blob_stats"])
