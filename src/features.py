"""
features.py

Feature extraction for the classical baseline classifier.
Combines:
  - LBP (Local Binary Pattern) histogram: captures local texture/roughness
  - GLCM (Gray-Level Co-occurrence Matrix) properties: captures texture
    coarseness, contrast, and regularity across the surface
  - Canny edge density: captures rim jaggedness (flash/burr indicator)
  - Blob/void stats from preprocessing.py: captures pit/void-like regions

Together these approximate the "Approach A" feature set described in the
pitch: texture descriptors (LBP, GLCM) + contour/blob features.
"""

import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from preprocessing import apply_clahe, threshold_and_clean, canny_edges, blob_analysis


LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"
LBP_NBINS = LBP_POINTS + 2  # uniform LBP has P+2 distinct bins


def lbp_histogram(gray: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, LBP_NBINS + 1), range=(0, LBP_NBINS)
    )
    hist = hist.astype(np.float64)
    hist /= (hist.sum() + 1e-7)  # normalize to a distribution
    return hist


def glcm_features(gray: np.ndarray) -> np.ndarray:
    # Downsample gray levels to 32 for a tractable co-occurrence matrix
    img_q = (gray / 8).astype(np.uint8)  # 256 -> 32 levels
    glcm = graycomatrix(
        img_q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32, symmetric=True, normed=True
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    feats = [graycoprops(glcm, p).mean() for p in props]
    return np.array(feats, dtype=np.float64)


def edge_density(edges: np.ndarray) -> float:
    return float(np.count_nonzero(edges)) / edges.size


def extract_features(gray: np.ndarray) -> np.ndarray:
    """
    Full feature vector for one grayscale image, reusing the same
    preprocessing pipeline steps defined in preprocessing.py so the
    features are consistent with what's shown on the pipeline slide.
    """
    enhanced = apply_clahe(gray)
    binary_mask, _ = threshold_and_clean(enhanced)
    edges = canny_edges(enhanced)
    blob_stats, _ = blob_analysis(binary_mask)

    lbp_hist = lbp_histogram(enhanced)          # LBP_NBINS features
    glcm_feats = glcm_features(enhanced)         # 6 features
    edge_feat = np.array([edge_density(edges)])  # 1 feature
    blob_feats = np.array([
        blob_stats["num_candidate_voids"],
        blob_stats["total_void_area_px"],
        blob_stats["largest_void_area_px"],
        blob_stats["mean_void_area_px"],
    ], dtype=np.float64)                          # 4 features

    return np.concatenate([lbp_hist, glcm_feats, edge_feat, blob_feats])


FEATURE_NAMES = (
    [f"lbp_bin_{i}" for i in range(LBP_NBINS)]
    + ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
       "glcm_energy", "glcm_correlation", "glcm_ASM"]
    + ["edge_density"]
    + ["num_voids", "total_void_area", "largest_void_area", "mean_void_area"]
)
