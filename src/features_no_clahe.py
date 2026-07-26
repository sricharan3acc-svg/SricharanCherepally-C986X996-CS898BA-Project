"""
Ablation variant of features.py: skips CLAHE enhancement to isolate
its contribution to classifier performance.
"""
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from preprocessing import threshold_and_clean, canny_edges, blob_analysis

LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"
LBP_NBINS = LBP_POINTS + 2

def lbp_histogram(gray):
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_NBINS + 1), range=(0, LBP_NBINS))
    hist = hist.astype(np.float64)
    hist /= (hist.sum() + 1e-7)
    return hist

def glcm_features(gray):
    img_q = (gray / 8).astype(np.uint8)
    glcm = graycomatrix(img_q, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    return np.array([graycoprops(glcm, p).mean() for p in props], dtype=np.float64)

def edge_density(edges):
    return float(np.count_nonzero(edges)) / edges.size

def extract_features_no_clahe(gray):
    # Skip CLAHE -- run the rest of the pipeline directly on the raw grayscale image
    binary_mask, _ = threshold_and_clean(gray)
    edges = canny_edges(gray)
    blob_stats, _ = blob_analysis(binary_mask)

    lbp_hist = lbp_histogram(gray)
    glcm_feats = glcm_features(gray)
    edge_feat = np.array([edge_density(edges)])
    blob_feats = np.array([
        blob_stats["num_candidate_voids"], blob_stats["total_void_area_px"],
        blob_stats["largest_void_area_px"], blob_stats["mean_void_area_px"],
    ], dtype=np.float64)

    return np.concatenate([lbp_hist, glcm_feats, edge_feat, blob_feats])
