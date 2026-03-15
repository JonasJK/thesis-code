"""
Feature extraction utilities for NIR prediction models.
"""

import logging

import cupy as cp
import numpy as np

log = logging.getLogger(__name__)

def extract_features(rgb, eps=None):
    """Extract enhanced features from RGB data for improved NIR prediction.
    Optimized with vectorized operations for better performance.

    Parameters:
    -----------
    rgb : np.ndarray
        RGB data of shape (n_pixels, 3)
    eps : float, optional
        Small epsilon value to avoid division by zero

    Returns:
    --------
    np.ndarray
        Feature array of shape (n_pixels, 8)
    """
    if eps is None:
        eps = np.finfo(np.float32).eps
    # Move arrays to GPU when available.
    rgb_gpu = cp.asarray(rgb, dtype=cp.float32)
    r, g, b = rgb_gpu[:, 0], rgb_gpu[:, 1], rgb_gpu[:, 2]

    features_gpu = cp.column_stack(
        [
            r,
            g,
            2 * g - r - b,
            1.4 * r - g,
            (g - r) / (g + r + eps),
            (g * g - r * b) / (g * g + r * b + eps),
            g - 0.39 * r - 0.61 * b,
            (r + g + b) / 3.0,
        ]
    )

    return cp.asnumpy(features_gpu)

def get_feature_names():
    """Get the names of all extracted features.

    Returns:
    --------
    list
        List of feature names in the same order as extract_features output
    """
    return ["R", "G", "ExG", "ExR", "NGRDI", "RGBVI", "TGI", "Brightness"]

def analyze_feature_statistics(features, feature_names=None):
    """Analyze statistical properties of extracted features.

    Parameters:
    -----------
    features : np.ndarray
        Feature array from extract_features
    feature_names : list, optional
        Names of features. If None, uses get_feature_names()

    Returns:
    --------
    dict
        Dictionary containing statistical analysis of features
    """
    if feature_names is None:
        feature_names = get_feature_names()

    stats = {}
    for i, name in enumerate(feature_names):
        if i < features.shape[1]:
            feature_data = features[:, i]
            stats[name] = {
                "mean": np.mean(feature_data),
                "std": np.std(feature_data),
                "min": np.min(feature_data),
                "max": np.max(feature_data),
                "median": np.median(feature_data),
                "q25": np.percentile(feature_data, 25),
                "q75": np.percentile(feature_data, 75),
            }

    return stats
