"""
Base class for NIR prediction models to share common functionality.
"""

import logging
import time

import numpy as np

from .data_processing import load_rgbi_image, reservoir_sampling_update
from .evaluation_core import evaluate_files
from .feature_extraction import extract_features, get_feature_names
from .memory_utils import get_memory_usage_mb, print_memory_info
from .training import train_from_data_loader

log = logging.getLogger(__name__)


class BaseNirModel:
    """Base class for NIR prediction models."""

    def __init__(self, max_samples=600_000_000):
        self.max_samples = max_samples
        self.total_pixels_seen = 0
        self.EPS = np.finfo(np.float32).eps
        self.training_rgb = np.empty((max_samples, 8), dtype=np.float32)  # RG + 6 additional features
        self.training_nir = np.empty(max_samples, dtype=np.float32)
        self.sample_count = 0
        self.training_files = set()

        self.timing = {"processing": 0.0, "fit": 0.0, "predict": 0.0, "evaluate": 0.0}
        self.model = None
        self.feature_importances_ = None
        print_memory_info("init")

    def _extract_features(self, rgb):
        """Extract RGB + VARI + ExG features from RGB data."""
        return extract_features(rgb, self.EPS)

    def _get_memory_usage_mb(self):
        """Get current memory usage in MB."""
        return get_memory_usage_mb()

    def _print_memory_info(self, stage):
        """Log current memory usage with a stage label."""
        print_memory_info(stage)

    def _reservoir_sampling_update(self, new_features, new_nir):
        """Update reservoir sampling with new data."""
        self.sample_count, self.total_pixels_seen, added, replaced = reservoir_sampling_update(
            new_features,
            new_nir,
            self.training_rgb,
            self.training_nir,
            self.sample_count,
            self.total_pixels_seen,
            self.max_samples,
        )

    def load_rgbi_image(self, file_path, chunk_size=None):
        """Load and process an RGBI image in chunks with adaptive sizing."""
        extract_features_func = lambda rgb: self._extract_features(rgb)
        return load_rgbi_image(file_path, chunk_size, extract_features_func, self.EPS)

    def train_from_data_loader(self, data_loader, max_files=None):
        """Train using DataLoader for true 1-by-1 processing."""
        train_from_data_loader(self, data_loader, max_files)

    def evaluate_files(self, file_paths_or_loader, downscale_to=None, sample_limit=None, max_files=None):
        """Evaluate model predictions against ground truth."""
        return evaluate_files(self, file_paths_or_loader, downscale_to, sample_limit, max_files)

    def train_from_directory(self, directory, file_list=None):
        """Train from a directory of files."""
        import os
        import random

        start = time.time()
        if file_list is None:
            all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".tif")]
        else:
            all_files = file_list

        log.info(f"Starting training data collection from {len(all_files)} files")
        random.shuffle(all_files)
        for idx, file_path in enumerate(all_files):
            log.info(f"Processing file {idx + 1}/{len(all_files)}: {file_path}")
            for features, nir in self.load_rgbi_image(file_path):
                self._reservoir_sampling_update(features, nir)

        duration = time.time() - start
        self.timing["processing"] = duration
        log.info(f"Finished sampling. Total samples stored: {self.sample_count} (processing time: {duration:.2f}s)")
        self._print_memory_info("train_from_directory_end")

    def print_summary(self):
        """Print timing summary and feature importances if available."""
        log.info("=== Summary ===")
        log.info(f"Processing time (sampling): {self.timing.get('processing', 0.0):.2f}s")
        log.info(f"Fit time: {self.timing.get('fit', 0.0):.2f}s")
        log.info(f"Predict time (cumulative): {self.timing.get('predict', 0.0):.2f}s")
        log.info(f"Evaluate time: {self.timing.get('evaluate', 0.0):.2f}s")
        if self.feature_importances_ is not None:
            feature_names = get_feature_names()
            log.info("Feature importances:")
            for name, importance in zip(feature_names, self.feature_importances_, strict=False):
                log.info(f"  {name}: {importance:.4f}")
        else:
            log.info("Feature importances: not available (model not fitted)")

    def predict_image(self, rgb_image):
        """Predict NIR for an RGB image. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement predict_image method")

    def fit_model(self):
        """Fit the model. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement fit_model method")
