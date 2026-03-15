"""
Data processing utilities for NIR prediction models.
"""

import logging

import dask
import numpy as np
import psutil
import rasterio
from rasterio.windows import Window

from .feature_extraction import extract_features
from .memory_utils import print_memory_info

log = logging.getLogger(__name__)

def load_rgbi_image(file_path, chunk_size=None, extract_features_func=None, eps=None):
    """Load and process an RGBI image in chunks with adaptive sizing.

    Parameters:
    -----------
    file_path : str
        Path to the RGBI image file
    chunk_size : int, optional
        Size of chunks to process. If None, will be calculated adaptively
    extract_features_func : callable, optional
        Function to extract features from RGB data
    eps : float, optional
        Small epsilon value for feature extraction

    Yields:
    -------
    tuple
        (features, nir) arrays for each valid chunk
    """
    log.info(f"Loading image: {file_path}")

    if extract_features_func is None:
        extract_features_func = lambda rgb: extract_features(rgb, eps)

    # Adaptive chunk size calculation
    if chunk_size is None:
        import os

        available_memory_gb = int(
            os.getenv("SLURM_MEM_PER_CPU", 0)
        ) / 1024 or psutil.virtual_memory().available / (1024**3)

        target_memory_mb = min(8192, max(1024, available_memory_gb * 1024))
        chunk_size = int(np.sqrt(target_memory_mb * 1024 * 1024 / 64))
        chunk_size = max(4096, min(32786, chunk_size))
        log.info(
            f"Adaptive chunk size: {chunk_size} (available memory: {available_memory_gb:.1f}GB)"
        )
    else:
        log.info(f"Using fixed chunk size: {chunk_size}")

    try:
        with rasterio.open(file_path) as src:
            for y in range(0, src.height, chunk_size):
                for x in range(0, src.width, chunk_size):
                    window = Window(
                        x,
                        y,
                        min(chunk_size, src.width - x),
                        min(chunk_size, src.height - y),
                    )
                    data = src.read(window=window)

                    def process_chunk(data):
                        rgb = np.moveaxis(data[:3], 0, -1).reshape(-1, 3)
                        nir = data[3].ravel()

                        valid_mask = (
                            (rgb[:, 0] + rgb[:, 1] + rgb[:, 2] > 0)
                            & (nir > 0)
                            & (rgb.max(axis=1) < 65535)
                            & (nir < 65535)
                        )

                        if np.any(valid_mask):
                            rgb_valid = rgb[valid_mask].astype(np.float32, copy=False)
                            features = extract_features_func(rgb_valid)
                            return features, nir[valid_mask].astype(
                                np.float32, copy=False
                            )
                        return None

                    result = dask.delayed(process_chunk)(data)
                    computed_result = dask.compute(result)[0]

                    if computed_result is not None:
                        yield computed_result

    except Exception as e:
        log.info(f"Error opening {file_path}: {e}")
        return

def reservoir_sampling_update(
    new_features,
    new_nir,
    training_rgb,
    training_nir,
    sample_count,
    total_pixels_seen,
    max_samples,
):
    """Update reservoir sampling arrays with new data using optimized vectorized operations.

    Parameters:
    -----------
    new_features : np.ndarray
        New feature data to add
    new_nir : np.ndarray
        New NIR data to add
    training_rgb : np.ndarray
        Training feature array to update
    training_nir : np.ndarray
        Training NIR array to update
    sample_count : int
        Current number of samples stored
    total_pixels_seen : int
        Total number of pixels seen so far
    max_samples : int
        Maximum number of samples to store

    Returns:
    --------
    tuple
        (new_sample_count, new_total_pixels_seen, added_count, replaced_count)
    """
    n_new = len(new_features)
    original_n_new = n_new
    space_left = max_samples - sample_count

    if space_left > 0:
        take_now = min(space_left, n_new)
        if take_now > 0:
            end_idx = sample_count + take_now
            training_rgb[sample_count:end_idx] = new_features[:take_now]
            training_nir[sample_count:end_idx] = new_nir[:take_now]
            sample_count += take_now

            if take_now < n_new:
                new_features = new_features[take_now:]
                new_nir = new_nir[take_now:]
                n_new -= take_now
            else:
                n_new = 0
    else:
        take_now = 0

    replaced = 0
    if n_new > 0:
        sample_indices = np.arange(n_new, dtype=np.int64) + 1
        accept_probs = max_samples / (total_pixels_seen + sample_indices)

        random_vals = np.random.random(n_new)
        keep_mask = random_vals < accept_probs
        n_keep = np.sum(keep_mask)

        if n_keep > 0:
            replace_indices = np.random.randint(
                0, max_samples, size=n_keep, dtype=np.int32
            )

            training_rgb[replace_indices] = new_features[keep_mask]
            training_nir[replace_indices] = new_nir[keep_mask]
            replaced = n_keep
    total_pixels_seen += original_n_new

    log.info(
        f"Reservoir update: added={take_now}, replaced={replaced}, sample_count={sample_count}, total_seen={total_pixels_seen}"
    )
    print_memory_info("reservoir_update")

    return sample_count, total_pixels_seen, take_now, replaced
