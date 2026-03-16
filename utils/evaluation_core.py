"""
Core evaluation functionality for NIR prediction models.
"""

import logging
import time
from pathlib import Path

import cupy as cp
import numpy as np
import rasterio
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

log = logging.getLogger(__name__)


def clear_gpu_memory():
    """Clear GPU memory cache to prevent OOM errors during long evaluation runs."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def evaluate_files(model, file_paths_or_loader, downscale_to=None, sample_limit=None, max_files=None):
    """Evaluate model predictions against ground truth.

    Can accept either:
    1. A list of file paths (traditional approach)
    2. A DataLoader instance (for 1-by-1 processing of ZIP files)

    Parameters:
    -----------
    model : object
        Model with predict_image method and timing attribute
    file_paths_or_loader : list or DataLoader
        Either a list of file paths or a DataLoader instance
    downscale_to : int, optional
        Downscale images to this size for SSIM calculation
    sample_limit : int, optional
        Limit the number of pixels used for evaluation per file
    max_files : int, optional
        Maximum number of files to evaluate

    Returns a dict with overall RMSE, R2, MAE and average SSIM.
    """
    if not hasattr(model, "model") or not model.model:
        raise ValueError("Model is not available for evaluation")

    per_file_ssim = []

    # Instead of storing all pixels, accumulate statistics incrementally
    sum_squared_error = 0.0
    sum_absolute_error = 0.0
    sum_actual = 0.0
    sum_actual_squared = 0.0
    sum_pred = 0.0
    sum_cross_product = 0.0
    total_pixels = 0

    start = time.time()

    if hasattr(file_paths_or_loader, "iterate_files"):
        log.info("Evaluating using DataLoader (1-by-1 processing)")
        file_iterator = file_paths_or_loader.iterate_files()
        total_files = len(file_paths_or_loader)
    else:
        log.info("Evaluating using file path list")
        file_iterator = [(Path(fp), False) for fp in file_paths_or_loader]
        total_files = len(file_paths_or_loader)

    files_to_evaluate = min(max_files, total_files) if max_files else total_files
    log.info(f"Will evaluate: {files_to_evaluate} files out of {total_files} available")

    processed_count = 0
    evaluated_count = 0
    for file_path, is_temp in file_iterator:
        if max_files and evaluated_count >= max_files:
            log.info(f"Reached max_files limit ({max_files}), stopping evaluation")
            break

        processed_count += 1

        # Check if this file was used in training
        normalized_path = str(Path(file_path).resolve()) if not is_temp else str(file_path)
        if hasattr(model, "training_files") and normalized_path in model.training_files:
            log.info(f"Skipping file {processed_count}/{total_files}: {file_path} (used in training)")
            continue

        evaluated_count += 1

        try:
            log.info(f"Evaluating file {evaluated_count}/{total_files}: {file_path} (temp: {is_temp})")

            with rasterio.open(file_path) as src:
                if src.count < 4:
                    log.info(f"Skipping {file_path}: needs 4 bands (RGBI), found {src.count}")
                    continue

                data = src.read()
                rgb = data[:3].transpose(1, 2, 0).astype(np.float32)
                nir_actual = data[3].astype(np.float32)

                pred_nir = model.predict_image(rgb)

                flat_rgb = rgb.reshape(-1, 3)
                flat_actual = nir_actual.flatten()
                flat_pred = pred_nir.flatten()

                valid_mask = (
                    (flat_rgb[:, 0] + flat_rgb[:, 1] + flat_rgb[:, 2] > 0)
                    & (flat_actual > 0)
                    & (flat_rgb.max(axis=1) < 65535)
                    & (flat_actual < 65535)
                )

                if not np.any(valid_mask):
                    log.info(f"Skipping {file_path}: no valid pixels found for evaluation")
                    continue

                actual_vals = flat_actual[valid_mask]
                pred_vals = flat_pred[valid_mask]

                if sample_limit is not None and len(actual_vals) > sample_limit:
                    np.random.seed(42)
                    idx = np.random.choice(len(actual_vals), size=sample_limit, replace=False)
                    actual_vals = actual_vals[idx]
                    pred_vals = pred_vals[idx]

                n = len(actual_vals)
                sum_squared_error += np.sum((actual_vals - pred_vals) ** 2)
                sum_absolute_error += np.sum(np.abs(actual_vals - pred_vals))
                sum_actual += np.sum(actual_vals)
                sum_actual_squared += np.sum(actual_vals**2)
                sum_pred += np.sum(pred_vals)
                sum_cross_product += np.sum(actual_vals * pred_vals)
                total_pixels += n

                del actual_vals, pred_vals, flat_rgb, flat_actual, flat_pred, valid_mask

                try:
                    if downscale_to is not None:
                        a_resized = resize(nir_actual, (downscale_to, downscale_to), anti_aliasing=True)
                        p_resized = resize(pred_nir, (downscale_to, downscale_to), anti_aliasing=True)
                    else:
                        a_resized = nir_actual
                        p_resized = pred_nir

                    data_range = float(p_resized.max() - p_resized.min())
                    file_ssim = 0.0 if data_range == 0 else ssim(a_resized, p_resized, data_range=data_range)
                except Exception as e:
                    log.info(f"SSIM calculation failed for {file_path}: {e}")
                    file_ssim = 0.0

                per_file_ssim.append(file_ssim)
                log.info(f"Metrics for {Path(file_path).name}: SSIM={file_ssim:.4f}, pixels={n}")

                del rgb, nir_actual, pred_nir, data
                if downscale_to is not None:
                    del a_resized, p_resized

                clear_gpu_memory()

        except Exception as e:
            log.info(f"Evaluation error for {file_path}: {e}")
            clear_gpu_memory()  # Also clear on error
            continue

    log.info(
        f"Evaluation summary: processed {processed_count} files, evaluated {evaluated_count} files (skipped {processed_count - evaluated_count} training files)"
    )

    if total_pixels == 0:
        log.info("No evaluation data collected.")
        return None

    rmse = float(np.sqrt(sum_squared_error / total_pixels))
    mae = float(sum_absolute_error / total_pixels)

    sum_actual / total_pixels
    ss_tot = sum_actual_squared - (sum_actual**2) / total_pixels
    ss_res = sum_squared_error

    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    avg_ssim = float(np.mean(per_file_ssim)) if per_file_ssim else 0.0

    results = {
        "RMSE": rmse,
        "R2": r2,
        "MAE": mae,
        "SSIM": avg_ssim,
        "n_pixels": int(total_pixels),
        "n_files": len(per_file_ssim),
    }
    duration = time.time() - start
    model.timing["evaluate"] = duration

    log.info(f"Evaluation summary across {results['n_files']} files and {results['n_pixels']:,} pixels:")
    log.info(f"RMSE={rmse:.4f}, R2={r2:.4f}, MAE={mae:.4f}, SSIM(avg)={avg_ssim:.4f}")
    log.info(f"Evaluation total time: {duration:.2f}s (predict cumulative: {model.timing['predict']:.2f}s)")
    return results
