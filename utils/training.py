"""
Training utilities for NIR prediction models.
"""

import logging
import time
from pathlib import Path

from .data_processing import load_rgbi_image, reservoir_sampling_update
from .memory_utils import print_memory_info

log = logging.getLogger(__name__)


def train_from_data_loader(model_instance, data_loader, max_files=None):
    """
    Train using DataLoader for true 1-by-1 processing.
    This is memory and disk-space efficient for ZIP archives.

    Parameters:
    -----------
    model_instance : object
        Model instance with training arrays and reservoir sampling attributes
    data_loader : DataLoader
        DataLoader instance (can be filelist or zipdir)
    max_files : int, optional
        Maximum number of files to process (for testing/limiting)
    """
    start = time.time()
    total_files = len(data_loader)
    files_to_process = min(max_files, total_files) if max_files else total_files

    log.info("Starting training data collection using DataLoader")
    log.info(f"Source type: {data_loader.source_type}, Total files: {total_files}")
    log.info(f"Will process: {files_to_process} files")

    processed_count = 0
    for file_path, is_temp in data_loader.iterate_files():
        if max_files and processed_count >= max_files:
            log.info(f"Reached max_files limit ({max_files}), stopping")
            break

        processed_count += 1
        normalized_path = str(Path(file_path).resolve()) if not is_temp else str(file_path)
        model_instance.training_files.add(normalized_path)

        log.info(f"Processing file {processed_count}/{files_to_process}: {file_path} (temp: {is_temp})")

        try:
            # Use the model's method if it has one, otherwise use the shared function
            if hasattr(model_instance, "load_rgbi_image"):
                image_loader = model_instance.load_rgbi_image
            else:
                # Create extract_features function that uses model's EPS
                extract_features_func = lambda rgb: model_instance._extract_features(rgb)
                image_loader = lambda fp: load_rgbi_image(fp, extract_features_func=extract_features_func)

            for features, nir in image_loader(file_path):
                # Update model's reservoir sampling
                if hasattr(model_instance, "_reservoir_sampling_update"):
                    model_instance._reservoir_sampling_update(features, nir)
                else:
                    (
                        model_instance.sample_count,
                        model_instance.total_pixels_seen,
                        added,
                        replaced,
                    ) = reservoir_sampling_update(
                        features,
                        nir,
                        model_instance.training_rgb,
                        model_instance.training_nir,
                        model_instance.sample_count,
                        model_instance.total_pixels_seen,
                        model_instance.max_samples,
                    )

            if processed_count % 10 == 0:
                log.info(
                    f"Progress: {processed_count}/{files_to_process} files, "
                    f"samples collected: {model_instance.sample_count}, "
                    f"total pixels seen: {model_instance.total_pixels_seen}"
                )
                print_memory_info(f"file_{processed_count}")

        except Exception as e:
            log.error(f"Error processing file {file_path}: {e}")
            continue

        # Note: cleanup happens automatically in DataLoader.iterate_files() for temp files

    duration = time.time() - start
    model_instance.timing["processing"] = duration
    log.info(f"Finished sampling from DataLoader. Files processed: {processed_count}")
    log.info(f"Total samples stored: {model_instance.sample_count} (processing time: {duration:.2f}s)")
    log.info(f"Training files tracked: {len(model_instance.training_files)}")
    print_memory_info("train_from_data_loader_end")
