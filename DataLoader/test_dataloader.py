#!/usr/bin/env python3
"""
Test script for the DataLoader class.
"""

import os
import sys
import tempfile
import zipfile
from pathlib import Path

# Add the code directory to the path
sys.path.append("/home/klugej/Dokumente/thesis/code")

import logging

from DataLoader import DataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

def create_test_files():
    """Create test files for testing DataLoader."""

    # Create a temporary directory for test files
    test_dir = Path(tempfile.mkdtemp(prefix="dataloader_test_"))
    log.info(f"Creating test files in: {test_dir}")

    # Create some dummy TIFF files (just empty files for testing)
    tiff_files = []
    for i in range(3):
        tiff_file = test_dir / f"test_image_{i}.tif"
        tiff_file.write_bytes(
            b"dummy tiff content"
        )  # Not a real TIFF, just for testing
        tiff_files.append(tiff_file)

    file_list = test_dir / "file_list.txt"
    with open(file_list, "w") as f:
        for tiff_file in tiff_files:
            f.write(f"{tiff_file}\n")

    # Create ZIP files containing the TIFF files
    zip_dir = test_dir / "zip_files"
    zip_dir.mkdir()

    zip_files = []
    for i, tiff_file in enumerate(tiff_files):
        zip_file = zip_dir / f"archive_{i}.zip"
        with zipfile.ZipFile(zip_file, "w") as zf:
            zf.write(tiff_file, tiff_file.name)
        zip_files.append(zip_file)

    return test_dir, file_list, zip_dir, tiff_files, zip_files

def test_file_list_loader():
    """Test DataLoader with file list."""
    log.info("Testing DataLoader with file list...")

    test_dir, file_list, zip_dir, tiff_files, zip_files = create_test_files()

    try:
        with DataLoader(file_list) as loader:
            log.info(f"Loader: {loader}")
            assert len(loader) == 3, f"Expected 3 files, got {len(loader)}"
            assert loader.source_type == "filelist"

            files_seen = []
            for file_path, is_temp in loader.iterate_files():
                log.info(f"File: {file_path}, temporary: {is_temp}")
                assert not is_temp, "Files from file list should not be temporary"
                files_seen.append(file_path)

            assert (
                len(files_seen) == 3
            ), f"Expected to see 3 files, saw {len(files_seen)}"
            log.info("File list test passed!")

    finally:
        import shutil

        shutil.rmtree(test_dir)

def test_zip_directory_loader():
    """Test DataLoader with ZIP directory."""
    log.info("Testing DataLoader with ZIP directory...")

    test_dir, file_list, zip_dir, tiff_files, zip_files = create_test_files()

    try:
        with DataLoader(zip_dir) as loader:
            log.info(f"Loader: {loader}")
            assert len(loader) == 3, f"Expected 3 ZIP files, got {len(loader)}"
            assert loader.source_type == "zipdir"

            files_seen = []
            for file_path, is_temp in loader.iterate_files():
                log.info(f"File: {file_path}, temporary: {is_temp}")
                assert is_temp, "Files from ZIP should be temporary"
                assert file_path.exists(), f"Extracted file should exist: {file_path}"
                files_seen.append(file_path)

            assert (
                len(files_seen) == 3
            ), f"Expected to see 3 files, saw {len(files_seen)}"
            log.info("ZIP directory test passed!")

    finally:
        import shutil

        shutil.rmtree(test_dir)

def test_error_handling():
    """Test DataLoader error handling."""
    log.info("Testing DataLoader error handling...")

    try:
        DataLoader("/non/existent/path")
        raise AssertionError("Should have raised FileNotFoundError")
    except FileNotFoundError:
        log.info("Correctly raised FileNotFoundError for non-existent path")

    test_dir = Path(tempfile.mkdtemp(prefix="dataloader_error_test_"))
    invalid_file = test_dir / "invalid.csv"
    invalid_file.write_text("not a txt file")

    try:
        DataLoader(invalid_file)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        log.info("Correctly raised ValueError for invalid file type")
    finally:
        import shutil

        shutil.rmtree(test_dir)

def main():
    """Run all tests."""
    log.info("Starting DataLoader tests...")

    try:
        test_file_list_loader()
        test_zip_directory_loader()
        test_error_handling()
        log.info("Tests passed.")
    except Exception as e:
        log.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
