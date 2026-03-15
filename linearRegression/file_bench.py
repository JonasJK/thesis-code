import random
import time
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

def benchmark_file_formats(file_list_path, n_files=50):
    """
    Simple benchmark: test 50 random files and compare TIF vs JP2 loading times.
    """
    # Read all files
    with open(file_list_path) as f:
        all_files = [line.strip() for line in f if line.strip()]

    # Randomly sample files
    random.seed(42)  # For reproducible results
    test_files = random.sample(all_files, min(n_files, len(all_files)))

    print(f"Testing {len(test_files)} random files from {len(all_files)} total files")

    tif_times = []
    jp2_times = []
    tif_sizes = []
    jp2_sizes = []

    for filepath in tqdm(test_files, desc="Benchmarking"):
        file_ext = Path(filepath).suffix.lower()
        file_size_mb = Path(filepath).stat().st_size / 1e6

        # Time the loading (same method as your main code)
        start_time = time.time()

        try:
            with rasterio.open(filepath) as src:
                height, width = src.height, src.width
                total_pixels = height * width
                sample_ratio = 0.01  # Same as your code
                n_samples = int(total_pixels * sample_ratio)

                if n_samples < 1000:
                    # Small image - read all
                    src.read()
                else:
                    # Large image - sample (same as your code)
                    sample_indices = np.random.choice(
                        total_pixels, n_samples, replace=False
                    )
                    row_indices = sample_indices // width
                    col_indices = sample_indices % width

                    # Read sampled pixels
                    for band in [1, 2, 3, 4]:  # RGBI
                        band_data = src.read(band)
                        band_data[row_indices, col_indices]

            elapsed = time.time() - start_time

            # Categorize by format
            if file_ext in [".tif", ".tiff"]:
                tif_times.append(elapsed)
                tif_sizes.append(file_size_mb)
                if elapsed > 20:
                    print(f"Slow TIF: {elapsed:.1f}s - {Path(filepath).name}")
            elif file_ext in [".jp2", ".j2k"]:
                jp2_times.append(elapsed)
                jp2_sizes.append(file_size_mb)
                if elapsed > 20:
                    print(f"Slow JP2: {elapsed:.1f}s - {Path(filepath).name}")

        except Exception as e:
            print(f"Error with {Path(filepath).name}: {e}")

    # Print results
    print(f"\n{'='*50}")
    print("BENCHMARK RESULTS")
    print(f"{'='*50}")

    if tif_times:
        print(f"TIF FILES ({len(tif_times)} tested):")
        print(f"  Avg time: {np.mean(tif_times):.2f}s")
        print(f"  Avg size: {np.mean(tif_sizes):.1f}MB")
        print(f"  Time/MB: {np.mean(tif_times)/np.mean(tif_sizes):.3f}s/MB")
        print(f"  Range: {np.min(tif_times):.1f}s - {np.max(tif_times):.1f}s")

    if jp2_times:
        print(f"\nJP2 FILES ({len(jp2_times)} tested):")
        print(f"  Avg time: {np.mean(jp2_times):.2f}s")
        print(f"  Avg size: {np.mean(jp2_sizes):.1f}MB")
        print(f"  Time/MB: {np.mean(jp2_times)/np.mean(jp2_sizes):.3f}s/MB")
        print(f"  Range: {np.min(jp2_times):.1f}s - {np.max(jp2_times):.1f}s")

    # Compare if we have both formats
    if tif_times and jp2_times:
        tif_avg = np.mean(tif_times)
        jp2_avg = np.mean(jp2_times)
        slowdown = jp2_avg / tif_avg

        print("\nCOMPARISON:")
        print(
            f"JP2 is {slowdown:.1f}x {'SLOWER' if slowdown > 1 else 'FASTER'} than TIF"
        )

        if slowdown > 3:
            print("DIAGNOSIS: JP2 files are likely causing your 10s->60s slowdown")

    return {
        "tif_times": tif_times,
        "jp2_times": jp2_times,
        "tif_sizes": tif_sizes,
        "jp2_sizes": jp2_sizes,
    }

if __name__ == "__main__":
    results = benchmark_file_formats("/home/klugej/thesis/file_list.txt", n_files=50)
