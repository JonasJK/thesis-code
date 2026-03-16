"""
Utility functions for splitting data into training and test sets.
"""

import logging
import random
import time
from pathlib import Path

log = logging.getLogger(__name__)


def split_file_list(
    input_file: str | Path,
    output_train: str | Path,
    output_test: str | Path,
    train_ratio: float = 0.8,
    random_seed: int | None = None,
) -> tuple[int, int]:
    """
    Split a file list into training and test sets.

    Parameters:
    -----------
    input_file : str or Path
        Path to the input file list (.txt)
    output_train : str or Path
        Path to write the training file list
    output_test : str or Path
        Path to write the test file list
    train_ratio : float
        Fraction of files to use for training (default: 0.8)
    random_seed : int or None
        Random seed for reproducible splits. If None, uses time-based random seed.

    Returns:
    --------
    Tuple[int, int]
        Number of files in training and test sets
    """
    input_file = Path(input_file)
    output_train = Path(output_train)
    output_test = Path(output_test)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Read all file paths
    file_paths = []
    with open(input_file) as f:
        for line in f:
            path = line.strip()
            if path:
                file_paths.append(path)

    log.info(f"Read {len(file_paths)} file paths from {input_file}")

    if random_seed is None:
        random_seed = int(time.time())
        log.info(f"Using time-based random seed: {random_seed}")
    else:
        log.info(f"Using provided random seed: {random_seed}")

    # Shuffle and split
    random.seed(random_seed)
    random.shuffle(file_paths)

    split_point = int(len(file_paths) * train_ratio)
    train_files = file_paths[:split_point]
    test_files = file_paths[split_point:]

    # Write training set
    with open(output_train, "w") as f:
        for file_path in train_files:
            f.write(f"{file_path}\n")

    # Write test set
    with open(output_test, "w") as f:
        for file_path in test_files:
            f.write(f"{file_path}\n")

    log.info("Split complete:")
    log.info(f"  Training set: {len(train_files)} files -> {output_train}")
    log.info(f"  Test set: {len(test_files)} files -> {output_test}")
    log.info(f"  Split ratio: {train_ratio:.2%} training, {1 - train_ratio:.2%} test")
    return len(train_files), len(test_files)


def create_stratified_split_by_directory(
    input_file: str | Path,
    output_train: str | Path,
    output_test: str | Path,
    train_ratio: float = 0.8,
    random_seed: int | None = None,
) -> tuple[int, int]:
    """
    Create a stratified split that ensures files from the same directory
    are represented in both training and test sets.

    This is useful when your files are organized by location/date/etc.
    """
    input_file = Path(input_file)

    # Set random seed
    if random_seed is None:
        random_seed = int(time.time())
        log.info(f"Using time-based random seed: {random_seed}")
    else:
        log.info(f"Using provided random seed: {random_seed}")

    # Read and group files by directory
    directory_groups = {}
    with open(input_file) as f:
        for line in f:
            path = line.strip()
            if path:
                dir_name = str(Path(path).parent)
                if dir_name not in directory_groups:
                    directory_groups[dir_name] = []
                directory_groups[dir_name].append(path)

    log.info(f"Found {len(directory_groups)} unique directories")
    for dir_name, files in directory_groups.items():
        log.info(f"  {dir_name}: {len(files)} files")

    # Split each directory separately
    random.seed(random_seed)
    train_files = []
    test_files = []

    for dir_name, files in directory_groups.items():
        random.shuffle(files)
        split_point = max(1, int(len(files) * train_ratio))  # Ensure at least 1 file in training
        train_files.extend(files[:split_point])
        test_files.extend(files[split_point:])

    # Shuffle the final lists
    random.shuffle(train_files)
    random.shuffle(test_files)

    # Write files
    with open(output_train, "w") as f:
        for file_path in train_files:
            f.write(f"{file_path}\n")

    with open(output_test, "w") as f:
        for file_path in test_files:
            f.write(f"{file_path}\n")

    log.info("Stratified split complete:")
    log.info(f"  Training set: {len(train_files)} files -> {output_train}")
    log.info(f"  Test set: {len(test_files)} files -> {output_test}")
    log.info(f"  Random seed used: {random_seed}")

    return len(train_files), len(test_files)


def main():
    """Command line interface for data splitting."""
    import argparse

    parser = argparse.ArgumentParser(description="Split file list into training and test sets")
    parser.add_argument("input_file", help="Input file list (.txt)")
    parser.add_argument("--output-train", required=True, help="Output training file list")
    parser.add_argument("--output-test", required=True, help="Output test file list")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)",
    )
    parser.add_argument("--stratified", action="store_true", help="Use stratified split by directory")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible splits (default: time-based)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        if args.stratified:
            n_train, n_test = create_stratified_split_by_directory(
                args.input_file,
                args.output_train,
                args.output_test,
                args.train_ratio,
                args.seed,
            )
        else:
            n_train, n_test = split_file_list(
                args.input_file,
                args.output_train,
                args.output_test,
                args.train_ratio,
                args.seed,
            )

        print(f"Split {n_train + n_test} files:")
        print(f"  Training: {n_train} files ({n_train / (n_train + n_test):.1%})")
        print(f"  Test: {n_test} files ({n_test / (n_train + n_test):.1%})")

    except Exception as e:
        log.error(f"Error during splitting: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
