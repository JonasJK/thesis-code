import logging
import os
import random
import shutil
import tempfile
import zipfile
from collections.abc import Generator
from pathlib import Path

log = logging.getLogger(__name__)

class DataLoader:
    """
    DataLoader class that can handle two types of data sources:
    1. A text file containing file paths to TIFF files
    2. A directory containing ZIP files, each containing one TIFF file
    """

    def __init__(
        self,
        source: str | Path,
        cleanup_temp=True,
        shuffle_zip_files=True,
        random_seed=None,
    ):
        """
        Initialize DataLoader with either a file list or directory of ZIP files.

        Parameters:
        -----------
        source : str or Path
            Either:
            - Path to a text file containing file paths (one per line)
            - Path to a directory containing ZIP files
        cleanup_temp : bool
            Whether to automatically clean up temporary files when iterating through ZIP files
        shuffle_zip_files : bool
            Whether to shuffle ZIP files for random sampling (only applies to ZIP directories)
        random_seed : int, optional
            Random seed for shuffling ZIP files (for reproducible sampling)
        """
        self.source = Path(source)
        self.cleanup_temp = cleanup_temp
        self.shuffle_zip_files = shuffle_zip_files
        self.random_seed = random_seed
        self.temp_dirs = []
        # Keep a cache for one extracted file to reduce repeated unzip work.
        self._current_extracted_zip = None
        self._current_extracted_path = None
        self._current_temp_dir = None

        if not self.source.exists():
            raise FileNotFoundError(f"Source path does not exist: {self.source}")

        # Detect whether the source is a file list or a ZIP directory.
        if self.source.is_file() and self.source.suffix == ".txt":
            self.source_type = "filelist"
            self.file_paths = self._load_file_list()
            log.info(
                f"DataLoader initialized with file list: {len(self.file_paths)} files"
            )
        elif self.source.is_dir():
            self.source_type = "zipdir"
            self.zip_files = self._find_zip_files()
            log.info(
                f"DataLoader initialized with ZIP directory: {len(self.zip_files)} ZIP files"
            )
        else:
            raise ValueError(
                f"Source must be either a .txt file or a directory, got: {self.source}"
            )

    def _load_file_list(self) -> list[Path]:
        """Load file paths from text file."""
        file_paths = []
        with open(self.source) as f:
            for line in f:
                path = line.strip()
                if path:
                    file_path = Path(path)
                    if file_path.exists():
                        file_paths.append(file_path)
                    else:
                        log.warning(f"File not found, skipping: {file_path}")
        random.shuffle(file_paths)
        return file_paths

    def _find_zip_files(self) -> list[Path]:
        """Find all ZIP files in the directory."""
        zip_files = list(self.source.glob("*.zip"))

        random.shuffle(zip_files)
        return zip_files

    def _extract_tif_from_zip(self, zip_path: Path) -> Path:
        """
        Extract the TIFF file from a ZIP archive to a temporary directory.
        Only keeps one extracted file at a time to minimize disk usage.

        Parameters:
        -----------
        zip_path : Path
            Path to the ZIP file

        Returns:
        --------
        Path
            Path to the extracted TIFF file
        """
        # Reuse the extracted file when it is already cached.
        if (
            self._current_extracted_zip == zip_path
            and self._current_extracted_path
            and self._current_extracted_path.exists()
        ):
            return self._current_extracted_path

        # Remove the previous extraction before processing the next file.
        self._cleanup_current_extraction()

        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp(prefix=f"dataloader_{zip_path.stem}_")

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                tif_files = [
                    name
                    for name in zip_ref.namelist()
                    if name.lower().endswith((".tif", ".tiff"))
                ]

                if not tif_files:
                    raise ValueError(f"No TIFF files found in {zip_path}")

                if len(tif_files) > 1:
                    log.warning(
                        f"Multiple TIFF files found in {zip_path}, using first one: {tif_files[0]}"
                    )

                tif_name = tif_files[0]
                zip_ref.extract(tif_name, temp_dir)

                extracted_path = Path(temp_dir) / tif_name
                log.debug(f"Extracted {tif_name} from {zip_path} to {extracted_path}")

                # Update the extraction cache.
                self._current_extracted_zip = zip_path
                self._current_extracted_path = extracted_path
                self._current_temp_dir = temp_dir

                return extracted_path

        except Exception as e:
            # Clean up temporary files if extraction fails.
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise RuntimeError(f"Failed to extract TIFF from {zip_path}: {e}")

    def _cleanup_current_extraction(self):
        """Clean up the currently extracted file and its temp directory."""
        if self._current_temp_dir and os.path.exists(self._current_temp_dir):
            try:
                shutil.rmtree(self._current_temp_dir)
                log.debug(f"Cleaned up current extraction: {self._current_temp_dir}")
            except Exception as e:
                log.warning(
                    f"Failed to clean up current extraction {self._current_temp_dir}: {e}"
                )

        self._current_extracted_zip = None
        self._current_extracted_path = None
        self._current_temp_dir = None

    def get_file_paths(self) -> list[Path]:
        """
        Get all file paths. For ZIP directories, this will extract files one by one.

        Warning: This method extracts ZIP files one at a time, keeping only one extracted at once.
        For large datasets, consider using iterate_files() instead.

        Returns:
        --------
        List[Path]
            List of paths to TIFF files
        """
        if self.source_type == "filelist":
            return self.file_paths.copy()
        elif self.source_type == "zipdir":
            extracted_paths = []
            for zip_path in self.zip_files:
                try:
                    extracted_path = self._extract_tif_from_zip(zip_path)
                    extracted_paths.append(extracted_path)
                except Exception as e:
                    log.error(f"Failed to extract {zip_path}: {e}")
                    continue
            return extracted_paths
        else:
            return []

    def iterate_files(self) -> Generator[tuple[Path, bool], None, None]:
        """
        Iterate through files, yielding (file_path, is_temporary) tuples.

        For file lists, is_temporary is always False.
        For ZIP directories, files are extracted on-demand and is_temporary is True.
        Only one file is kept extracted at a time.

        Yields:
        -------
        Tuple[Path, bool]
            (file_path, is_temporary) where is_temporary indicates if the file
            should be cleaned up after use
        """
        if self.source_type == "filelist":
            for file_path in self.file_paths:
                yield file_path, False

        elif self.source_type == "zipdir":
            for zip_path in self.zip_files:
                try:
                    extracted_path = self._extract_tif_from_zip(zip_path)
                    yield extracted_path, True

                    # Note: cleanup happens automatically when next file is extracted
                    # or when the DataLoader is destroyed/cleaned up

                except Exception as e:
                    log.error(f"Failed to process {zip_path}: {e}")
                    continue

    def cleanup(self):
        """Clean up all temporary directories created during operation."""
        # Remove the currently extracted temporary file.
        self._cleanup_current_extraction()

    def __len__(self) -> int:
        """Return the number of files available."""
        if self.source_type == "filelist":
            return len(self.file_paths)
        elif self.source_type == "zipdir":
            return len(self.zip_files)
        return 0

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up temporary files."""
        self.cleanup()

    def __repr__(self) -> str:
        """String representation of the DataLoader."""
        if self.source_type == "filelist":
            return f"DataLoader(filelist={self.source}, files={len(self.file_paths)})"
        elif self.source_type == "zipdir":
            return f"DataLoader(zipdir={self.source}, zips={len(self.zip_files)})"
        return f"DataLoader(source={self.source})"

def example_usage():
    """Example of how to use the DataLoader class."""

    print("Example 1: File list")
    try:
        with DataLoader("/path/to/file_list.txt") as loader:
            print(f"Loaded {len(loader)} files")

            for file_path, is_temp in loader.iterate_files():
                print(f"Processing: {file_path} (temporary: {is_temp})")
                break

    except FileNotFoundError:
        print("File list not found (this is expected in example)")

    print("\nExample 2: ZIP directory")
    try:
        with DataLoader("/path/to/zip/directory") as loader:
            print(f"Loaded {len(loader)} ZIP files")

            for file_path, is_temp in loader.iterate_files():
                print(f"Processing: {file_path} (temporary: {is_temp})")
                break

    except FileNotFoundError:
        print("ZIP directory not found (this is expected in example)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    example_usage()
