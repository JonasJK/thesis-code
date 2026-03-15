"""
DataLoader package for handling TIFF files from various sources.

This package provides the DataLoader class which can load TIFF files from:
1. Text files containing file paths
2. Directories containing ZIP files with TIFF files inside

Example usage:
    from DataLoader import DataLoader

    with DataLoader('file_list.txt') as loader:
        for file_path, is_temp in loader.iterate_files():
            pass

    with DataLoader('/path/to/zip/dir') as loader:
        for file_path, is_temp in loader.iterate_files():
            pass
"""

from .DataLoader import DataLoader
