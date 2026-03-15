"""
Memory monitoring utilities.
"""

import logging
import os

import psutil
import torch

log = logging.getLogger(__name__)

def get_memory_usage_mb():
    """Get current memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def print_memory_info(stage):
    """Log current memory usage with a stage label."""
    # Skip this log block unless the logger is in DEBUG mode.
    if not log.isEnabledFor(logging.DEBUG):
        return
    try:
        mem_mb = get_memory_usage_mb()
        log.debug(f"Memory usage ({stage}): {mem_mb:.1f} MB")
    except Exception:
        log.debug(f"Memory usage ({stage}): unknown")

def log_cuda_info():
    """Log information about CUDA devices being used."""

    # Skip this log block unless the logger is in DEBUG mode.
    if not log.isEnabledFor(logging.DEBUG):
        return
    try:
        log.debug("=== CUDA Device Information ===")
        log.debug("XGBoost device setting: cuda")

        # Query CUDA device details through PyTorch.
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)

            log.debug("PyTorch CUDA available: Yes")
            log.debug(f"CUDA device count: {device_count}")
            log.debug(f"Current CUDA device: {current_device}")
            log.debug(f"Current device name: {device_name}")

            # Log memory stats for the active CUDA device.
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
            log.debug(f"GPU memory allocated: {memory_allocated:.2f} GB")
            log.debug(f"GPU memory reserved: {memory_reserved:.2f} GB")

            # Log all visible CUDA devices.
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                log.debug(f"GPU {i}: {device_name}")
        else:
            log.warning("PyTorch CUDA is not available")

        # Log CUDA-related environment variables for debugging.
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is not None:
            log.debug(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
        else:
            log.debug("CUDA_VISIBLE_DEVICES: Not set (all devices visible)")

        # If pynvml is available, collect additional GPU details.
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            log.debug(f"NVIDIA-ML device count: {device_count}")

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                log.debug(f"GPU {i} detailed info: {name}")
                log.debug(f"  Memory Total: {memory_info.total / 1024**3:.2f} GB")
                log.debug(f"  Memory Used: {memory_info.used / 1024**3:.2f} GB")
                log.debug(f"  Memory Free: {memory_info.free / 1024**3:.2f} GB")
        except ImportError:
            log.debug("pynvml not available for detailed GPU info")
        except Exception as e:
            log.warning(f"Could not get NVIDIA-ML info: {e}")

        log.debug("XGBoost will attempt to use CUDA device for training")
        log.debug("=== End CUDA Device Information ===")

    except Exception as e:
        log.error(f"Error logging CUDA information: {e}")
