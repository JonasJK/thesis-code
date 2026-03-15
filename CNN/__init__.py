"""
CNN-based NIR prediction from RGB images using patch-based approach.
"""

from .cnn_nir import CNNNir, NIRPredictionCNN

__all__ = ["CNNNir", "NIRPredictionCNN"]
