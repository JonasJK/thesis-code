#!/usr/bin/env python3
"""
Test script to verify that the modularization works correctly.
"""

import os
import sys

import numpy as np

# Add the code directory to path
code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

# Test imports
print("Testing imports...")
try:
    from utils import BaseNirModel, extract_features, get_memory_usage_mb

    print("Utils imports successful")
except ImportError as e:
    print(f"Utils import failed: {e}")
    sys.exit(1)

try:
    from randomForest.randomForest import RandomForestNir

    print("RandomForest import successful")
except ImportError as e:
    print(f"RandomForest import failed: {e}")
    sys.exit(1)

try:
    from XGBoost.xgboost_nir import XGBoostNir

    print("XGBoost import successful")
except ImportError as e:
    print(f"XGBoost import failed: {e}")
    sys.exit(1)

# Test basic functionality
print("\nTesting basic functionality...")

# Test feature extraction
print("Testing feature extraction...")
rgb_data = np.random.rand(100, 3) * 1000  # Simulate RGB data
features = extract_features(rgb_data)
assert features.shape == (100, 8), f"Expected shape (100, 8), got {features.shape}"
print("Feature extraction works correctly")

# Test memory monitoring
print("Testing memory monitoring...")
mem_mb = get_memory_usage_mb()
assert isinstance(mem_mb, float), "Memory usage should return a float"
assert mem_mb > 0, "Memory usage should be positive"
print(f"Memory monitoring works correctly: {mem_mb:.1f} MB")

# Test model initialization
print("Testing model initialization...")
try:
    rf_model = RandomForestNir(max_samples=1000, n_estimators=10)
    assert hasattr(
        rf_model, "_extract_features"
    ), "RandomForest should have _extract_features method"
    assert hasattr(rf_model, "timing"), "RandomForest should have timing attribute"
    print("RandomForest model initialization successful")
except Exception as e:
    print(f"RandomForest model initialization failed: {e}")

try:
    xgb_model = XGBoostNir(max_samples=1000, n_estimators=10)
    assert hasattr(
        xgb_model, "_extract_features"
    ), "XGBoost should have _extract_features method"
    assert hasattr(xgb_model, "timing"), "XGBoost should have timing attribute"
    print("XGBoost model initialization successful")
except Exception as e:
    print(f"XGBoost model initialization failed: {e}")

# Test feature extraction on models
print("Testing feature extraction on models...")
rf_features = rf_model._extract_features(rgb_data)
xgb_features = xgb_model._extract_features(rgb_data)

assert np.allclose(
    rf_features, xgb_features
), "Both models should produce identical features"
assert np.allclose(
    rf_features, features
), "Model features should match utility function"

print("Tests passed.")
