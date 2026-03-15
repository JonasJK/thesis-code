import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

def detect_model_type(model_path):
    """Detect model type from file extension and path."""
    model_path = Path(model_path)

    if model_path.suffix == ".pth":
        return "cnn"
    elif model_path.suffix == ".pkl":
        name_lower = model_path.name.lower()
        if "linear" in name_lower or "linreg" in name_lower:
            return "linear_regression"
        elif "xgb" in name_lower or "xgboost" in name_lower:
            return "xgboost"
        elif "forest" in name_lower or "rf" in name_lower:
            return "random_forest"
        else:
            log.warning(
                f"Cannot determine model type from filename '{model_path.name}'; defaulting to 'linear_regression'"
            )
            return "linear_regression"
    else:
        raise ValueError(f"Unsupported model file extension: {model_path.suffix}")

def load_rgbi_image(file_path):
    """
    Load RGBI image and return RGB and NIR separately.

    Args:
        file_path: Path to RGBI TIFF file

    Returns:
        rgb_image: numpy array (H, W, 3)
        nir_image: numpy array (H, W) - ground truth if available
        metadata: rasterio profile for saving output
    """
    import rasterio

    with rasterio.open(file_path) as src:
        data = src.read()  # Shape: (4, H, W) for RGBI

        if data.shape[0] >= 3:
            rgb = data[:3].transpose(1, 2, 0)  # (H, W, 3)
        else:
            raise ValueError(f"Expected at least 3 bands in image, got {data.shape[0]}")

        # Ground-truth NIR is optional and only used for comparison.
        nir = data[3] if data.shape[0] >= 4 else None

        # Reuse source metadata for the output raster.
        profile = src.profile.copy()
        profile.update(count=1, dtype=rasterio.float32)

    return (
        rgb.astype(np.uint8),
        nir.astype(np.float32) if nir is not None else None,
        profile,
    )

def save_nir_image(nir_array, output_path, profile):
    """
    Save NIR prediction as a GeoTIFF.

    Args:
        nir_array: numpy array (H, W)
        output_path: Path to save output
        profile: rasterio profile with metadata
    """
    import rasterio

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(nir_array.astype(np.float32), 1)

    log.info(f"NIR prediction saved to: {output_path}")

def load_model(model_path, model_type):
    """
    Load a trained model based on model type.

    Args:
        model_path: Path to model file
        model_type: Type of model ('linear_regression', 'cnn', 'xgboost', 'random_forest')

    Returns:
        Loaded model instance
    """
    script_dir = Path(__file__).parent.resolve()
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    log.info(f"Loading {model_type} model from {model_path}")

    if model_type == "linear_regression":
        from linearRegression.linearRegression import LinRegNir

        model = LinRegNir.load_model(model_path)

    elif model_type == "cnn":
        import torch

        from CNN.cnn_nir import CNNNir, NIRPredictionCNN

        model = CNNNir()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.model = NIRPredictionCNN().to(device)
        model.model.load_state_dict(torch.load(model_path, map_location=device))
        model.model.eval()
        model.device = device

        log.info(f"CNN model loaded on device: {device}")

    elif model_type == "xgboost":
        from XGBoost.xgboost_nir import XGBoostNir

        model = XGBoostNir.load_model(model_path)

    elif model_type == "random_forest":
        from randomForest.randomForest import RandomForestNir

        model = RandomForestNir.load_model(model_path)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model

def predict_nir(model, rgb_image, model_type):
    """
    Predict NIR from RGB image using the loaded model.

    Args:
        model: Loaded model instance
        rgb_image: numpy array (H, W, 3)
        model_type: Type of model

    Returns:
        nir_prediction: numpy array (H, W)
    """
    log.info(f"Predicting NIR for image of shape {rgb_image.shape}")
    start_time = time.time()

    nir_prediction = model.predict_image(rgb_image)

    duration = time.time() - start_time
    log.info(f"Prediction completed in {duration:.2f}s")

    return nir_prediction

def calculate_metrics(nir_true, nir_pred):
    """
    Calculate evaluation metrics if ground truth is available.

    Args:
        nir_true: Ground truth NIR array (H, W)
        nir_pred: Predicted NIR array (H, W)

    Returns:
        Dictionary with metrics
    """
    from skimage.metrics import structural_similarity as ssim
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    nir_true_flat = nir_true.flatten()
    nir_pred_flat = nir_pred.flatten()

    rmse = np.sqrt(mean_squared_error(nir_true_flat, nir_pred_flat))
    r2 = r2_score(nir_true_flat, nir_pred_flat)
    mae = mean_absolute_error(nir_true_flat, nir_pred_flat)

    data_range = nir_true.max() - nir_true.min()
    ssim_score = (
        ssim(nir_true, nir_pred, data_range=data_range) if data_range > 0 else 0.0
    )

    metrics = {"RMSE": rmse, "R²": r2, "MAE": mae, "SSIM": ssim_score}

    return metrics

def main():
    parser = argparse.ArgumentParser(
        description="Perform NIR inference on a single RGBI image using a trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model file (.pkl for sklearn models, .pth for CNN)",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input RGBI image file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["linear_regression", "cnn", "xgboost", "random_forest"],
        help="Type of model (auto-detected if not specified)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare prediction with ground truth NIR band (if available)",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input)

    if not model_path.exists():
        log.error(f"Model file not found: {model_path}")
        sys.exit(1)

    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Use the requested model type, or infer it from the model file name.
    model_type = args.model_type if args.model_type else detect_model_type(model_path)
    log.info(f"Model type: {model_type}")

    # Write output next to the input file and append a model suffix.
    model_suffix_map = {
        "linear_regression": "lr",
        "cnn": "cnn",
        "xgboost": "xgb",
        "random_forest": "rf",
    }
    suffix = model_suffix_map.get(model_type, model_type)
    output_path = Path(
        input_path.parent, f"{input_path.stem}_{suffix}{input_path.suffix}"
    )
    log.info(f"Output will be saved to: {output_path}")

    try:
        log.info(f"Loading image: {input_path}")
        rgb_image, nir_true, profile = load_rgbi_image(input_path)
        log.info(
            f"Image shape: RGB={rgb_image.shape}, NIR={'None' if nir_true is None else nir_true.shape}"
        )

        model = load_model(model_path, model_type)

        nir_prediction = predict_nir(model, rgb_image, model_type)

        log.info(f"Saving prediction to: {output_path}")
        save_nir_image(nir_prediction, output_path, profile)

        if args.compare and nir_true is not None:
            log.info("\n" + "=" * 60)
            log.info("COMPARISON WITH GROUND TRUTH")
            log.info("=" * 60)

            metrics = calculate_metrics(nir_true, nir_prediction)

            log.info(f"RMSE: {metrics['RMSE']:.4f}")
            log.info(f"R²:   {metrics['R²']:.4f}")
            log.info(f"MAE:  {metrics['MAE']:.4f}")
            log.info(f"SSIM: {metrics['SSIM']:.4f}")

        elif args.compare and nir_true is None:
            log.warning(
                "Comparison requested but no ground truth NIR band found in input image"
            )

        log.info("\n" + "=" * 60)
        log.info("INFERENCE COMPLETE")
        log.info("=" * 60)
        log.info(f"Model: {model_path}")
        log.info(f"Input: {input_path}")
        log.info(f"Output: {output_path}")

    except Exception as e:
        log.error(f"Inference failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
