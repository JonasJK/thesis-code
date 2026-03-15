"""
Evaluation utilities for NIR prediction models.
"""

import logging
import time
import traceback

import numpy as np
import rasterio
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

log = logging.getLogger(__name__)

def calculate_ssim_for_files(original_file, predicted_file, downscale_to=None):
    """Calculate SSIM between original and predicted NIR images.

    Parameters:
    -----------
    original_file : str
        Path to original RGBI file (NIR in band 4)
    predicted_file : str
        Path to predicted NIR file (single band)
    downscale_to : int, optional
        Downscale images to this size for SSIM calculation

    Returns:
    --------
    float
        SSIM score between 0 and 1
    """
    log.info(f"Calculating SSIM between {original_file} and {predicted_file}")
    with rasterio.open(original_file) as src:
        nir = src.read(4).astype(np.float32)
    with rasterio.open(predicted_file) as src:
        pred_nir = src.read(1).astype(np.float32)

    if downscale_to is not None:
        nir = resize(nir, (downscale_to, downscale_to), anti_aliasing=True)
        pred_nir = resize(pred_nir, (downscale_to, downscale_to), anti_aliasing=True)

    data_range = float(pred_nir.max() - pred_nir.min())
    if data_range == 0:
        log.info(
            "Predicted NIR image has zero data range; SSIM undefined. Returning 0."
        )
        return 0.0

    result = ssim(nir, pred_nir, data_range=data_range)
    log.info(f"SSIM result: {result:.4f}")
    return result

def predict_and_save_nir(model, input_rgb_file, output_nir_file):
    """
    Takes an RGB image file, predicts the NIR band using the trained model,
    and saves the result as a single-band GeoTIFF file.

    Parameters:
    -----------
    model : object
        The trained model with predict_image method
    input_rgb_file : str
        Path to the input RGB GeoTIFF file
    output_nir_file : str
        Path where the output NIR GeoTIFF will be saved

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    log.info(f"Predicting NIR from RGB file: {input_rgb_file}")
    log.info(f"Output will be saved to: {output_nir_file}")

    try:
        with rasterio.open(input_rgb_file) as src:
            if src.count < 3:
                log.info(
                    f"Error: Input file needs at least 3 bands (RGB), found {src.count}"
                )
                return False

            rgb = src.read([1, 2, 3])

            profile = src.profile.copy()

            rgb_for_prediction = np.moveaxis(rgb, 0, -1)

            log.info(f"RGB data shape for prediction: {rgb_for_prediction.shape}")

            start_time = time.time()
            predicted_nir = model.predict_image(rgb_for_prediction)
            duration = time.time() - start_time
            log.info(f"NIR prediction completed in {duration:.2f}s")

            profile.update(
                count=1,
                dtype=rasterio.float32,
                compress="lzw",
                description="Predicted NIR band",
            )

            # Save the predicted NIR band
            with rasterio.open(output_nir_file, "w", **profile) as dst:
                dst.write(predicted_nir.astype(rasterio.float32), 1)

            log.info(f"Saved predicted NIR to: {output_nir_file}")
            return True

    except Exception as e:
        log.info(f"Error predicting/saving NIR: {e}")
        log.info(traceback.format_exc())
        return False

def xgboost_eval(y_true, y_pred):
    """
    Custom evaluation metric for XGBoost early stopping.

    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values from the model

    Returns:
    --------
    float
        Combined RMSE and SSIM score (lower is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    combined_score = (rmse + mae) / 2

    return {"RMSE": rmse, "MAE": mae, "combined": combined_score}
