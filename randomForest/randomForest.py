import logging
import os
import sys
import time
from datetime import datetime

import cupy as cp
import joblib
import numpy as np
from cuml.ensemble import RandomForestRegressor as cuRF

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import BaseNirModel, predict_and_save_nir, profile_execution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Core Random Forest Logic.


class RandomForestNir(BaseNirModel):
    def __init__(self, max_samples=600_000_000, max_depth=22, n_estimators=298):
        super().__init__(max_samples)
        self.max_depth = max_depth
        self.n_estimators = n_estimators

        self.model = cuRF(n_estimators=n_estimators, max_depth=max_depth, max_features=1)
        log.info(
            f"Initialized cuML RandomForestNir(max_samples={max_samples}, n_estimators={n_estimators}, max_depth={max_depth})"
        )

    def fit_model(self):
        log.info(f"Fitting RandomForest model on {self.sample_count} samples")
        start = time.time()

        X = cp.asarray(self.training_rgb[: self.sample_count], dtype=cp.float32)
        y = cp.asarray(self.training_nir[: self.sample_count], dtype=cp.float32)

        self.model.fit(X, y)
        duration = time.time() - start
        self.timing["fit"] = duration

        try:
            self.feature_importances_ = cp.asnumpy(self.model.feature_importances_)
            log.info(f"Feature importances: {self.feature_importances_}")
        except Exception as e:
            self.feature_importances_ = None
            log.info("Could not retrieve feature importances: " + str(e))

        log.info(f"Model fit complete in {duration:.2f}s")
        self._print_memory_info("fit_model")

    def predict_image(self, rgb_image):
        log.info(f"Predicting NIR for image of shape {rgb_image.shape}")
        self._print_memory_info("predict_start")
        start = time.time()

        rgb_flat = rgb_image.reshape(-1, 3).astype(np.float32, copy=False)
        features = self._extract_features(rgb_flat)
        features_gpu = cp.asarray(features, dtype=cp.float32)

        pred_nir_gpu = self.model.predict(features_gpu)
        pred_nir = cp.asnumpy(pred_nir_gpu)

        duration = time.time() - start
        self.timing["predict"] += duration
        log.info(
            f"Prediction for image (shape={rgb_image.shape}) took {duration:.2f}s (cumulative predict time: {self.timing['predict']:.2f}s)"
        )
        self._print_memory_info("predict_end")

        return pred_nir.reshape(rgb_image.shape[:2])

    def save_model(self, filepath="random_forest_model.pkl"):
        """
        Save the trained Random Forest model to a file for easy reuse.

        Args:
            filepath: Path where the model will be saved (default: random_forest_model.pkl)
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")

        model_data = {
            "model": self.model,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "sample_count": self.sample_count,
            "feature_importances": self.feature_importances_,
            "timing": self.timing,
            "timestamp": datetime.now().isoformat(),
            "cuml_version": __import__("cuml").__version__,
            "cupy_version": cp.__version__,
            "numpy_version": np.__version__,
        }

        joblib.dump(model_data, filepath, compress=3)

        log.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath="random_forest_model.pkl"):
        """
        Load a trained Random Forest model from a file.

        Args:
            filepath: Path to the saved model file

        Returns:
            Loaded RandomForestNir instance
        """
        model_data = joblib.load(filepath)

        # Create a new instance with the saved hyperparameters
        instance = cls(
            max_samples=model_data["max_samples"],
            max_depth=model_data["max_depth"],
            n_estimators=model_data["n_estimators"],
        )

        instance.model = model_data["model"]
        instance.sample_count = model_data["sample_count"]
        instance.feature_importances_ = model_data["feature_importances"]
        instance.timing = model_data["timing"]

        log.info(f"Model loaded from {filepath}")
        log.info(f"  Trained on {instance.sample_count:,} samples")
        log.info(f"  Hyperparameters: n_estimators={instance.n_estimators}, max_depth={instance.max_depth}")
        log.info(f"  Saved at: {model_data['timestamp']}")

        if instance.feature_importances_ is not None:
            feature_names = [
                "Red",
                "Green",
                "Blue",
                "VARI",
                "ExG",
                "ExGR",
                "CIVE",
                "VEG",
            ]
            log.info("  Feature importances:")
            for name, importance in zip(feature_names, instance.feature_importances_, strict=False):
                log.info(f"    {name}: {importance:.4f}")

        return instance


@profile_execution
def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Train and evaluate RandomForest NIR prediction model")
    parser.add_argument(
        "-n",
        type=int,
        help="Number of files to use for training (random sample)",
        default=None,
    )
    parser.add_argument(
        "--predict",
        type=str,
        help="Path to RGB image file for prediction",
        default=None,
    )
    parser.add_argument("--output", type=str, help="Path to save predicted NIR image", default=None)
    parser.add_argument(
        "--data-source",
        type=str,
        help="Path to file list (.txt) or directory containing ZIP files",
        default="/home/klugej/thesis/rgbi_files.txt",
    )
    parser.add_argument(
        "--zip-dir",
        action="store_true",
        help="Treat data-source as directory of ZIP files",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        help="Path to saved model file to load instead of training",
        default=None,
    )
    parser.add_argument("--save-model", type=str, help="Path to save the trained model", default=None)

    args = parser.parse_args()

    # Load existing model or create new one
    if args.load_model:
        log.info(f"Loading model from {args.load_model}")
        model = RandomForestNir.load_model(args.load_model)
    else:
        model = RandomForestNir()

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from DataLoader import DataLoader

    if not args.load_model:
        try:
            with DataLoader(args.data_source) as data_loader:
                log.info(f"DataLoader initialized: {data_loader}")

                model.train_from_data_loader(data_loader, max_files=args.n)
                model.fit_model()
                print(f"Training complete. Samples used: {model.sample_count}")

                # Save the trained model
                if args.save_model:
                    model_filename = args.save_model
                else:
                    n_files = args.n if args.n else "all"
                    model_filename = f"random_forest_model_{n_files}files.pkl"
                model.save_model(filepath=model_filename)

        except Exception as e:
            log.error(f"Failed to initialize DataLoader: {e}")
            return
    else:
        log.info("Skipping training, model already loaded")

    if not (args.predict and args.output):
        try:
            with DataLoader(args.data_source) as eval_data_loader:
                log.info(f"Created DataLoader for evaluation: {eval_data_loader}")
                max_eval_files = min(args.n, 1000) if args.n else 1000
                eval_results = model.evaluate_files(
                    eval_data_loader,
                    downscale_to=None,
                    sample_limit=5_000_000,
                    max_files=max_eval_files,
                )
                if eval_results is not None:
                    print("Evaluation results:")
                    print(f"  RMSE: {eval_results['RMSE']:.4f}")
                    print(f"  R²:   {eval_results['R2']:.4f}")
                    print(f"  MAE:  {eval_results['MAE']:.4f}")
                    print(f"  SSIM: {eval_results['SSIM']:.4f}")
        except Exception as e:
            log.error(f"Evaluation failed: {e}")

    # Handle prediction if requested
    if args.predict and args.output:
        log.info(f"Predicting NIR for {args.predict} and saving to {args.output}")
        success = predict_and_save_nir(model, args.predict, args.output)
        if success:
            log.info("NIR prediction and saving completed")
        else:
            log.error("NIR prediction or saving failed")
    elif args.predict or args.output:
        log.error("Both --predict and --output arguments must be provided to generate NIR prediction")


if __name__ == "__main__":
    main()
