import csv
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from DataLoader import DataLoader
from utils.base_model import BaseNirModel

logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",  # Set the date format to exclude milliseconds
    handlers=[logging.StreamHandler(sys.stdout)],  # Output to stdout
)
log = logging.getLogger(__name__)


class LinRegNir(BaseNirModel):
    def __init__(self):
        """
        Fast baseline linear regression: NIR = a*R + b*G + c*B + d + additional features

        """

        super().__init__()

        self.model = SGDRegressor(
            random_state=42,
            max_iter=1,
            tol=None,
            learning_rate="invscaling",  # Decreases over time: eta = eta0 / (t^power_t)
            eta0=0.01,  # Increased from 0.0001 since features will be standardized
            power_t=0.25,  # Controls how quickly learning rate decreases
            alpha=0.0001,  # L2 regularization
            penalty="l2",
            fit_intercept=True,
        )

        # Scaler for feature standardization (incremental fitting)
        self.scaler = StandardScaler()
        self.scaler_fitted = False

        self.coefficients_ = None
        self.intercept_ = None
        self.is_trained = False
        self.files_processed = 0
        self.total_pixels_trained = 0

    def partial_fit_on_file(self, file_path):
        """
        Incrementally train on a single file using partial_fit.

        Args:
            file_path: Path to the RGBI image file
        """
        log.info(f"Processing file {self.files_processed + 1}: {file_path}")
        start = time.time()

        chunks_processed = 0
        for features, nir in self.load_rgbi_image(file_path):
            self.scaler.partial_fit(features)
            self.scaler_fitted = True

            # Standardize features before training
            features_scaled = self.scaler.transform(features)

            self.model.partial_fit(features_scaled, nir)
            self.total_pixels_trained += len(features)
            chunks_processed += 1

        if chunks_processed > 0:
            self.files_processed += 1
            self.is_trained = True

            self.coefficients_ = self.model.coef_
            self.intercept_ = self.model.intercept_

            file_time = time.time() - start
            log.info(f"File processed in {file_time:.2f}s ({self.total_pixels_trained:,} total pixels trained)")
        else:
            file_time = time.time() - start
            log.warning(f"No data loaded from {file_path} (file may be corrupted or empty)")

        return file_time

    def _print_coefficients(self):
        """Print the learned coefficients."""
        if self.coefficients_ is None or self.intercept_ is None:
            log.info("No coefficients available yet (model not trained)")
            return

        feature_names = ["Red", "Green", "Blue", "VARI", "ExG", "ExGR", "CIVE", "VEG"]

        log.info("\nLearned coefficients (standardized features):")
        for i, (name, coeff) in enumerate(zip(feature_names, self.coefficients_, strict=False)):
            log.info(f"  {name}: {coeff:.6f}")
        log.info(f"  Intercept: {self.intercept_[0]:.6f}")

        # Show feature scales if scaler is fitted
        if self.scaler_fitted:
            log.info("\nFeature scales (mean ± std):")
            for i, name in enumerate(feature_names):
                log.info(f"  {name}: {self.scaler.mean_[i]:.2f} ± {np.sqrt(self.scaler.var_[i]):.2f}")

        abs_coeffs = np.abs(self.coefficients_)
        max_idx = np.argmax(abs_coeffs)
        log.info(f"Strongest predictor: {feature_names[max_idx]} (coeff = {self.coefficients_[max_idx]:.6f})")

    def predict_image(self, rgb_image: np.ndarray) -> np.ndarray:
        """Predict NIR for an RGB image.

        Args:
            rgb_image: RGB image array of shape (height, width, 3)

        Returns:
            Predicted NIR image array of shape (height, width)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        if not self.scaler_fitted:
            raise ValueError("Scaler must be fitted first")

        height, width = rgb_image.shape[:2]
        rgb_pixels = rgb_image.reshape(-1, 3)

        features = self._extract_features(rgb_pixels)

        # Apply same standardization as during training
        features_scaled = self.scaler.transform(features)

        nir_predicted = self.model.predict(features_scaled)

        return nir_predicted.reshape(height, width)

    def train(self, data_source, max_files=None):
        """
        Train the linear regression model using DataLoader in streaming mode.
        Each file is loaded, trained on, and discarded before moving to the next.

        Args:
            data_source: Either a DataLoader instance or path to file list
            max_files: Maximum number of files to process

        Returns:
            results: Dictionary with training metrics and timing
        """
        log.info("=" * 60)
        log.info("LINEAR REGRESSION NIR PREDICTION (STREAMING MODE)")
        log.info("=" * 60)

        overall_start = time.time()
        self._print_memory_info("train_start")

        if isinstance(data_source, (str, Path)):
            with DataLoader(data_source) as data_loader:
                return self._train_with_loader(data_loader, max_files, overall_start)
        else:
            return self._train_with_loader(data_source, max_files, overall_start)

    def _train_with_loader(self, data_loader, max_files, overall_start):
        """Internal training method using DataLoader in streaming mode."""
        log.info("Streaming training: Processing files one by one with partial_fit...")

        train_start = time.time()
        file_count = 0

        for file_path, _is_temp in data_loader.iterate_files():
            if max_files and file_count >= max_files:
                log.info(f"Reached max_files limit ({max_files})")
                break

            try:
                self.partial_fit_on_file(file_path)
                file_count += 1

                if file_count % 10 == 0:
                    self._print_coefficients()
                    self._print_memory_info(f"after_{file_count}_files")

            except Exception as e:
                log.error(f"Error processing {file_path}: {e}")
                continue

        train_time = time.time() - train_start
        self.timing["fit"] = train_time

        log.info(f"\nTraining complete: Processed {file_count} files")
        self._print_coefficients()

        script_dir = Path(__file__).parent.resolve()
        if max_files:
            model_filename = script_dir / f"linear_regression_model_{max_files}files.pkl"
        else:
            model_filename = script_dir / f"linear_regression_model_{file_count}files.pkl"
        self.save_model(filepath=str(model_filename))

        log.info("\nPhase 2: Evaluation...")
        eval_start = time.time()

        eval_results = self.evaluate_files(data_loader, max_files=max_files)

        eval_time = time.time() - eval_start
        self.timing["evaluate"] = eval_time

        total_time = time.time() - overall_start

        results = {
            "n_train": self.total_pixels_trained,
            "files_processed": file_count,
            "total_time": total_time,
            "train_time": train_time,
            "eval_time": eval_time,
            "pixels_per_second": (self.total_pixels_trained / total_time if total_time > 0 else 0),
            **eval_results,  # Add RMSE, R², MAE, SSIM from evaluation
        }

        self._print_results(results)
        self._print_memory_info("final")

        return results

    def _print_results(self, results):
        """Print training and evaluation results."""
        log.info("\n" + "=" * 60)
        log.info("RESULTS")
        log.info("=" * 60)

        if "RMSE" in results:
            log.info(f"RMSE: {results['RMSE']:.4f}")
        if "R²" in results:
            log.info(f"R²: {results['R²']:.4f}")
        if "MAE" in results:
            log.info(f"MAE: {results['MAE']:.4f}")
        if "SSIM" in results:
            log.info(f"SSIM: {results['SSIM']:.4f}")

        log.info("\nPerformance:")
        log.info(f"Total time: {results['total_time']:.1f}s")
        if "train_time" in results:
            log.info(f"Training time: {results['train_time']:.1f}s")
        log.info(f"Evaluation time: {results['eval_time']:.1f}s")
        log.info(f"Processing speed: {results['pixels_per_second']:,.0f} pixels/second")

        log.info("\nTraining data:")
        log.info(f"Files processed: {results.get('files_processed', 'N/A')}")
        log.info(f"Total pixels trained: {results['n_train']:,}")

    def append_coefficients_to_csv(self, csv_file="coefficients.csv"):
        """
        Append the current coefficients to a CSV file.

        Args:
            csv_file: Path to the CSV file where coefficients will be saved.
        """
        if self.coefficients_ is None or self.intercept_ is None:
            raise ValueError("Model coefficients and intercept are not available. Train the model first.")

        coeffs = list(self.coefficients_) + [self.intercept_[0]]

        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(coeffs)

        log.info(f"Appended coefficients to {csv_file}")
        self._print_coefficients()

    def save_model(self, filepath="linear_regression_model.pkl"):
        """
        Save the trained model to a file for easy reuse.

        Args:
            filepath: Path where the model will be saved (default: linear_regression_model.pkl)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "scaler_fitted": self.scaler_fitted,
            "coefficients": self.coefficients_,
            "intercept": self.intercept_,
            "is_trained": self.is_trained,
            "files_processed": self.files_processed,
            "total_pixels_trained": self.total_pixels_trained,
            "timestamp": datetime.now().isoformat(),
            "sklearn_version": __import__("sklearn").__version__,
            "numpy_version": np.__version__,
        }

        # Convert to absolute path for clarity
        abs_filepath = Path(filepath).resolve()
        joblib.dump(model_data, str(abs_filepath), compress=3)

        log.info(f"Model saved to: {abs_filepath}")
        log.info(f"File size: {abs_filepath.stat().st_size / (1024 * 1024):.2f} MB")

    @classmethod
    def load_model(cls, filepath="linear_regression_model.pkl"):
        """
        Load a trained model from a file.

        Args:
            filepath: Path to the saved model file

        Returns:
            Loaded LinRegNir instance
        """
        model_data = joblib.load(filepath)

        instance = cls()

        instance.model = model_data["model"]
        instance.scaler = model_data.get("scaler", StandardScaler())  # Backwards compatibility
        instance.scaler_fitted = model_data.get("scaler_fitted", False)
        instance.coefficients_ = model_data["coefficients"]
        instance.intercept_ = model_data["intercept"]
        instance.is_trained = model_data["is_trained"]
        instance.files_processed = model_data["files_processed"]
        instance.total_pixels_trained = model_data["total_pixels_trained"]

        log.info(f"Model loaded from {filepath}")
        log.info(f"  Trained on {instance.files_processed} files ({instance.total_pixels_trained:,} pixels)")
        log.info(f"  Saved at: {model_data['timestamp']}")
        log.info(f"  Scaler fitted: {instance.scaler_fitted}")

        return instance


if __name__ == "__main__":
    try:
        from memory_profiler import profile as memory_profile
    except ImportError:

        def memory_profile(func):
            return func

    # @memory_profile
    def run_training():
        log.info(f"Script location: {Path(__file__).resolve()}")
        log.info(f"Working directory: {Path.cwd()}")

        file_path = "/data/lacy-vme/khant/dop/sn-2023/"

        #     raise FileNotFoundError(f"File list not found at {file_list_path}")

        num_files = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        log.info(f"Will use up to {num_files} files for streaming training")

        data_loader = DataLoader(file_path)
        log.info(f"DataLoader created with {len(data_loader)} total files")

        model = LinRegNir()

        try:
            log.info(f"Initial memory usage: {model._get_memory_usage_mb():.1f} MB")

            model.train(data_loader, max_files=num_files)

            log.info("\nTraining complete.")
            log.info(f"Final memory usage: {model._get_memory_usage_mb():.1f} MB")

            # Model is already saved during training
            # model.append_coefficients_to_csv(csv_file="coefficients.csv")

        except Exception as e:
            log.error(f"Error during training: {e}")
            import traceback

            traceback.print_exc()
        finally:
            data_loader.cleanup()

    log.info("Starting linear regression training in streaming mode with DataLoader...")
    run_training()
