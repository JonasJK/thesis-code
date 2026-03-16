import json
import logging
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

import xgboost as xgb

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import BaseNirModel, predict_and_save_nir, profile_execution
from utils.evaluation import xgboost_eval
from utils.memory_utils import log_cuda_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m-%d %H:%M:%S:",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Core XGBoost Logic.


class XGBoostNir(BaseNirModel):
    def __init__(self, config_path="config.json", validation_split=0.3):
        super().__init__()
        self.validation_split = validation_split

        self.config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    self.config = json.load(f)
            except Exception as e:
                logging.warning(f"Could not load config from {config_path}: {e}")

        log_cuda_info()

        self.model = xgb.XGBRegressor(
            n_estimators=self.config.get("n_estimators", 1000),
            max_depth=self.config.get("max_depth", 6),
            learning_rate=self.config.get("learning_rate", 0.1),
            subsample=self.config.get("subsample", 0.8),
            colsample_bytree=self.config.get("colsample_bytree", 0.8),
            random_state=self.config.get("random_state", 42),
            n_jobs=self.config.get("n_jobs", -1),
            device="cuda",
            tree_method=self.config.get("tree_method", "hist"),
            early_stopping_rounds=self.config.get("early_stopping_rounds", 50),
            enable_categorical=False,
            eval_metric=xgboost_eval,
        )

    def fit_model(self):
        log.info(f"Fitting XGBoost model on {self.sample_count} samples")
        start_time = time.time()

        X = self.training_rgb[: self.sample_count]
        y = self.training_nir[: self.sample_count]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_split, random_state=42)
        log.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")

        # Convert to DMatrix for low-level API
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            "objective": "reg:squarederror",
            "tree_method": self.config.get("tree_method", "hist"),
            "learning_rate": self.config.get("learning_rate", 0.1),
            "max_depth": self.config.get("max_depth", 6),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "device": "cuda",
            "seed": self.config.get("random_state", 42),
        }

        num_boost_round = self.config.get("n_estimators", 1000)
        early_stopping_rounds = self.config.get("early_stopping_rounds", 10)

        best_score = float("inf")
        best_iteration = 0
        no_improve_rounds = 0

        booster = xgb.Booster(params, [dtrain])

        for i in range(num_boost_round):
            booster.update(dtrain, i)
            y_pred_val = booster.predict(dval)
            score_dict = xgboost_eval(y_val, y_pred_val)
            combined_score = score_dict.get("RMSE", score_dict.get("combined", np.inf))
            if combined_score < best_score - 0.01:
                best_score = combined_score
                best_iteration = i
                no_improve_rounds = 0
                booster_best = booster.copy()
            else:
                no_improve_rounds += 1
            if no_improve_rounds >= early_stopping_rounds:
                log.info(f"Early stopping triggered at iteration {best_iteration}, score={best_score:.4f}")
                break
            if i % 10 == 0:
                log.info(f"Iteration {i}: validation score={combined_score:.4f}")

        duration = time.time() - start_time
        self.timing["fit"] = duration

        self.model = xgb.XGBRegressor()
        self.model._Booster = booster_best

        try:
            self.feature_importances_ = booster_best.get_score(importance_type="weight")
            log.info(f"Feature importances: {self.feature_importances_}")
        except Exception:
            self.feature_importances_ = None

        log.info(f"Model fit complete in {duration:.2f}s")
        self._print_memory_info("fit_model")

    def predict_image(self, rgb_image):
        log.info(f"Predicting NIR for image of shape {rgb_image.shape}")
        log.info("Using CUDA device for prediction")
        self._print_memory_info("predict_start")
        start = time.time()
        rgb_flat = rgb_image.reshape(-1, 3).astype(np.float32, copy=False)
        features = self._extract_features(rgb_flat)
        pred_nir = self.model.predict(features)
        duration = time.time() - start
        self.timing["predict"] += duration
        log.info(
            f"Prediction for image (shape={rgb_image.shape}) took {duration:.2f}s (cumulative predict time: {self.timing['predict']:.2f}s)"
        )
        self._print_memory_info("predict_end")
        return pred_nir.reshape(rgb_image.shape[:2])

    def save_model(self, filepath="xgboost_model.pkl"):
        """
        Save the trained XGBoost model to a file for easy reuse.

        Args:
            filepath: Path where the model will be saved (default: xgboost_model.pkl)
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")

        model_data = {
            "model": self.model,
            "config": self.config,
            "validation_split": self.validation_split,
            "sample_count": self.sample_count,
            "feature_importances": self.feature_importances_,
            "timing": self.timing,
            "timestamp": datetime.now().isoformat(),
            "xgboost_version": xgb.__version__,
            "numpy_version": np.__version__,
        }

        joblib.dump(model_data, filepath, compress=3)

        log.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath="xgboost_model.pkl"):
        """
        Load a trained XGBoost model from a file.

        Args:
            filepath: Path to the saved model file

        Returns:
            Loaded XGBoostNir instance
        """
        model_data = joblib.load(filepath)

        # Create a new instance with the saved config
        instance = cls.__new__(cls)
        BaseNirModel.__init__(instance)

        instance.model = model_data["model"]
        instance.config = model_data["config"]
        instance.validation_split = model_data["validation_split"]
        instance.sample_count = model_data["sample_count"]
        instance.feature_importances_ = model_data["feature_importances"]
        instance.timing = model_data["timing"]

        log.info(f"Model loaded from {filepath}")
        log.info(f"  Trained on {instance.sample_count:,} samples")
        log.info(f"  Validation split: {instance.validation_split:.1%}")
        log.info(f"  Saved at: {model_data['timestamp']}")

        if instance.config:
            log.info("  Hyperparameters:")
            for key, value in instance.config.items():
                log.info(f"    {key}: {value}")

        if instance.feature_importances_ is not None:
            log.info(f"  Feature importances: {instance.feature_importances_}")

        return instance


@profile_execution
def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Train and evaluate XGBoost NIR prediction model")
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
        "--eval-data-source",
        type=str,
        help="Separate data source for evaluation (optional)",
        default=None,
    )
    parser.add_argument(
        "--zip-dir",
        action="store_true",
        help="Treat data-source as directory of ZIP files",
    )
    parser.add_argument(
        "--train-test-split",
        type=float,
        help="Fraction of files to use for training (rest for evaluation)",
        default=0.8,
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
        model = XGBoostNir.load_model(args.load_model)
    else:
        model = XGBoostNir(validation_split=0.3)

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from DataLoader import DataLoader

    # Training phase (skip if model is loaded)
    if not args.load_model:
        try:
            if args.eval_data_source:
                log.info(
                    f"Using separate data sources - Training: {args.data_source}, Evaluation: {args.eval_data_source}"
                )

                with DataLoader(args.data_source) as data_loader:
                    log.info(f"Training DataLoader initialized: {data_loader}")
                    model.train_from_data_loader(data_loader, max_files=args.n)
                    model.fit_model()
                    print(f"Training complete. Samples used: {model.sample_count}")

                    # Save the trained model
                    if args.save_model:
                        model_filename = args.save_model
                    else:
                        n_files = args.n if args.n else "all"
                        model_filename = f"xgboost_model_{n_files}files.pkl"
                    model.save_model(filepath=model_filename)

            else:
                # Use single data source with automatic train/test split based on training files tracking
                log.info(f"Using single data source with automatic train/test split: {args.data_source}")

                with DataLoader(args.data_source) as data_loader:
                    log.info(f"DataLoader initialized: {data_loader}")

                    total_files = len(data_loader)
                    max_train_files = int(total_files * args.train_test_split) if args.n is None else args.n

                    log.info(
                        f"Total files: {total_files}, Using {max_train_files} for training ({args.train_test_split:.1%} split)"
                    )

                    model.train_from_data_loader(data_loader, max_files=max_train_files)
                    model.fit_model()
                    print(f"Training complete. Samples used: {model.sample_count}")

                    # Save the trained model
                    if args.save_model:
                        model_filename = args.save_model
                    else:
                        n_files = max_train_files if max_train_files else "all"
                        model_filename = f"xgboost_model_{n_files}files.pkl"
                    model.save_model(filepath=model_filename)

        except Exception as e:
            log.error(f"Failed to initialize DataLoader or train model: {e}")
            return
    else:
        log.info("Skipping training, model already loaded")

    try:
        if args.eval_data_source:
            log.info(f"Evaluating on separate data source: {args.eval_data_source}")
            with DataLoader(args.eval_data_source) as eval_data_loader:
                log.info(f"Evaluation DataLoader initialized: {eval_data_loader}")
                eval_results = model.evaluate_files(
                    eval_data_loader,
                    downscale_to=None,
                    sample_limit=5000000,
                    max_files=min(args.n or 500, 500),
                )
                if eval_results is not None:
                    print("Evaluation results:")
                    print(f"  RMSE: {eval_results['RMSE']:.4f}")
                    print(f"  R²:   {eval_results['R2']:.4f}")
                    print(f"  MAE:  {eval_results['MAE']:.4f}")
                    print(f"  SSIM: {eval_results['SSIM']:.4f}")
        else:
            log.info(f"Evaluating on data source: {args.data_source}")
            with DataLoader(args.data_source) as eval_data_loader:
                log.info(f"Created separate DataLoader for evaluation: {eval_data_loader}")
                if hasattr(model, "training_files") and model.training_files:
                    log.info(f"Evaluation will automatically skip {len(model.training_files)} files used in training")
                eval_results = model.evaluate_files(
                    eval_data_loader,
                    downscale_to=None,
                    sample_limit=5000000,
                    max_files=min(args.n or 20, 500),
                )
                if eval_results is not None:
                    print("Evaluation results:")
                    print(f"  RMSE: {eval_results['RMSE']:.4f}")
                    print(f"  R²:   {eval_results['R2']:.4f}")
                    print(f"  MAE:  {eval_results['MAE']:.4f}")
                    print(f"  SSIM: {eval_results['SSIM']:.4f}")
                    print(f"  Files used for evaluation: {eval_results['n_files']}")
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
