import csv
import gc
import json
import logging
import os
import sys
from datetime import datetime

import optuna

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import importlib.util
import sys

# Import XGBoostNir
xgb_path = os.path.join(os.path.dirname(__file__), "xgboost_nir.py")
spec = importlib.util.spec_from_file_location("xgboost_nir", xgb_path)
xgb_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xgb_module)
XGBoostNir = xgb_module.XGBoostNir

from DataLoader.DataLoader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


class ModelPerformanceTracker:
    def __init__(self, model_name="XGBoostNIR"):
        self.model_name = model_name
        self.experiments = []

    def track_experiment(self, model, eval_results, hyperparams, dataset_info):
        experiment = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "hyperparameters": hyperparams,
            "dataset": dataset_info,
            "performance_metrics": eval_results,
            "computational_metrics": {
                "training_time": model.timing.get("fit", 0),
                "prediction_time": model.timing.get("predict", 0),
                "total_time": sum(model.timing.values()),
            },
            "feature_importance": (model.feature_importances_ if model.feature_importances_ is not None else None),
        }
        self.experiments.append(experiment)
        return experiment

    def save_results(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.experiments, f, indent=2)


def evaluate_xgboost(data_source, results_dir, n_files=None, is_zip_dir=False, config_path="config.json"):
    """
    Evaluate an XGBoost NIR prediction model.

    Args:
        data_source: Path to file list (.txt) or directory containing data files
        results_dir: Directory to save results
        n_files: Number of files to use for training (None = use all)
        is_zip_dir: Whether data_source contains ZIP files
        config_path: Path to config file with hyperparameters
    """
    os.makedirs(results_dir, exist_ok=True)

    tracker = ModelPerformanceTracker()
    model = None

    try:
        model = XGBoostNir(config_path=config_path, validation_split=0.3)

        log.info(f"Loading data from: {data_source}")
        log.info(f"Using up to {n_files if n_files else 'all'} files for training")

        with DataLoader(data_source) as data_loader:
            log.info(f"DataLoader initialized with {len(data_loader)} files")

            log.info("Training model...")
            model.train_from_data_loader(data_loader, max_files=n_files)

            if model.sample_count == 0:
                log.error("No samples loaded for training")
                return None

            model.fit_model()
            log.info(f"Model trained on {model.sample_count} samples")

            log.info("Evaluating model...")
            eval_files_limit = min(n_files or 10, 10)  # Limit evaluation to max 10 files
            eval_results = model.evaluate_files(
                data_loader,
                downscale_to=None,
                sample_limit=5000000,
                max_files=eval_files_limit,
            )

            if eval_results:
                log.info("Evaluation done")

                hyperparams = {
                    "n_estimators": model.config.get("n_estimators", 1000),
                    "max_depth": model.config.get("max_depth", 6),
                    "learning_rate": model.config.get("learning_rate", 0.1),
                    "subsample": model.config.get("subsample", 0.8),
                    "colsample_bytree": model.config.get("colsample_bytree", 0.8),
                    "tree_method": model.config.get("tree_method", "hist"),
                    "validation_split": model.validation_split,
                }

                dataset_info = {
                    "data_source": str(data_source),
                    "n_files_available": len(data_loader),
                    "n_files_used_training": (
                        len(model.training_files) if hasattr(model, "training_files") else n_files
                    ),
                    "n_samples": model.sample_count,
                    "is_zip_dir": is_zip_dir,
                }

                experiment = tracker.track_experiment(model, eval_results, hyperparams, dataset_info)

                results_file = os.path.join(results_dir, "evaluation_results.json")
                tracker.save_results(results_file)
                log.info(f"Results saved to: {results_file}")

                print("\n" + "=" * 50)
                print("EVALUATION SUMMARY")
                print("=" * 50)
                print(f"Training samples: {model.sample_count:,}")
                print(f"RMSE: {eval_results.get('RMSE', 'N/A'):.4f}")
                print(f"R²: {eval_results.get('R2', 'N/A'):.4f}")
                print(f"MAE: {eval_results.get('MAE', 'N/A'):.4f}")
                print(f"SSIM: {eval_results.get('SSIM', 'N/A'):.4f}")
                print(f"Training time: {model.timing.get('fit', 0):.2f}s")
                print(f"Prediction time: {model.timing.get('predict', 0):.2f}s")
                print("=" * 50)

                return experiment
            else:
                log.error("Evaluation failed - no results returned")
                return None

    except Exception as e:
        log.error(f"Error during evaluation: {e}", exc_info=True)
        return None

    finally:
        if model is not None:
            del model
        gc.collect()
        log.info("Memory cleanup completed")


def append_trial_to_csv(csv_path, rmse, hyperparams):
    fieldnames = ["rmse"] + list(hyperparams.keys())
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {"rmse": rmse}
        row.update(hyperparams)
        writer.writerow(row)


def perform_hyperparameter_optimization(data_source, results_dir, n_trials=50, n_files=None):
    os.makedirs(results_dir, exist_ok=True)
    # Write CSV in the same directory as this Python file
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optuna_trials.csv")

    def objective(trial):
        # Define the hyperparameter search space for XGBoost
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 10, 50)

        # Create a temporary config dict for this trial
        config = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": early_stopping_rounds,
        }

        # Create a temporary config file
        temp_config_path = os.path.join(os.path.dirname(__file__), "temp_config.json")
        with open(temp_config_path, "w") as f:
            json.dump(config, f)

        model = XGBoostNir(config_path=temp_config_path, validation_split=0.3)
        rmse = float("inf")

        try:
            with DataLoader(data_source) as data_loader:
                log.info(
                    f"Training model with params: n_estimators={n_estimators}, max_depth={max_depth}, "
                    f"learning_rate={learning_rate:.4f}, subsample={subsample:.2f}, "
                    f"colsample_bytree={colsample_bytree:.2f}"
                )
                model.train_from_data_loader(data_loader, max_files=n_files)

                if model.sample_count == 0:
                    log.warning("No samples loaded for training")
                    rmse = float("inf")
                else:
                    model.fit_model()
                    eval_results = model.evaluate_files(
                        data_loader,
                        downscale_to=None,
                        sample_limit=1000000,
                        max_files=min(n_files or 5, 5),
                    )

                    if eval_results and "RMSE" in eval_results:
                        rmse = eval_results["RMSE"]
                        log.info(f"Trial completed with RMSE: {rmse:.4f}")
                    else:
                        log.warning("Evaluation failed, returning high penalty")
                        rmse = float("inf")

        except Exception as e:
            log.error(f"Error during trial: {e}")
            rmse = float("inf")

        finally:
            hyperparams = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "early_stopping_rounds": early_stopping_rounds,
            }
            append_trial_to_csv(csv_path, rmse, hyperparams)

            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            del model
            gc.collect()

        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Save the best parameters and results
    best_params = study.best_params
    best_value = study.best_value

    results = {
        "best_params": best_params,
        "best_value": best_value,
        "n_trials": n_trials,
        "study_summary": {
            "trials_count": len(study.trials),
            "best_trial_number": study.best_trial.number if study.best_trial else None,
        },
    }

    with open(os.path.join(results_dir, "optuna_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Update the config.json with best parameters
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "w") as f:
        json.dump(best_params, f, indent=2)

    log.info(f"Hyperparameter optimization complete. Best parameters: {best_params}, Best value: {best_value:.4f}")
    log.info("Updated config.json with best parameters")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate XGBoost model or perform hyperparameter optimization")
    parser.add_argument(
        "--data-source",
        type=str,
        required=True,
        help="Path to file list (.txt) or directory containing ZIP files",
    )
    parser.add_argument("--results-dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("-n", type=int, default=None, help="Number of files to use for training")
    parser.add_argument(
        "--zip-dir",
        action="store_true",
        help="Flag to indicate if data source contains ZIP files",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Perform hyperparameter optimization instead of evaluation",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials (only used with --optimize)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config file (only used without --optimize)",
    )

    args = parser.parse_args()

    if args.optimize:
        perform_hyperparameter_optimization(args.data_source, args.results_dir, args.n_trials, args.n)
    else:
        evaluate_xgboost(args.data_source, args.results_dir, args.n, args.zip_dir, args.config)
