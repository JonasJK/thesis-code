import csv
import gc
import json
import logging
import os
import sys
from datetime import datetime

import optuna
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import importlib.util

cnn_path = os.path.join(os.path.dirname(__file__), "cnn_nir.py")
spec = importlib.util.spec_from_file_location("cnn_nir", cnn_path)
cnn_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnn_module)
CNNNir = cnn_module.CNNNir

from DataLoader.DataLoader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

class ModelPerformanceTracker:

    def __init__(self, model_name="CNN_NIR"):
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
                "data_processing_time": model.timing.get("processing", 0),
                "training_time": model.timing.get("fit", 0),
                "prediction_time": model.timing.get("predict", 0),
                "evaluation_time": model.timing.get("evaluate", 0),
                "total_time": sum(model.timing.values()),
            },
        }
        self.experiments.append(experiment)
        return experiment

    def save_results(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.experiments, f, indent=2)

def evaluate_cnn(
    data_source, results_dir, n_files=None, is_zip_dir=False, config_path="config.json"
):
    """
    Evaluate a CNN NIR prediction model.

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
        config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    config = json.load(f)
                log.info(f"Loaded config from {config_path}")
            except Exception as e:
                log.warning(f"Could not load config from {config_path}: {e}")

        model = CNNNir(
            patch_size=config.get("patch_size", 50),
            batch_size=config.get("batch_size", 32),
            epochs=config.get("epochs", 20),
            learning_rate=config.get("learning_rate", 0.001),
            validation_split=config.get("validation_split", 0.2),
            num_workers=config.get("num_workers", None),
        )

        log.info(f"Loading data from: {data_source}")
        log.info(f"Using up to {n_files if n_files else 'all'} files for training")

        with DataLoader(data_source) as data_loader:
            log.info(f"DataLoader initialized with {len(data_loader)} files")

            log.info("Training model...")
            model.train_from_data_loader(data_loader, max_files=n_files)

            if len(model.patches_rgb) == 0:
                log.error("No patches loaded for training")
                return None

            model.fit_model()
            log.info(f"Model trained on {len(model.patches_rgb)} patches")

            log.info("Evaluating model...")
            eval_files_limit = min(
                n_files or 10, 10
            )  # Limit evaluation to max 10 files
            eval_results = model.evaluate_files(data_loader, max_files=eval_files_limit)

            if eval_results:
                log.info("Evaluation done")

                hyperparams = {
                    "patch_size": model.patch_size,
                    "batch_size": model.batch_size,
                    "epochs": model.epochs,
                    "learning_rate": model.learning_rate,
                    "validation_split": model.validation_split,
                    "num_workers": model.num_workers,
                }

                dataset_info = {
                    "data_source": str(data_source),
                    "n_files_available": len(data_loader),
                    "n_files_used_training": (
                        len(model.training_files)
                        if hasattr(model, "training_files")
                        else n_files
                    ),
                    "n_patches": len(model.patches_rgb),
                    "is_zip_dir": is_zip_dir,
                }

                experiment = tracker.track_experiment(
                    model, eval_results, hyperparams, dataset_info
                )

                results_file = os.path.join(results_dir, "evaluation_results.json")
                tracker.save_results(results_file)
                log.info(f"Results saved to: {results_file}")

                print("\n" + "=" * 50)
                print("EVALUATION SUMMARY")
                print("=" * 50)
                print(f"Training patches: {len(model.patches_rgb):,}")
                print(f"RMSE: {eval_results.get('RMSE', 'N/A'):.4f}")
                print(f"R²: {eval_results.get('R2', 'N/A'):.4f}")
                print(f"MAE: {eval_results.get('MAE', 'N/A'):.4f}")
                print(f"SSIM: {eval_results.get('SSIM', 'N/A'):.4f}")
                print(f"Data processing time: {model.timing.get('processing', 0):.2f}s")
                print(f"Training time: {model.timing.get('fit', 0):.2f}s")
                print(f"Prediction time: {model.timing.get('predict', 0):.2f}s")
                print(f"Evaluation time: {model.timing.get('evaluate', 0):.2f}s")
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
        torch.cuda.empty_cache()
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

def perform_hyperparameter_optimization(
    data_source, results_dir, n_trials=30, n_files=None
):
    """
    Perform hyperparameter optimization for CNN model.

    Note: This is optimized for time efficiency by:
    - Limiting epochs during trials
    - Using aggressive early stopping
    - Focusing on most impactful hyperparameters
    """
    os.makedirs(results_dir, exist_ok=True)
    # Write CSV in the same directory as this Python file
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "optuna_trials.csv"
    )

    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        validation_split = trial.suggest_float("validation_split", 0.15, 0.25)

        epochs = 15
        patch_size = 50

        log.info(
            f"Trial params: batch_size={batch_size}, lr={learning_rate:.6f}, val_split={validation_split:.2f}"
        )

        model = CNNNir(
            patch_size=patch_size,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            validation_split=validation_split,
            num_workers=None,  # Auto-detect
        )

        rmse = float("inf")

        try:
            with DataLoader(data_source) as data_loader:
                trial_n_files = min(n_files or 5, 5)
                log.info(f"Training model with {trial_n_files} files for trial")

                model.train_from_data_loader(data_loader, max_files=trial_n_files)

                if len(model.patches_rgb) == 0:
                    log.warning("No patches loaded for training")
                    rmse = float("inf")
                else:
                    model.fit_model()

                    # Evaluate on a limited set
                    eval_results = model.evaluate_files(
                        data_loader,
                        max_files=min(
                            trial_n_files, 3
                        ),  # Even fewer files for evaluation
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
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "validation_split": validation_split,
                "epochs": epochs,
                "patch_size": patch_size,
            }
            append_trial_to_csv(csv_path, rmse, hyperparams)

            del model
            torch.cuda.empty_cache()
            gc.collect()

        return rmse

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value

    results = {
        "best_params": best_params,
        "best_value": best_value,
        "n_trials": n_trials,
        "study_summary": {
            "trials_count": len(study.trials),
            "best_trial_number": study.best_trial.number if study.best_trial else None,
            "pruned_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            ),
        },
    }

    with open(os.path.join(results_dir, "optuna_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    best_config = {
        "patch_size": 50,
        "batch_size": best_params["batch_size"],
        "epochs": 20,
        "learning_rate": best_params["learning_rate"],
        "validation_split": best_params["validation_split"],
        "num_workers": None,
    }
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)

    log.info(
        f"Hyperparameter optimization complete. Best parameters: {best_params}, Best RMSE: {best_value:.4f}"
    )
    log.info("Updated config.json with best parameters")
    log.info(
        f"Pruned {results['study_summary']['pruned_trials']} trials for efficiency"
    )

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate CNN model or perform hyperparameter optimization"
    )
    parser.add_argument(
        "--data-source",
        type=str,
        required=True,
        help="Path to file list (.txt) or directory containing ZIP files",
    )
    parser.add_argument(
        "--results-dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument(
        "-n", type=int, default=None, help="Number of files to use for training"
    )
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
        default=30,
        help="Number of optimization trials (only used with --optimize). Default: 30 for time efficiency",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config file (only used without --optimize)",
    )

    args = parser.parse_args()

    if args.optimize:
        perform_hyperparameter_optimization(
            args.data_source, args.results_dir, args.n_trials, args.n
        )
    else:
        evaluate_cnn(
            args.data_source, args.results_dir, args.n, args.zip_dir, args.config
        )
