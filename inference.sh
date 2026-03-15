#!/bin/bash
 
#SBATCH --job-name=inference
#SBATCH --chdir=/work/klugej
#SBATCH --output=/work/%u/%x-%j.log
#SBATCH --time=03:00:00
#SBATCH -G 1
#SBATCH -C "nvidia-a100|tesla-v100"
#SBATCH --mem-per-cpu=32G

module load CUDA/12.4.0
cd /home/klugej/thesis/code/|| exit 1
uv sync
for MODEL in \
  "linearRegression/linear_regression_model_1000files.pkl" \
  "randomForest/random_forest_model_4500files.pkl" \
  "XGBoost/xgboost_model_4500files.pkl" \
  "CNN/best_cnn_model.pth"
do
  for D in 112 211 312 311; do
    uv run python inference.py --model "$MODEL" --input "/work/klugej/clc/$D/${D}.tif"
  done
done
