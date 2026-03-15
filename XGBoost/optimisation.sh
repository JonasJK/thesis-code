#!/bin/bash

#SBATCH --job-name=nirXGB_opt
#SBATCH --chdir=/work/klugej
#SBATCH --output=/work/%u/%x-%A-%a.log
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=96gb
#SBATCH --cpus-per-task=1
#SBATCH -G 1
#SBATCH --constraint=a100-vram-80G

cd /home/klugej/thesis/code/XGBoost || exit 1
module load CUDA/12.4.0
uv sync
uv run evaluate_xgboost.py --data-source /data/lacy-vme/khant/dop/sn-2023/ --results-dir /home/klugej/thesis/results --zip-dir --optimize --n-trials 50 -n "$1"
