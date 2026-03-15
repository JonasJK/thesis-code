#!/bin/bash

#SBATCH --job-name=nirRF_opt
#SBATCH --chdir=/work/klugej
#SBATCH --output=/work/%u/%x-%j.log
#SBATCH --time=3-00:00:00
#SBATCH --mail-type ALL 
#SBATCH --mail-user=jonas.kluge@ufz.de
#SBATCH --mem-per-cpu=256gb
#SBATCH --cpus-per-task=1
#SBATCH -G 1
#SBATCH -C "nvidia-a100|tesla-v100"

cd /home/klugej/thesis/code/randomForest || exit 1
module load CUDA/12.4.0
uv sync
uv run evaluate_random_forest.py --data-source /data/lacy-vme/khant/dop/sn-2023/ --results-dir /home/klugej/thesis/results --zip-dir --optimize --n-trials 50 -n "$1"
