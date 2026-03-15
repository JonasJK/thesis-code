#!/bin/bash
 
#SBATCH --job-name=randomF
#SBATCH --chdir=/work/klugej
#SBATCH --output=/work/%u/%x-%j.log
#SBATCH --time=3-00:00:00

#SBATCH --mem-per-cpu=96gb
#SBATCH --cpus-per-task=1
#SBATCH -G nvidia-a100:1
#SBATCH --constraint a100-vram-80G

module load CUDA/12.4.0
cd /home/klugej/thesis/code/randomForest || exit 1
uv sync
uv run randomForest.py -n "$1" --data-source /data/lacy-vme/khant/dop/sn-2023/ --zip-dir  --load-model /home/klugej/thesis/code/randomForest/random_forest_model_4500files.pkl
