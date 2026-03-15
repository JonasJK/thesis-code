#!/bin/bash
 
#SBATCH --job-name=cnn_nir
#SBATCH --chdir=/work/klugej
#SBATCH --output=/work/%u/%x-%j.log
#SBATCH --time=1-00:00:00

#SBATCH --mem-per-cpu=64gb
#SBATCH --cpus-per-task=1
#SBATCH -G 1
#SBATCH -C "nvidia-a100|tesla-v100"

module load CUDA/12.4.0
cd /home/klugej/thesis/code/CNN || exit 1
uv sync
#uv run finetune.py
uv run cnn_nir.py -n "$1" --data-source /data/lacy-vme/khant/dop/sn-2023/ --load-model best_cnn_model.pth

