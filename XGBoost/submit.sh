#!/bin/bash
 
#SBATCH --job-name=xgbRF
#SBATCH --chdir=/work/klugej
#SBATCH --output=/work/%u/%x-%j.log
#SBATCH --time=3-00:00:00

#SBATCH --mem-per-cpu=96gb
#SBATCH --cpus-per-task=1
#SBATCH -G 1
#SBATCH --constraint=a100-vram-80G
module load CUDA/12.4.0
cd /home/klugej/thesis/code/XGBoost || exit 1
uv sync
uv run python -c "import cupy; print('CuPy available:', cupy.cuda.is_available())"
uv run xgboost_nir.py -n "$1" --data-source /data/lacy-vme/khant/dop/sn-2023/ --zip-dir --load-model /home/klugej/thesis/code/XGBoost/xgboost_model_4500files.pkl
