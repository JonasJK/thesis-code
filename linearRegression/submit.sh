#!/bin/bash
 
#SBATCH --job-name=SGDReg
#SBATCH --chdir=/work/klugej
#SBATCH --output=/work/%u/%x-%j.log
#SBATCH --time=3-00:00:00
#SBATCH -G 1
#SBATCH -C "nvidia-a100|tesla-v100"
#SBATCH --mem-per-cpu=384G

cd /home/klugej/thesis/code/linearRegression || exit 1
uv sync
uv run python linearRegression.py "$@"
