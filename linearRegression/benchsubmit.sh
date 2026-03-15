#!/bin/bash
 
#SBATCH --job-name=bench
#SBATCH --chdir=/work/klugej
#SBATCH --output=/work/%u/%x-%j.log
#SBATCH --time=1-00:30:00

#SBATCH --mem-per-cpu=5G

cd /home/klugej/thesis/code/linearRegression || exit 1

uv sync

uv run file_bench.py