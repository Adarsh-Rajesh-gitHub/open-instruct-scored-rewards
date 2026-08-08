#!/bin/bash
#SBATCH --job-name=tv_sweep_plots
#SBATCH --output=/home/zsophia/orcd/scratch/Engram/runs/task_vector_sweep/logs/plots_%j.out
#SBATCH --partition=mit_quicktest
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail
ROOT=/home/zsophia/orcd/scratch/Engram
cd "$ROOT"
export PYTHONPATH="$ROOT"
source "$HOME/venv_test/bin/activate"

python -u scripts/plot_task_vector_sweep.py \
  --input-dir runs/task_vector_sweep \
  --output-dir runs/task_vector_sweep/plots
