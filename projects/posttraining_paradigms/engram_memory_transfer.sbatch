oh#!/bin/bash
#SBATCH --job-name=tv_engram
#SBATCH --output=/home/zsophia/orcd/scratch/Engram/runs/task_vector_transfer/engram_%j.out
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:20:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail
ROOT=/home/zsophia/orcd/scratch/Engram
cd "$ROOT"
export PYTHONPATH="$ROOT"
source "$HOME/venv_test/bin/activate"

python -u scripts/task_vector_transfer.py \
  --arm engram \
  --source runs/climbmix_engram/snapshots/step0003660.pt \
  --target runs/climbmix_engram/snapshots/step0012200.pt \
  --val-bin data/climbmix/val.bin \
  --output runs/task_vector_transfer/engram_results_160.json \
  --steps 160 \
  --learning-rate 1e-5 \
  --batch-size 8 \
  --alpha 0.25 \
  --alpha 0.5 \
  --alpha 1.0 \
  --alpha 1.5 \
  --alpha 2.0 \
  --log-every 10
