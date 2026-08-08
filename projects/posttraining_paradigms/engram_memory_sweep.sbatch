#!/bin/bash
#SBATCH --job-name=tv_engram_sweep
#SBATCH --output=/home/zsophia/orcd/scratch/Engram/runs/task_vector_sweep/logs/engram_%A_%a.out
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-4%5
#SBATCH --time=00:20:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail
ROOT=/home/zsophia/orcd/scratch/Engram
cd "$ROOT"
export PYTHONPATH="$ROOT"
source "$HOME/venv_test/bin/activate"

STEPS=(1220 3660 6100 8540 10980)
SOURCE_STEP="${STEPS[$SLURM_ARRAY_TASK_ID]}"
SOURCE="$(printf 'runs/climbmix_engram/snapshots/step%07d.pt' "$SOURCE_STEP")"

python -u scripts/task_vector_sweep.py \
  --arm engram \
  --source "$SOURCE" \
  --target runs/climbmix_engram/snapshots/step0012200.pt \
  --climbmix-val data/climbmix/val.bin \
  --benchmark-file data/transfer_benchmarks.pt \
  --output "runs/task_vector_sweep/engram_step${SOURCE_STEP}.json" \
  --steps 160 \
  --learning-rate 1e-5 \
  --repair-learning-rate 1e-6 \
  --batch-size 8 \
  --log-every 20
