#!/bin/bash
#SBATCH --job-name=oracle_state
#SBATCH --output=/home/zsophia/orcd/scratch/open-instruct/projects/student_state_tutor/runs/oracle_nonmath/slurm_%j.out
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
source /home/zsophia/orcd/scratch/grpo_tutor/scripts/env.sh
export HF_HUB_OFFLINE=1

python -u /home/zsophia/orcd/scratch/open-instruct/projects/student_state_tutor/oracle_state_ablation.py \
  --traces /home/zsophia/orcd/scratch/grpo_tutor/runs/gen_traces_v2.jsonl \
  --out-dir /home/zsophia/orcd/scratch/open-instruct/projects/student_state_tutor/runs/oracle_nonmath \
  --limit 180 \
  --generation-batch-size 24 \
  --scoring-batch-size 128
