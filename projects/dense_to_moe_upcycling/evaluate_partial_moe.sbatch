#!/bin/bash
#SBATCH --job-name=teacher_moe_eval
#SBATCH --output=/home/zsophia/orcd/scratch/open-instruct/projects/dense_to_moe_upcycling/runs/partial_moe/eval_%j.out
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=0:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
source /home/zsophia/orcd/scratch/grpo_tutor/scripts/env.sh
export HF_HUB_OFFLINE=1
cd /home/zsophia/orcd/scratch/open-instruct

python -u -m projects.dense_to_moe_upcycling.evaluate_checkpoint \
  --checkpoint projects/dense_to_moe_upcycling/runs/partial_moe/moe_checkpoint.pt \
  --out projects/dense_to_moe_upcycling/runs/partial_moe/generation.json
