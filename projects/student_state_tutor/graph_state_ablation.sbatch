#!/bin/bash
#SBATCH --job-name=graph_state
#SBATCH --output=/home/zsophia/orcd/scratch/open-instruct/projects/student_state_tutor/runs/graph_state_nonmath/slurm_%j.out
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
cd /home/zsophia/orcd/scratch/open-instruct

python -u -m projects.student_state_tutor.graph_state_ablation \
  --input projects/student_state_tutor/runs/oracle_nonmath/scored.jsonl \
  --out-dir projects/student_state_tutor/runs/graph_state_nonmath \
  --generation-batch-size 24 \
  --scoring-batch-size 128
