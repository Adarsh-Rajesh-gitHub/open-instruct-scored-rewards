#!/bin/bash
#SBATCH --job-name=state_memory
#SBATCH --output=/home/zsophia/orcd/scratch/open-instruct/projects/student_state_tutor/runs/state_memory/slurm_%j.out
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

python -u -m projects.student_state_tutor.memory_state_ablation \
  --traces /home/zsophia/orcd/scratch/grpo_tutor/runs/gen_traces_v2.jsonl \
  --graph-input projects/student_state_tutor/runs/graph_state_nonmath/scored.jsonl \
  --out-dir projects/student_state_tutor/runs/state_memory \
  --limit 150 \
  --generation-batch-size 24 \
  --scoring-batch-size 16
