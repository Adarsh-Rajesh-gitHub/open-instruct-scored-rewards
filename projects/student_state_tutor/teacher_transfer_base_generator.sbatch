#!/bin/bash
#SBATCH --job-name=teacher_transfer_base
#SBATCH --output=/home/zsophia/orcd/scratch/open-instruct/projects/student_state_tutor/runs/teacher_transfer_base/slurm_%j.out
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --open-mode=append

set -euo pipefail
source /home/zsophia/orcd/scratch/grpo_tutor/scripts/env.sh
export HF_HUB_OFFLINE=1
cd /home/zsophia/orcd/scratch/open-instruct

mkdir -p projects/student_state_tutor/runs/teacher_transfer_base
python -u -m projects.student_state_tutor.teacher_transfer_ablation \
  --banks projects/student_state_tutor/runs/estela/estela_banks.jsonl \
  --action-rows projects/student_state_tutor/runs/teacher_action_sft_v2_eval/decisions.jsonl \
  --out-dir projects/student_state_tutor/runs/teacher_transfer_base \
  --limit 32 \
  --generation-batch-size 16 \
  --scoring-batch-size 64
