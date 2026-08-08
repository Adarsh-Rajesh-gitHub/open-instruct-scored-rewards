#!/bin/bash
#SBATCH --job-name=teacher_transfer_sft
#SBATCH --output=/home/zsophia/orcd/scratch/open-instruct/projects/student_state_tutor/runs/teacher_transfer_sft/slurm_%j.out
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

mkdir -p projects/student_state_tutor/runs/teacher_transfer_sft
python -u -m projects.student_state_tutor.teacher_transfer_ablation \
  --banks projects/student_state_tutor/runs/estela/estela_banks.jsonl \
  --action-rows projects/student_state_tutor/runs/teacher_action_sft_v2_eval/decisions.jsonl \
  --out-dir projects/student_state_tutor/runs/teacher_transfer_sft \
  --teacher-adapter projects/student_state_tutor/runs/teacher_action_sft_v2/adapter \
  --limit 32 \
  --generation-batch-size 16 \
  --scoring-batch-size 64
