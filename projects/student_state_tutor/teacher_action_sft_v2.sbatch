#!/bin/bash
#SBATCH --job-name=teacher_action_sft2
#SBATCH --output=/home/zsophia/orcd/scratch/open-instruct/projects/student_state_tutor/runs/teacher_action_sft_v2/slurm_%j.out
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=2:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
#SBATCH --open-mode=append

set -euo pipefail
source /home/zsophia/orcd/scratch/grpo_tutor/scripts/env.sh
export HF_HUB_OFFLINE=1
cd /home/zsophia/orcd/scratch/open-instruct

mkdir -p \
  projects/student_state_tutor/runs/teacher_action_sft_v2 \
  projects/student_state_tutor/runs/teacher_action_sft_v2_eval

python -u -m projects.student_state_tutor.teacher_action_sft \
  --banks projects/student_state_tutor/runs/estela/estela_banks.jsonl \
  --out-dir projects/student_state_tutor/runs/teacher_action_sft_v2 \
  --examples-per-action 96 \
  --epochs 2 \
  --batch-size 4 \
  --gradient-accumulation 4

python -u -m projects.student_state_tutor.teacher_action_ablation \
  --banks projects/student_state_tutor/runs/estela/estela_banks.jsonl \
  --out-dir projects/student_state_tutor/runs/teacher_action_sft_v2_eval \
  --teacher-adapter projects/student_state_tutor/runs/teacher_action_sft_v2/adapter \
  --replicates 8 \
  --batch-size 16
