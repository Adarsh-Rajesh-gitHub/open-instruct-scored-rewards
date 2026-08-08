#!/bin/bash
#SBATCH --job-name=teacher_action_sft
#SBATCH --output=/home/zsophia/orcd/scratch/open-instruct/projects/student_state_tutor/runs/teacher_action_sft/slurm_%j.out
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
source /home/zsophia/orcd/scratch/grpo_tutor/scripts/env.sh
export HF_HUB_OFFLINE=1
cd /home/zsophia/orcd/scratch/open-instruct

mkdir -p \
  projects/student_state_tutor/runs/teacher_action_sft \
  projects/student_state_tutor/runs/teacher_action_sft_eval

python -u -m projects.student_state_tutor.teacher_action_sft \
  --banks projects/student_state_tutor/runs/estela/estela_banks.jsonl \
  --out-dir projects/student_state_tutor/runs/teacher_action_sft \
  --examples-per-action 96 \
  --epochs 3 \
  --batch-size 4 \
  --gradient-accumulation 4

python -u -m projects.student_state_tutor.teacher_action_ablation \
  --banks projects/student_state_tutor/runs/estela/estela_banks.jsonl \
  --out-dir projects/student_state_tutor/runs/teacher_action_sft_eval \
  --teacher-adapter projects/student_state_tutor/runs/teacher_action_sft/adapter \
  --replicates 8 \
  --batch-size 16
