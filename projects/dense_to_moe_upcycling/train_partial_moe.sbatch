#!/bin/bash
#SBATCH --job-name=teacher_moe_smoke
#SBATCH --output=/home/zsophia/orcd/scratch/open-instruct/projects/dense_to_moe_upcycling/runs/partial_moe/slurm_%j.out
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
source /home/zsophia/orcd/scratch/grpo_tutor/scripts/env.sh
export HF_HUB_OFFLINE=1
cd /home/zsophia/orcd/scratch/open-instruct

mkdir -p projects/dense_to_moe_upcycling/runs/partial_moe
python -u -m projects.dense_to_moe_upcycling.train_cpt \
  --data projects/dense_to_moe_upcycling/runs/data/climbmix_1m.jsonl \
  --out-dir projects/dense_to_moe_upcycling/runs/partial_moe \
  --layers last2 \
  --num-experts 4 \
  --top-k 2 \
  --sequence-length 256 \
  --max-blocks 2048 \
  --eval-blocks 16 \
  --steps 50 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --expert-lr 2e-6 \
  --router-lr 3e-4 \
  --load-balance-coef 0.1
