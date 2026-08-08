#!/bin/bash
#SBATCH -J rlvr_math
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/rlvr_math_%j.out
#SBATCH -e logs/rlvr_math_%j.err
#
# MATHS. Ai2's own RLVR mixes, verified by `math` and `gsm8k`, both already in open-instruct.
#
# RLVR-GSM-MATH-IF-Mixed-Constraints is the mix the tulu3 8B RLVR run used, so it is the one with a
# published result to compare against; it carries gsm8k, math and ifeval rows and each row names its
# own verifier, so one mix exercises three of them. Set TRAIN_MIX to a single-domain set instead if
# you want the per-verifier signal unmixed.
set -euo pipefail
export EXP=${EXP:-rlvr_math}
export TRAIN_MIX=${TRAIN_MIX:-"allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0"}
export EVAL_MIX=${EVAL_MIX:-"allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16"}
export WANDB_PROJECT=${WANDB_PROJECT:-rlvr-verifiable}
export INNER=projects/rlvr_verifiable/scripts/train.sh
exec bash "${REPO:-/orcd/scratch/orcd/013/zsophia/open-instruct}/projects/pedagogy_rm/scripts/train.sbatch"
