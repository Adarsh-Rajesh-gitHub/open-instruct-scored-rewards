#!/bin/bash
#SBATCH -J rlvr_factual
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/rlvr_factual_%j.out
#SBATCH -e logs/rlvr_factual_%j.err
#
# FACTUAL QA, verified by `string_f1` - token F1 against the reference answer.
#
# WHY F1 AND NOT EXACT MATCH. `string_matcher` is also available and is stricter, but short-answer QA
# has too many equally correct surface forms for exact match to be a usable gradient: "Paris" against
# "Paris, France" scores 0 under matching and 0.67 under F1. F1 is the softer signal and the one the
# repo's own QA conversions target.
#
# THIS IS THE DOMAIN WITH THE WEAKEST VERIFIER, and it is worth saying so before reading results.
# Token overlap rewards saying more of the right words, which is gameable by padding an answer with
# plausible tokens. If reward climbs here while answers get longer, suspect that before believing it.
set -euo pipefail
export EXP=${EXP:-rlvr_factual}
export TRAIN_MIX=${TRAIN_MIX:?"set TRAIN_MIX to a string_f1 dataset; see README for candidates"}
export EVAL_MIX=${EVAL_MIX:-$TRAIN_MIX}
export WANDB_PROJECT=${WANDB_PROJECT:-rlvr-verifiable}
export INNER=projects/rlvr_verifiable/scripts/train.sh
exec bash "${REPO:-/orcd/scratch/orcd/013/zsophia/open-instruct}/projects/pedagogy_rm/scripts/train.sbatch"
