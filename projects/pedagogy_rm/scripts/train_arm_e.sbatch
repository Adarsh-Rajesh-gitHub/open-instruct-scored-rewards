#!/bin/bash
#SBATCH -J pedagogy_grpo_e
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_grpo_e_%j.out
#SBATCH -e logs/pedagogy_grpo_e_%j.err
#
# ARM E. Arm D's reward, at the learning rate a sweep measured rather than the one arm A
# inherited, with the batch shape that makes a step 1.4x cheaper, over 200 steps instead of 120.
#
# THE LEARNING RATE WAS MEASURED, NOT CHOSEN. Four 35-step runs, identical apart from the rate,
# compared on reward gained per unit of KL drift - because reward alone ranks runs by learning
# rate and says nothing:
#
#     rate     reward gain   KL      gain/KL   tokens    in band
#     2e-5       +0.526     0.074     7.11     50->40      0.90
#     1e-5       +0.348     0.080     4.35     50->31      0.81
#     4e-5       +0.526     0.122     4.31     50->43      0.92
#     8e-5       +0.606     0.172     3.52     49->36      0.94
#
# 2e-5 buys the same gain as 4e-5 for 40% less drift and nearly as much as 8e-5 for 43% of the
# drift. Above it the pattern is arm B's - more reward at a worse exchange rate, which is what
# produced arm B's +0.49 claimed against +0.37 paid.
#
# THE TOKEN COLUMN IS THE MORE INTERESTING RESULT. The slowest rate produced the SHORTEST turns
# (31 words at 1e-5 against 40 at 2e-5), and in-band share rose monotonically with the rate. At
# 1e-5 the head's pull towards brevity still dominates and the policy sits at the band's lower
# edge; from 2e-5 up it reaches the band and stays. So the rate that is efficient is also the one
# that lands where the rubric says it should.
#
# MICRO-BATCH 4 WITH CHECKPOINTING KEPT. The run is training-bound, not generation-bound: 28.3s
# of each 29.3s step is the trainer and vLLM idles 4.9s waiting for it. Packing puts ~10 sequences
# in each 2048-token buffer, so a step is ~26 buffers, and at micro-batch 1 that is 26 sequential
# forward-backward passes. Four at a time makes it seven. Arm B measured 19.9s against 28.3s for
# this change, though it also turned checkpointing off on an H200's 141GB; keeping checkpointing on
# holds activations small enough for an 80GB card sharing with vLLM, and takes most of the gain.
#
# 200 STEPS, WHICH IS ONLY DEFENSIBLE NOW. Buying more steps at an inefficient step size just
# reproduces arm B further along. At the efficient rate it is the cheapest remaining way to raise
# the reward, and the 1.4x batch saving pays for two thirds of the extra time.
#
# WHAT TO WATCH. `sequence_lengths` should hold near 40 words rather than falling; `dim_length`
# should climb past 0.90; `objective/kl1_avg` should land near 0.20-0.25, not arm B's 0.63. If
# reward is still climbing at 200 steps that is worth knowing, but the checkpoints every 10 steps
# are there so quality can be checked against reward rather than assumed to track it.
set -euo pipefail

if [ -z "${PEDAGOGY_SNAPSHOT:-}" ]; then
    snapshot="${TMPDIR:-/tmp}/train_e_${SLURM_JOB_ID:-$$}.sbatch"
    cp "${BASH_SOURCE[0]}" "$snapshot"
    export PEDAGOGY_SNAPSHOT="$snapshot"
    exec bash "$snapshot"
fi

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

export SCORER=pedagogy
export HEAD=${HEAD:-data/head5.npz}
export LENGTH_BAND=${LENGTH_BAND:-18-40}
export LENGTH_RAMP=${LENGTH_RAMP:-8-90}
export LENGTH_WEIGHT=${LENGTH_WEIGHT:-2.0}
export EXP=${EXP:-pedagogy_olmo7b_e}
export WANDB_PROJECT=${WANDB_PROJECT:-pedagogy-rm}

export LR=${LR:-2e-5}
export MICRO_BATCH=${MICRO_BATCH:-4}
export EPISODES=${EPISODES:-51200}
export SAVE_FREQ=${SAVE_FREQ:-10}

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
export SCRATCH REPO

if [ ! -f "$REPO/$HEAD" ]; then
    echo "$REPO/$HEAD does not exist." >&2
    exit 1
fi

exec bash "$REPO/projects/pedagogy_rm/scripts/train.sbatch"
