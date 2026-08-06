#!/bin/bash
#SBATCH -J pedagogy_lr
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_lr_%j.out
#SBATCH -e logs/pedagogy_lr_%j.err
#
# A SHORT RUN AT ONE LEARNING RATE, to be compared against its siblings. Thirty-five steps, no
# checkpoints, arm D's reward exactly - the only thing that varies across the set is LR.
#
# WHY MEASURE INSTEAD OF PICKING. Raising the learning rate raises the reward; that is not in
# question and it is not informative. Arm B already ran at 3.4x arm A's effective step size and
# got 3.3x the KL drift, a reward gain of +0.49 against +0.36, and a human-rated gain of +0.37
# against +0.35. It bought reward and not quality, and the reason is visible in the ratio: it
# spent 3.3 units of policy drift to buy 1.4 units of extra claimed reward.
#
# So the quantity to compare is reward gained per unit of KL drift, not reward. That ratio has a
# maximum somewhere - too small a step and the run wastes wall clock, too large and it distorts
# the policy faster than it improves it - and thirty-five steps is enough to see which side of it
# a given LR sits on.
#
# WHY THIRTY-FIVE. Arm C's `dim_length` and reward were both still climbing steeply at step 20 and
# had begun to flatten by 40, so the early slope is measurable by 35, and 35 steps is about
# eighteen minutes including the two model loads. Four of these cost the same wall clock as one
# and answer a question the single run cannot.
#
# Usage: LR=2e-5 sbatch projects/pedagogy_rm/scripts/lr_probe.sbatch
set -euo pipefail

if [ -z "${PEDAGOGY_SNAPSHOT:-}" ]; then
    snapshot="${TMPDIR:-/tmp}/lr_${SLURM_JOB_ID:-$$}.sbatch"
    cp "${BASH_SOURCE[0]}" "$snapshot"
    export PEDAGOGY_SNAPSHOT="$snapshot"
    exec bash "$snapshot"
fi

: "${LR:?set LR, e.g. LR=2e-5}"
echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}, LR=$LR"

export SCORER=pedagogy
export HEAD=${HEAD:-data/head5.npz}
export LENGTH_BAND=${LENGTH_BAND:-18-40}
export LENGTH_RAMP=${LENGTH_RAMP:-8-90}
export LENGTH_WEIGHT=${LENGTH_WEIGHT:-2.0}
export EXP=${EXP:-pedagogy_lr_$LR}
export WANDB_PROJECT=${WANDB_PROJECT:-pedagogy-rm}

# 35 steps at 256 episodes each. No checkpoints: these adapters are never sampled from, only
# their curves are read, and writing twelve of them wastes the disk and the time.
export EPISODES=${EPISODES:-8960}
export SAVE_FREQ=${SAVE_FREQ:-1000}
export CKPT_ROOT=""

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
export SCRATCH REPO LR

exec bash "$REPO/projects/pedagogy_rm/scripts/train.sbatch"
