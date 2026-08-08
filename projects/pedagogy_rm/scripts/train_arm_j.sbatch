#!/bin/bash
#SBATCH -J pedagogy_grpo_j
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_grpo_j_%j.out
#SBATCH -e logs/pedagogy_grpo_j_%j.err
#
# ARM J. Arm E with the length band moved up: 26-52 instead of 18-40, ramp 14-90 instead of 8-90.
# Same head5.npz, same 2e-5, same micro-batch, same 200 steps. One variable.
#
# (There is no arm I. That letter was used in discussion for a per-dimension standardisation arm that
# was not run, and reusing it would make the logs ambiguous.)
#
# WHY. Arm E rests at 27-28 words, which is the bottom quarter of what the human accepts: over 1131
# length judgements the 90%-acceptable range is 21-58 words. The turns read as clipped, and the
# reward is why rather than the model:
#
#   The band is FLAT inside, so within it there is no length gradient at all. The only force acting
#   on length inside the band is the probe's own pull, and that pull is downward - the five-dimension
#   reward correlates -0.60 with log word count, with four of five dimensions pushing short. So the
#   policy slides to the lower edge and the ramp below it, plus sample variance, is what holds the
#   mean up.
#
# THE OBSERVED RULE, across three arms, is that the resting mean lands about ten words above the
# band's lower edge:
#
#     arm   band      rested at
#     E     18-40     27-28 words
#     C     30-58     39-46 words
#     H     none      18-20 and still falling at step 90
#
# So the lower edge is the length dial, and 18 was set low. 26 should rest near 36 words - a step up
# from arm E without going to arm C's 46, which is where `leak` started to suffer.
#
# WHAT THIS IS AND IS NOT TESTING. It is not testing whether longer is better; that needs the
# length-variant labelling experiment, where content is held fixed across lengths. It tests whether
# the resting length follows the lower edge as the three arms above imply, and what it costs on the
# five dimensions to sit ten words higher.
#
# WHAT TO WATCH, and the number that matters. `scored/pedagogy/words` should settle near 36. Then
# compare the FIVE-DIMENSION MEAN against arm E's 1.879 held-out, at matched KL drift rather than at
# matched step - arm H showed why, having looked 0.034 better than arm E at matched step purely
# because it had drifted twice as far. If arm J's 5-dim mean lands near arm E's, length is close to
# free in this range and the band can be placed on pedagogical grounds alone. If it drops, that is
# the price of the extra ten words and it should be quoted.
set -euo pipefail

if [ -z "${PEDAGOGY_SNAPSHOT:-}" ]; then
    snapshot="${TMPDIR:-/tmp}/train_j_${SLURM_JOB_ID:-$$}.sbatch"
    cp "${BASH_SOURCE[0]}" "$snapshot"
    export PEDAGOGY_SNAPSHOT="$snapshot"
    exec bash "$snapshot"
fi

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

export SCORER=pedagogy
export HEAD=${HEAD:-data/head5.npz}

# The only change from arm E. The ramp has to bracket the band or plugin.py refuses at startup, and
# the lower ramp foot moves up with the band so the penalty below the band keeps the same slope.
export LENGTH_BAND=${LENGTH_BAND:-26-52}
export LENGTH_RAMP=${LENGTH_RAMP:-14-90}
export LENGTH_WEIGHT=${LENGTH_WEIGHT:-2.0}

export EXP=${EXP:-pedagogy_olmo7b_j}
export WANDB_PROJECT=${WANDB_PROJECT:-pedagogy-rm}

# Identical to arm E.
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

mkdir -p "${CKPT_ROOT:-$SCRATCH/ckpt}"

exec bash "$REPO/projects/pedagogy_rm/scripts/train.sbatch"
