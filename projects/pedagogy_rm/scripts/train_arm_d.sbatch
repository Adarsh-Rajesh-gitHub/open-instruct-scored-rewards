#!/bin/bash
#SBATCH -J pedagogy_grpo_d
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_grpo_d_%j.out
#SBATCH -e logs/pedagogy_grpo_d_%j.err
#
# ARM D. One change from arm C: the length term is a ramp rather than a hard band, and the band
# has moved down. Everything else - the head, the scorer, the learning rate, the step count - is
# arm C's, so the comparison is readable.
#
# WHY THE BAND MOVED DOWN, 30-58 TO 18-40. Arm C landed at a median of 46 words and the human's
# ratings under the new rubric put 40-60 words at 1.81 against 2.00-2.11 below 40, and 60+ at
# 0.67. Her combined score correlates -0.59 with log length. Arm C overshot.
#
# WHY A RAMP AND NOT A BAND. A hard band cannot tell 59 words from 200 - both score zero - so
# nothing pushes a runaway turn back towards the band, and arm C's longest turns were unpunished
# relative to its merely-long ones. The ramp makes 46 words cost 0.12 and 75 words cost 0.70,
# where the band charged both of them nothing. The interior stays flat, because that is what pins
# the resting length: a smooth peak drifts with the quality slope, a flat band does not.
#
# WHY THIS TERM IS CARRYING MORE WEIGHT THAN IT LOOKS. The head's `leak` tracks length at +0.70
# on policy-generated turns against the human's +0.43, so it pulls towards brevity harder than
# she does. Two attempts to fix that inside the head both failed: a rank-1 correction moved it to
# 0.67 and residualising length out before fitting moved it to 0.74, because both were estimated
# in-distribution and the over-coupling is distribution shift. Until there are labels on
# policy-generated turns, this ramp is the only lever on length that behaves predictably.
#
# WHAT WAS NOT CHANGED, DELIBERATELY. The learning rate is arm A's. A larger one was considered
# and rejected on evidence: arm B ran at 3.4x the effective step size, drifted 3.3x as far in KL,
# claimed +0.49 against arm A's +0.36, and was paid +0.37 by humans against arm A's +0.35. It
# bought reward, not quality. With the head still miscalibrated on length, a run tuned to maximise
# this reward would be pushing hardest on the known defect.
#
# WHAT TO WATCH. `dim_length` is the mean ramp credit, so it should climb towards 1.0 rather than
# towards 0.75 as arm C's in-band share did; if it stalls below 0.6 by step 40 the weight is low.
# `sequence_lengths` should settle around 22-28 tokens rather than arm C's 48-51. And
# `objective/kl1_avg` should track arm A and arm C at roughly 0.20, not arm B's 0.63.
set -euo pipefail

if [ -z "${PEDAGOGY_SNAPSHOT:-}" ]; then
    snapshot="${TMPDIR:-/tmp}/train_d_${SLURM_JOB_ID:-$$}.sbatch"
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
export EXP=${EXP:-pedagogy_olmo7b_d}
export WANDB_PROJECT=${WANDB_PROJECT:-pedagogy-rm}

export EPISODES=${EPISODES:-30720}
export SAVE_FREQ=${SAVE_FREQ:-10}

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
export SCRATCH REPO

if [ ! -f "$REPO/$HEAD" ]; then
    echo "$REPO/$HEAD does not exist. Build it with:" >&2
    echo "  python projects/pedagogy_rm/fit_head.py --decouple '' \\" >&2
    echo "      --dimensions leak,targeted,actionable,elicits,correct --out $HEAD" >&2
    exit 1
fi

exec bash "$REPO/projects/pedagogy_rm/scripts/train.sbatch"
