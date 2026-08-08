#!/bin/bash
#SBATCH -J pedagogy_grpo_h
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_grpo_h_%j.out
#SBATCH -e logs/pedagogy_grpo_h_%j.err
#
# ARM H. Arm E with the length term removed and nothing else changed: same head5.npz, same 2e-5,
# same micro-batch, same 200 steps. One variable.
#
# WHY THIS RUN EXISTS. Decomposing arm E's reward showed the two halves of the objective started
# with almost exactly the same room to improve - 0.578 for the five probe dimensions against 0.604
# for the length term - and collected wildly different fractions of it:
#
#     component        start     end    gained   best   headroom   captured
#     5-dim mean       1.622   1.852    +0.230   2.20     0.578       40%
#     length x 2       1.396   1.981    +0.585   2.00     0.604       97%
#
# That rules out the obvious explanation. The probe dimensions did not stall because they were
# already near their ceiling; they had as much left as length did and took less than half as much
# of it. So something else stopped them, and the two candidates have different consequences:
#
#   (a) The length band was BLOCKING them. The V1 rubric's degenerate optimum is a short bare
#       question, which satisfies four of five dimensions at once - that is the 12-word collapse
#       arm A found and the reason the rubric was rebuilt. If the band was the only thing keeping
#       the policy off that path, then removing it should let the 5-dim mean climb PAST 1.852.
#
#   (b) The probe's remaining headroom is genuinely hard to reach. Then removing the band gets a
#       shorter policy and no more reward on the dimensions, and the paper's ceiling claim survives
#       - relocated from "what the probe can express" to "what the policy can find".
#
# THE PREDICTION, WRITTEN DOWN BEFORE THE RUN so the outcome cannot be rationalised afterwards:
# (a) is expected. Arm A, the only other run with no length term, went 38 -> 12 words. Arm H should
# collapse similarly, and its 5-dim mean should END ABOVE arm E's 1.852. If it ends at or below
# 1.852 the prediction is wrong and (b) holds, which is the more interesting result of the two
# because it says the reward model is not merely being fought by its own length patch.
#
# THE CONFOUND TO WATCH, and the reason this is not a perfectly clean ablation. GRPO's advantage is
# reward minus the group mean, so what sets the effective step size is the SPREAD of reward within a
# group, not its level. Early in training the length term varies a lot across the 8 samples of a
# prompt, so dropping it removes within-group variance and shrinks the effective step. Arm B is the
# same effect with the opposite sign: z-scoring shrank the reward scale, raised within-group spread,
# and drifted 3.6x as far as arm A at the same nominal rate.
#
# So if arm H's KL drift lands well below arm E's 0.25, it was trained more gently and the
# comparison of captured headroom is not apples-to-apples. Check `objective/kl1_avg` before reading
# anything into the dimension numbers. If drift is much lower, the follow-up is arm H at a higher
# rate matched on drift rather than on nominal learning rate.
#
# WHAT TO WATCH. `scored/pedagogy/words` should fall well below arm E's 28 and keep falling. That
# metric used to be written only when the length term was active, so this arm - the one where a
# collapse in length is the expected outcome - was the one that would not have recorded it; plugin.py
# now logs it unconditionally. `dim_length` is absent here by design, since it is a component of a
# reward this arm does not have. The five dimension curves are the point of the run.
set -euo pipefail

if [ -z "${PEDAGOGY_SNAPSHOT:-}" ]; then
    snapshot="${TMPDIR:-/tmp}/train_h_${SLURM_JOB_ID:-$$}.sbatch"
    cp "${BASH_SOURCE[0]}" "$snapshot"
    export PEDAGOGY_SNAPSHOT="$snapshot"
    exec bash "$snapshot"
fi

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

export SCORER=pedagogy
export HEAD=${HEAD:-data/head5.npz}

# THE WHOLE POINT OF THE ARM. Set explicitly rather than by omission, and NOT with `:-`, because
# arm E writes `${LENGTH_BAND:-18-40}` and `:-` substitutes on empty as well as unset - so passing
# LENGTH_BAND="" to arm E's script silently gets the band back. train.sh only appends the length
# arguments to the scorer spec when the band is non-empty, so an empty band removes the term
# entirely rather than giving it zero weight.
export LENGTH_BAND=""
export LENGTH_RAMP=""
export LENGTH_WEIGHT=0

export EXP=${EXP:-pedagogy_olmo7b_h}
export WANDB_PROJECT=${WANDB_PROJECT:-pedagogy-rm}

# Identical to arm E in every other respect.
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

# The checkpoint root was deleted in a cleanup after the last round, and a preemptable partition
# needs it to exist for the requeue to resume rather than restart.
mkdir -p "${CKPT_ROOT:-$SCRATCH/ckpt}"

exec bash "$REPO/projects/pedagogy_rm/scripts/train.sbatch"
