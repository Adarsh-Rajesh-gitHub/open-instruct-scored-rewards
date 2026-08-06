#!/bin/bash
#SBATCH -J pedagogy_grpo_c
# SAME PARTITION AND CARD AS ARM A, DELIBERATELY. Arm B ran on an H200 with a bigger
# micro-batch and no gradient checkpointing, which made its wall clock incomparable to arm A's
# for reasons that had nothing to do with the reward. This arm changes the reward and nothing
# else, so it goes back on the H100 where arm A ran and keeps arm A's batch settings.
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_grpo_c_%j.out
#SBATCH -e logs/pedagogy_grpo_c_%j.err
#
# WHY THIS ARM EXISTS, AND WHY IT COPIES ARM A RATHER THAN ARM B. The blind evaluation scored
# both finished arms against one human and six calibrated raters. Arm A's reward claimed a
# gain of +0.36 over base and the two rater panels independently paid +0.34 and +0.35 - as
# close to calibrated as three separate measurements get. Arm B claimed +0.49 and was paid
# +0.37 and +0.40, about a third more than anyone bought, and it was indistinguishable from
# arm A in 17 blind head-to-head pairs (8-9). So z-scoring bought a 3x faster KL drift, a
# wider gap between claim and payment, and nothing a person could see. This arm uses arm A's
# plain average.
#
# TWO CHANGES, BOTH FROM MEASUREMENTS RATHER THAN FROM TASTE.
#
# 1. head5.npz, which adds `correct` to the four. It was dropped in the first round for
#    missing its agreement gate; the rewritten rubric reaches 0.43 on the original 600-turn
#    corpus and 0.51 on the eval pool, and the activations read it at r=0.63 against a
#    surface baseline of 0.47. It is the weakest of the five and the only one that asks
#    whether the tutor said something false - which the reward previously had nothing for.
#
# 2. A flat length band, 30-58 whitespace words, weight 2.0. The blind evaluation found the
#    one real regression of the first two arms: by the human's labels, turns rated too short
#    went from 12% to 29% and 35% while too-long went from 18% to zero. The net length score
#    looked flat only because the two errors cancel.
#
#    THE BAND IS NOT A HEURISTIC, IT IS THE FITTED ANSWER. 1131 length judgements over the
#    180 blinded turns put the 90%-acceptable range at 21-58 words, with a gentle shoulder
#    below (55% acceptable at 12 words) and a cliff above (98% at 55, 45% at 75).
#
#    THE LOWER EDGE IS THE TARGET, NOT THE LIMIT. Inside the band the term is constant, so
#    the head's pull towards brevity slides the policy to the bottom edge and rests there.
#    That is why the edge is 30 and not 21: at 30 words about 90% of turns read as the right
#    length, against 83% resting at 21. plugin.py carries the rest of the argument, including
#    why a smooth curve fitted to the same data is worse (it rests at 25 but moves to 19 or 32
#    as the head's length slope drifts, and that slope does drift during training).
#
# WHAT TO WATCH. `dim_length` is the share of sampled turns inside the band; if it does not
# climb above roughly 0.4 in the first thirty steps the weight is too low and the run should
# be stopped rather than left going. `objective/kl1_avg` should track arm A's, not arm B's -
# this arm is not rescaled, so a 3x faster drift would mean something else changed.
set -euo pipefail

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

export SCORER=pedagogy
export HEAD=${HEAD:-data/head5.npz}
export LENGTH_BAND=${LENGTH_BAND:-30-58}
export LENGTH_WEIGHT=${LENGTH_WEIGHT:-2.0}
export EXP=${EXP:-pedagogy_olmo7b_c}
export WANDB_PROJECT=${WANDB_PROJECT:-pedagogy-rm}

# Arm A's numbers exactly, so the two are comparable step for step.
export EPISODES=${EPISODES:-30720}
export SAVE_FREQ=${SAVE_FREQ:-10}

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
export SCRATCH REPO

if [ ! -f "$REPO/$HEAD" ]; then
    echo "$REPO/$HEAD does not exist. Build it with:" >&2
    echo "  python projects/pedagogy_rm/fit_head.py \\" >&2
    echo "      --dimensions leak,targeted,actionable,elicits,correct --out $HEAD" >&2
    exit 1
fi

exec bash "$REPO/projects/pedagogy_rm/scripts/train.sbatch"
