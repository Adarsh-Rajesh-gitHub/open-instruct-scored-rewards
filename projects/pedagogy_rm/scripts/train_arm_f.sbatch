#!/bin/bash
#SBATCH -J pedagogy_grpo_f
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_grpo_f_%j.out
#SBATCH -e logs/pedagogy_grpo_f_%j.err
#
# ARM F. Arm E at twice the learning rate, and nothing else different. 4e-5 against 2e-5, same
# head, same length ramp, same 200 steps, same seed.
#
# WHY THIS IS THE RUN WORTH SPENDING A SECOND GPU ON. The question "does a bigger step buy a
# better tutor or only a bigger number" has been asked twice and answered once, badly. Arm B was
# the first attempt and it was confounded: z-scoring the reward raised the effective step size
# 3.4x as a side effect, so its extra reward-hacking could have come from the scalarisation or from
# the step size and there was no way to tell. This pair separates them - identical reward, one knob.
#
# The 35-step sweep says 2e-5 is 1.6x more efficient per unit of KL drift, but efficiency over 35
# steps does not settle where 200 steps ends up. Two things could happen and they look the same in
# the reward curve: 4e-5 could reach a genuinely better policy sooner, or it could reach a higher
# claimed reward while the gap between claimed and paid widens, which is what arm B did (+0.49
# claimed against +0.37 paid, and a coin flip against arm A in blind pairs).
#
# THE READOUT IS NOT THE REWARD CURVE. Whichever run ends higher on `pedagogy/reward` tells us
# nothing on its own - a larger step size producing more reward is the premise, not the finding.
# The comparison is the blind eval afterwards: sample both at step 200, mix them into one pool with
# the arm stripped, and see which turns a person prefers. If 4e-5 wins on reward and loses on
# preference, that is arm B again with the confound removed, and it settles the question.
#
# WHAT TO WATCH MEANWHILE. `objective/kl1_avg` should land near twice arm E's. If it lands near
# arm B's 0.63 while reward climbs, the run has left the range the head was fitted on - 1.60 was
# the 78th percentile of the labelled corpus and above about 1.8 the head is extrapolating.
set -euo pipefail

if [ -z "${PEDAGOGY_SNAPSHOT:-}" ]; then
    snapshot="${TMPDIR:-/tmp}/train_f_${SLURM_JOB_ID:-$$}.sbatch"
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
export EXP=${EXP:-pedagogy_olmo7b_f}
export WANDB_PROJECT=${WANDB_PROJECT:-pedagogy-rm}

# The one difference from arm E.
export LR=${LR:-4e-5}

export MICRO_BATCH=${MICRO_BATCH:-4}
export EPISODES=${EPISODES:-51200}
export SAVE_FREQ=${SAVE_FREQ:-10}

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
export SCRATCH REPO

exec bash "$REPO/projects/pedagogy_rm/scripts/train.sbatch"
