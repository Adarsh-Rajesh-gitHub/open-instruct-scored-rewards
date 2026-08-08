#!/bin/bash
#SBATCH -J refit_qwen
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/refit_qwen_%j.out
#SBATCH -e logs/refit_qwen_%j.err
#
# Refit the reward head on Qwen2.5-7B-Instruct, which the encoder sweep picked.
#
# WHAT THE SWEEP FOUND. Best-cell r on the same 600 turns, same labels, question-grouped folds:
#
#     dimension    olmo7b  qwen3b  qwen7b  qwen14b  qwen32b   ceiling
#     leak           0.84    0.87    0.87     0.87     0.87      0.95
#     targeted       0.85    0.83    0.86     0.86     0.86      0.94
#     actionable     0.94    0.94    0.94     0.94     0.95      0.98
#     elicits        0.95    0.95    0.95     0.95     0.96      0.98
#
# Family beats size. Qwen at 3B already reads `leak` better than OLMo at 7B, and 7B -> 14B -> 32B
# adds nothing. Qwen2.5-7B is therefore the pick: same footprint as the encoder it replaces, all of
# the available gain, and nothing above it is worth the memory.
#
# THREE STEPS, BECAUSE TWO INPUTS DO NOT TRANSFER FROM THE OLMO FIT.
#
#   1. `correct` has no measured cell. The sweep ran the default DIMENSIONS - leak, targeted,
#      actionable, elicits, concise - but the deployed head scores `correct` and not `concise`. Its
#      best pooling and layer on Qwen are unknown and are measured here.
#
#   2. The adversarial states are OLMo's. fit_head hardens `leak` against answer-injection using
#      hidden states of corrupted turns, and those were extracted with OLMo. Reusing them would fit
#      a Qwen head on OLMo activations, which is not a subtle error but it would run and produce a
#      plausible-looking head. They are re-extracted here.
#
#   3. The layer indices differ. OLMo reads `leak` best at last/16; Qwen reads it at eot/21. The
#      --cells flag exists so the table is passed in rather than edited into fit_head.py.
#
# --nested IS ON for the cell measurement. The plain sweep reports the largest of 21 pooling x layer
# cells, each a noisy cross-validated estimate, so its maximum is optimistic. Nested selection picks
# the cell inside each outer fold's training questions and scores it on the held-out fold, which
# costs a little accuracy on paper and buys a number that transfers.
set -euo pipefail

if [ -z "${PEDAGOGY_SNAPSHOT:-}" ]; then
    snapshot="${TMPDIR:-/tmp}/refit_${SLURM_JOB_ID:-$$}.sbatch"
    cp "${BASH_SOURCE[0]}" "$snapshot"
    export PEDAGOGY_SNAPSHOT="$snapshot"
    exec bash "$snapshot"
fi

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$SCRATCH/venv_oi}
ENCODER=${ENCODER:-Qwen/Qwen2.5-7B-Instruct}
TAG=${TAG:-qwen7b}

export HF_HOME=$SCRATCH/hf-cache HF_HUB_CACHE=$SCRATCH/hf-cache/hub
export XDG_CACHE_HOME=$SCRATCH/xdg-cache TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor-cache

set +u
command -v module >/dev/null 2>&1 || source /usr/share/lmod/lmod/init/bash
module load cuda/13.0.1
source "$VENV/bin/activate"
set -u
cd "$REPO"; export PYTHONPATH=$REPO; mkdir -p logs data/encoders

echo ""
echo "=== 1. best cell per dimension on $TAG, including correct, nested ==="
python -u -m projects.pedagogy_rm.probe \
    --hidden "data/encoders/${TAG}.npz" --labels data/labels/*.json \
    --dimensions leak,targeted,actionable,elicits,correct \
    --skip-mlp --ceilings --nested --verbose \
    | tee "logs/cells_${TAG}.txt"

echo ""
echo "=== 2. adversarial states for the leak hardening, re-extracted with $ENCODER ==="
if [ -f "data/encoders/hack_${TAG}.npz" ]; then
    echo "  already present, skipping"
else
    python -u -m projects.pedagogy_rm.extract_hidden \
        --units data/hack_units.json --out "data/encoders/hack_${TAG}.npz" --model "$ENCODER"
fi

echo ""
echo "=== 3. fit the head ==="
echo "  CELLS must come from step 1. Read logs/cells_${TAG}.txt, then rerun this step with --cells."
echo "  Not guessed here: a wrong layer produces a head that fits and scores nothing meaningful."
if [ -n "${CELLS:-}" ]; then
    python -u -m projects.pedagogy_rm.fit_head \
        --hidden "data/encoders/${TAG}.npz" \
        --hack "data/encoders/hack_${TAG}.npz" \
        --labels 'data/labels/*.json' \
        --model "$ENCODER" \
        --dimensions leak,targeted,actionable,elicits,correct \
        --cells "$CELLS" \
        --out "data/head5_${TAG}.npz"
else
    echo "  CELLS unset, stopping after measurement. Rerun with e.g."
    echo "    CELLS='leak=eot:21,targeted=eot:18,actionable=last:18,elicits=eot:14,correct=eot:21' \\"
    echo "      sbatch projects/pedagogy_rm/scripts/refit_qwen.sbatch"
fi

echo ""
echo "done $(date -Is)"
