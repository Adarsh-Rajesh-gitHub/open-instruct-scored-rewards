#!/bin/bash
#SBATCH -J pedagogy_sample_c
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_sample_c_%j.out
#SBATCH -e logs/pedagogy_sample_c_%j.err
#
# ONE ARM, NOT THREE, WHICH IS THE WHOLE POINT OF THIS FILE. sample.sbatch redraws base,
# arm A and arm B together. Redrawing base and arm A here would be worse than wasteful: their
# existing turns have already been rated by six agents and by a human, and new draws from the
# same policies would throw all of that away and need re-labelling to compare against.
#
# The turns in data/samples/base.json and arm_a.json stay exactly as they are, so arm C is
# compared against a reference that is already labelled. Only its own 60 turns are new.
#
# THE BLIND IDs SURVIVE THIS, which is what makes the reuse safe. build_blind_set.py derives
# each id from a salt (seeded), the arm's *name* and the source unit id - so as long as this
# rebuild passes the same seed and the same two names, every base and arm_a id comes out
# byte-identical and the existing label files still join. That is also the trap: passing
# `lora=` instead of `arm_a=` silently renames every id and orphans the labels.
#
# SCORED WITH head5.npz, the head arm C trained against, so `scores` in the sampled file is
# the quantity that was optimised. The four-dimension head.npz remains the right one for
# re-scoring base and arm A on their own terms; the two are not comparable and are not mixed.
set -euo pipefail

if [ -z "${PEDAGOGY_SNAPSHOT:-}" ]; then
    snapshot="${TMPDIR:-/tmp}/sample_c_${SLURM_JOB_ID:-$$}.sbatch"
    cp "${BASH_SOURCE[0]}" "$snapshot"
    export PEDAGOGY_SNAPSHOT="$snapshot"
    exec bash "$snapshot"
fi

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$SCRATCH/venv_oi}
MOMENTS=${MOMENTS:-60}
HEAD=${HEAD:-data/head5.npz}

# Resolved rather than hard-coded, because the run directory carries a timestamp. Takes the
# highest step present, which is the finished adapter unless the run was preempted short.
ARM_C=${ARM_C:-}
if [ -z "$ARM_C" ]; then
    ARM_C=$(ls -d "$REPO"/output/pedagogy_olmo7b_c/*_checkpoints/step_* 2>/dev/null \
            | sed 's/.*step_//' | sort -n | tail -1 \
            | xargs -I{} sh -c 'ls -d '"$REPO"'/output/pedagogy_olmo7b_c/*_checkpoints/step_{}')
fi
if [ ! -d "$ARM_C" ]; then
    echo "no arm C adapter found under $REPO/output/pedagogy_olmo7b_c/" >&2
    echo "pass ARM_C=/path/to/step_N explicitly, or wait for training to save one" >&2
    exit 1
fi
echo "arm C adapter: $ARM_C"

export SCRATCH CACHE_ROOT=$SCRATCH HF_HUB_OFFLINE=1
export HF_HOME=$SCRATCH/hf-cache HF_HUB_CACHE=$SCRATCH/hf-cache/hub
export XDG_CACHE_HOME=$SCRATCH/xdg-cache TRITON_CACHE_DIR=$SCRATCH/triton-cache
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

set +u
command -v module >/dev/null 2>&1 || source /usr/share/lmod/lmod/init/bash
module load cuda/13.0.1
source "$VENV/bin/activate"
set -u

GCC_LIB=${GCC_LIB:-/orcd/software/core/001/spack/pkg/gcc/12.2.0/yt6vabm/lib64}
[ -d "$GCC_LIB" ] && export LD_LIBRARY_PATH="$GCC_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR:-$SCRATCH/flashinfer-cache/$(hostname -s)}
mkdir -p "$FLASHINFER_CACHE_DIR"

cd "$REPO"
export PYTHONPATH=$REPO
mkdir -p logs data/samples

python -u projects/pedagogy_rm/sample_policy.py generate \
    --prompts data/rl/eval.jsonl --limit "$MOMENTS" --samples 1 --temperature 1.0 \
    --policy allenai/OLMo-2-1124-7B-Instruct --adapter "$ARM_C" \
    --tag arm_c --out data/samples/arm_c.json
python -u projects/pedagogy_rm/sample_policy.py score --units data/samples/arm_c.json --head "$HEAD"

# Rebuilt over the three arms that are still live. arm B is left out deliberately: it was
# indistinguishable from arm A in 17 blind pairs and claimed a third more than it was paid,
# so a fourth condition would cost labelling effort to re-answer a settled question.
python -u projects/pedagogy_rm/build_blind_set.py \
    --arm base=data/samples/base.json \
    --arm arm_a=data/samples/arm_a.json \
    --arm arm_c=data/samples/arm_c.json \
    --out-dir data/eval_c --moments 0 --seed 1

echo "done $(date -Is)"
