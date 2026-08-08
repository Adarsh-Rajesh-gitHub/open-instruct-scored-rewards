#!/bin/bash
#SBATCH -J pedagogy_prompted_base
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_prompted_base_%j.out
#SBATCH -e logs/pedagogy_prompted_base_%j.err
#
# THE CONTROL THE PROJECT NEVER RAN, and the cheapest thing that could overturn its headline.
#
# The reward is a 5-dimension probe mean plus an explicit length term. Three measurements together
# make one rival explanation urgent:
#
#   1. The probe reward correlates -0.60 with log word count. Four of its five dimensions push
#      toward shorter turns; only `targeted` is length-neutral at +0.03.
#   2. The explicit length term supplied ~70% of arm E's measured gain and, by design, owns 48% of
#      the objective (2.0 of a 4.2 maximum).
#   3. At matched KL drift, removing the length term buys NO probe score - arm H scores 1.850 at
#      KL 0.247 against arm E's 1.852 - it only buys shorter turns, 22.4 words against 27.6.
#
# Taken together: most of what training moved is length, and length is the one thing a prompt can
# ask for directly. So the question this run answers is whether 200 GRPO steps beat one sentence of
# instruction. Every comparison so far has been against an UNPROMPTED base, which cannot tell those
# apart.
#
# WHAT WOULD FALSIFY WHAT. If prompted base lands near arm E on the probe score and inside the
# length band, then the training contributed little that a prompt does not, and the paper's claim
# has to shrink to match. If it hits the band but scores clearly below arm E on the five
# dimensions, the training bought something a prompt cannot - which is the result the project needs
# and has never actually demonstrated.
#
# THE INSTRUCTION IS THE BAND ITSELF, 18-40 words, because that is what the reward was tuned to and
# so is the strongest fair version of the rival. Asking for something vaguer ("be brief") would
# make the control weaker than the hypothesis it is testing, which is the standard way to rig this
# kind of comparison without meaning to.
#
# SAME PROMPTS, SAME SAMPLING, SAME HEAD as every other arm: data/rl/eval.jsonl, temperature 1.0,
# two samples per prompt, head5.npz. The only difference from the existing base sample is one
# appended sentence, so the pairing by question is exact.
set -euo pipefail

if [ -z "${PEDAGOGY_SNAPSHOT:-}" ]; then
    snapshot="${TMPDIR:-/tmp}/prompted_base_${SLURM_JOB_ID:-$$}.sbatch"
    cp "${BASH_SOURCE[0]}" "$snapshot"
    export PEDAGOGY_SNAPSHOT="$snapshot"
    exec bash "$snapshot"
fi

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$SCRATCH/venv_oi}
HEAD=${HEAD:-data/head5.npz}
LIMIT=${LIMIT:-100}
SAMPLES=${SAMPLES:-2}

export SCRATCH CACHE_ROOT=$SCRATCH
export HF_HOME=$SCRATCH/hf-cache HF_HUB_CACHE=$SCRATCH/hf-cache/hub HF_HUB_OFFLINE=1
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
mkdir -p logs data/samples_prompted

BAND="Keep your reply between 18 and 40 words."

draw () {
    tag=$1; out=$2; instruction=$3
    echo "=== $tag ==="
    python -u projects/pedagogy_rm/sample_policy.py generate \
        --policy allenai/OLMo-2-1124-7B-Instruct \
        --prompts data/rl/eval.jsonl \
        --limit "$LIMIT" --samples "$SAMPLES" \
        --temperature 1.0 \
        --tag "$tag" --out "$out" \
        ${instruction:+--append-system "$instruction"}
    python -u projects/pedagogy_rm/sample_policy.py score \
        --units "$out" --head "$HEAD"
}

# The control, and an unprompted redraw beside it so the comparison does not rest on a base sample
# taken weeks ago under a possibly different sampling configuration.
draw prompted_base data/samples_prompted/prompted_base.json "$BAND"
draw plain_base    data/samples_prompted/plain_base.json    ""

echo "done $(date -Is)"
