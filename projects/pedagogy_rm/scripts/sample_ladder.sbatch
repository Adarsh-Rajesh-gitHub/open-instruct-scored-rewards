#!/bin/bash
#SBATCH -J pedagogy_sample_ladder
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_sample_ladder_%j.out
#SBATCH -e logs/pedagogy_sample_ladder_%j.err
#
# Draw held-out turns from the three learning-rate arms and blind them together with base.
#
# WHAT THIS IS FOR, AND WHY THE REWARD CURVES CANNOT ANSWER IT. E, F and G differ only in learning
# rate (2e-5, 4e-5, 8e-5) and all three post a higher reward than base by construction - a larger
# step buying more reward is the premise. The question is whether the larger steps produced turns a
# person prefers, or only turns the head scores higher, which is exactly what arm B did while being
# a coin flip in blind pairs. Only a blinded human comparison separates those.
#
# BASE IS REDRAWN RATHER THAN REUSED. Its existing turns were scored with the four-dimension head
# and are already in two labelled pools; drawing it fresh here keeps all four arms on one sampling
# pass and one head, so nothing in the comparison depends on which round a turn came from.
#
# SCORED WITH head5.npz, the head all three arms trained against, so `scores` in each file is the
# quantity that was optimised rather than a different head's opinion of it.
set -euo pipefail

if [ -z "${PEDAGOGY_SNAPSHOT:-}" ]; then
    snapshot="${TMPDIR:-/tmp}/sample_ladder_${SLURM_JOB_ID:-$$}.sbatch"
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

# Highest saved step per arm, so a preempted run still contributes its furthest checkpoint.
latest () {
    ls -d "$REPO"/output/"$1"/*_checkpoints/step_* 2>/dev/null \
        | sed 's/.*step_//' | sort -n | tail -1 \
        | xargs -I{} sh -c "ls -d $REPO/output/$1/*_checkpoints/step_{}"
}

draw () {
    tag=$1; adapter=$2
    echo "=== $tag  ${adapter:-(no adapter)} ==="
    python -u projects/pedagogy_rm/sample_policy.py generate \
        --prompts data/rl/eval.jsonl --limit "$MOMENTS" --samples 1 --temperature 1.0 \
        --policy allenai/OLMo-2-1124-7B-Instruct ${adapter:+--adapter "$adapter"} \
        --tag "$tag" --out "data/samples/${tag}.json"
    python -u projects/pedagogy_rm/sample_policy.py score \
        --units "data/samples/${tag}.json" --head "$HEAD"
}

draw base_l ""
for pair in "arm_e:pedagogy_olmo7b_e" "arm_f:pedagogy_olmo7b_f" "arm_g:pedagogy_olmo7b_g"; do
    tag=${pair%%:*}; run=${pair#*:}
    ckpt=$(latest "$run")
    if [ -z "$ckpt" ] || [ ! -d "$ckpt" ]; then
        echo "no checkpoint for $run - skipping $tag" >&2
        continue
    fi
    echo "$tag -> $ckpt"
    draw "$tag" "$ckpt"
done

args=(--arm base=data/samples/base_l.json)
for tag in arm_e arm_f arm_g; do
    [ -f "data/samples/${tag}.json" ] && args+=(--arm "${tag}=data/samples/${tag}.json")
done
python -u projects/pedagogy_rm/build_blind_set.py "${args[@]}" \
    --out-dir data/eval_ladder --moments 0 --seed 1

echo "done $(date -Is)"
