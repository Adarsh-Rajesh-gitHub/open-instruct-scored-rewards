#!/bin/bash
#SBATCH -J pedagogy_bench
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_bench_%j.out
#SBATCH -e logs/pedagogy_bench_%j.err
#
# Does 200 steps of tutoring-shaped reinforcement learning cost the model any academic ability?
# Nothing measured so far would notice: every reward dimension is about how a turn is phrased, and
# the blind evaluations only ever compared tutoring against tutoring.
#
# HF_HUB_OFFLINE IS DELIBERATELY NOT SET HERE, unlike every other script in this directory. The
# benchmark datasets are not in the local cache and have to be fetched once; these nodes can reach
# huggingface.co. Everything else stays pointed at the shared cache so the 7B weights come from
# disk rather than the network.
set -euo pipefail

if [ -z "${PEDAGOGY_SNAPSHOT:-}" ]; then
    snapshot="${TMPDIR:-/tmp}/bench_${SLURM_JOB_ID:-$$}.sbatch"
    cp "${BASH_SOURCE[0]}" "$snapshot"
    export PEDAGOGY_SNAPSHOT="$snapshot"
    exec bash "$snapshot"
fi

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$SCRATCH/venv_oi}
LIMIT=${LIMIT:-400}

export SCRATCH CACHE_ROOT=$SCRATCH
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
mkdir -p logs data/benchmarks

latest () {
    ls -d "$REPO"/output/"$1"/*_checkpoints/step_* 2>/dev/null \
        | sed 's/.*step_//' | sort -n | tail -1 \
        | xargs -I{} sh -c "ls -d $REPO/output/$1/*_checkpoints/step_{}"
}

# Base first, so if the job is preempted there is still a reference point to compare against.
echo "=== base (no adapter) ==="
python -u projects/pedagogy_rm/benchmark.py --adapter "" --tag base --limit "$LIMIT"

for pair in "arm_e:pedagogy_olmo7b_e" "arm_g:pedagogy_olmo7b_g"; do
    tag=${pair%%:*}; run=${pair#*:}
    ckpt=$(latest "$run")
    [ -z "$ckpt" ] && { echo "no checkpoint for $run; skipping" >&2; continue; }
    echo "=== $tag  $ckpt ==="
    python -u projects/pedagogy_rm/benchmark.py --adapter "$ckpt" --tag "$tag" --limit "$LIMIT"
done

echo "done $(date -Is)"
