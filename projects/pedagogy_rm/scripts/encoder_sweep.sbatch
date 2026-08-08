#!/bin/bash
#SBATCH -J pedagogy_encoders
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_encoders_%j.out
#SBATCH -e logs/pedagogy_encoders_%j.err
#
# DOES A STRONGER ENCODER READ TEACHING QUALITY BETTER? Same 600 turns, same labels, same
# question-grouped split, same layer and pooling sweep. Only the frozen model changes.
#
# WHY THIS IS THE CHEAPEST THING LEFT. Every other route to a better reward needs annotation:
# a new rubric needs re-rating, fixing `leak`'s length coupling needs labels on policy-generated
# turns. This one reuses the existing 600 labelled turns exactly as they are and costs one job.
#
# WHERE THE ROOM IS, against the inter-rater ceiling measured on these same labels:
#
#     dimension     probe   ceiling   room
#     targeted       0.85     0.94     0.09
#     leak           0.84     0.95     0.11
#     actionable     0.94     0.98     0.04
#     elicits        0.95     0.98     0.03
#
# The two with room are the two that matter: `leak` carries the length coupling that is the
# project's central unsolved problem, and `targeted` is the only length-neutral dimension and the
# one where activations most beat surface features (0.85 against 0.36). If a stronger encoder helps
# anywhere, those are where it would show, and they are the ones worth having.
#
# IT ALSO SEPARATES ENCODER FROM POLICY. Today both are OLMo-2-7B, so a residual "does this look
# like something I would write" component cannot be ruled out. A Qwen encoder scoring an OLMo policy
# removes that by construction.
#
# WHAT THE ANSWER DOES NOT FIX. Everything here is measured on base-model turns. The probe's real
# problem is that it reads policy-generated text it was never fitted on - `leak`'s length coupling
# is 0.44 in distribution and 0.70 out of it - and no choice of encoder addresses that. Only labels
# on policy-generated turns will. Read this sweep as "which encoder to fit on", not as a fix.
#
# DEPLOYMENT CONSTRAINT, because a winner has to fit beside the trainer and vLLM at training time.
# plugin.py drops every block above the deepest layer a head reads, so what must fit is roughly
# 21/N of the model rather than all of it: 7B costs 9.6GB, 14B costs 12.9GB because it is deeper,
# and both sit inside the ~20GB left on an 80GB card. 32B would not, which is why it is absent.
#
# HF_HUB_OFFLINE IS DELIBERATELY NOT SET, unlike the training scripts: three of these encoders are
# not in the cache and the point of the job is to fetch them. Everything else still points at the
# shared cache so nothing is downloaded twice.
set -euo pipefail

if [ -z "${PEDAGOGY_SNAPSHOT:-}" ]; then
    snapshot="${TMPDIR:-/tmp}/encoders_${SLURM_JOB_ID:-$$}.sbatch"
    cp "${BASH_SOURCE[0]}" "$snapshot"
    export PEDAGOGY_SNAPSHOT="$snapshot"
    exec bash "$snapshot"
fi

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$SCRATCH/venv_oi}
UNITS=${UNITS:-data/label_slices/slice_1.json}
LABELS=${LABELS:-"data/labels/*.json"}

export HF_HOME=$SCRATCH/hf-cache HF_HUB_CACHE=$SCRATCH/hf-cache/hub
export XDG_CACHE_HOME=$SCRATCH/xdg-cache TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor-cache

set +u
command -v module >/dev/null 2>&1 || source /usr/share/lmod/lmod/init/bash
module load cuda/13.0.1
source "$VENV/bin/activate"
set -u

cd "$REPO"
export PYTHONPATH=$REPO
mkdir -p logs data/encoders

# tag:model. The baseline is re-extracted rather than reusing data/hidden.npz, so every row of the
# comparison comes from one code path on one day.
#
# THE LIST IS A LADDER, NOT AN ASSORTMENT, so the result attributes rather than just ranks:
#   qwen3b -> qwen7b -> qwen14b -> qwen32b   varies SIZE with the family fixed
#   olmo7b vs qwen7b                         varies FAMILY with the size fixed
# Without the ladder, "Qwen-14B beats OLMo-7B" cannot distinguish a size effect from a family one,
# and those have different consequences: size says buy a bigger encoder, family says the current
# one is simply weaker than its parameter count suggests.
#
# 3B IS IN THE LIST TO CATCH A NULL CHEAPLY. If it reads these dimensions as well as 7B does, then
# encoder capacity is not the binding constraint and nothing above 7B is worth deploying either -
# which is a useful answer, and one the big models alone could not give.
#
# 32B IS MEASURED BUT MAY NOT BE DEPLOYABLE. Extraction has the whole card; at training time the
# encoder shares it with the trainer and vLLM, and 21/64 of a 32B is ~21.5GB against ~20GB free.
# It would need vllm_gpu_memory_utilization dropped or a second GPU. Worth knowing whether that
# trade buys anything before deciding to make it.
#
# Everything except 14B is already in the cache, so this costs one 28GB download.
ENCODERS=${ENCODERS:-"olmo7b:allenai/OLMo-2-1124-7B-Instruct \
qwen3b:Qwen/Qwen2.5-3B-Instruct \
qwen7b:Qwen/Qwen2.5-7B-Instruct \
qwen14b:Qwen/Qwen2.5-14B-Instruct \
qwen32b:Qwen/Qwen2.5-32B-Instruct"}

for pair in $ENCODERS; do
    tag=${pair%%:*}; model=${pair#*:}
    out=data/encoders/${tag}.npz
    echo ""
    echo "================ $tag  ($model) ================"
    if [ -f "$out" ]; then
        echo "  $out exists, skipping extraction"
    else
        # A failure here should not lose the encoders after it: a 14B download can fail on its own.
        if ! python -u -m projects.pedagogy_rm.extract_hidden \
                --units "$UNITS" --out "$out" --model "$model"; then
            echo "  EXTRACTION FAILED for $tag; continuing with the rest" >&2
            continue
        fi
    fi
    echo "---- probe accuracy, $tag ----"
    # --skip-mlp because ridge beat the MLP on every dimension when that was checked, and it makes
    # the pooling x layer sweep ~20x faster. --ceilings prints the inter-rater bound beside each
    # number so the comparison is against what is achievable, not against 1.0.
    # shellcheck disable=SC2086
    python -u -m projects.pedagogy_rm.probe \
        --hidden "$out" --labels $LABELS --skip-mlp --ceilings \
        || echo "  PROBE FAILED for $tag" >&2
done

echo ""
echo "================ summary ================"
echo "per-encoder probe output is above; data/encoders/*.npz kept for refitting the winner"
ls -la data/encoders/ 2>/dev/null | awk '{print $5, $9}'
echo "done $(date -Is)"
