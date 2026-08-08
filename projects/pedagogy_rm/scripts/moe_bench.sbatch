#!/bin/bash
#SBATCH -J moe_bench
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/moe_bench_%j.out
#SBATCH -e logs/moe_bench_%j.err
#
# Runs the same benchmark under both environments so the transformers 4.x ModuleList path and the
# 5.x fused path are one number apart rather than an argument. Same model, batch, sequence length
# and ranks in both; only the environment differs.
set -euo pipefail
SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
export HF_HOME=$SCRATCH/hf-cache HF_HUB_CACHE=$SCRATCH/hf-cache/hub
export XDG_CACHE_HOME=$SCRATCH/xdg-cache TRITON_CACHE_DIR=$SCRATCH/triton-cache
set +u
command -v module >/dev/null 2>&1 || source /usr/share/lmod/lmod/init/bash
module load cuda/13.0.1
set -u
cd "$REPO"; export PYTHONPATH=$REPO; mkdir -p logs

for pair in "venv_test:$HOME/venv_test" "venv_oi:$SCRATCH/venv_oi"; do
    label=${pair%%:*}; venv=${pair#*:}
    echo ""
    if [ ! -x "$venv/bin/python" ]; then echo "no python in $venv, skipping"; continue; fi
    "$venv/bin/python" -u projects/pedagogy_rm/scripts/moe_bench.py --label "$label" \
        || echo "  BENCHMARK FAILED under $label" >&2
done
echo ""
echo "done $(date -Is)"
