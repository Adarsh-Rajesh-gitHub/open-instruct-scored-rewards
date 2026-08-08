#!/bin/bash
#SBATCH -J rlvr_code
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/rlvr_code_%j.out
#SBATCH -e logs/rlvr_code_%j.err
#
# CODE, verified by actually running it. `code` executes the extracted python against assert-style
# tests; `code_stdio` feeds (stdin, expected stdout) pairs. Both are open-instruct's CodeVerifier.
#
# THE ONE DOMAIN THAT NEEDS A SERVICE, which is why this launcher is longer than the other two.
# CodeVerifier does not execute anything in-process: it POSTs the candidate program and its tests to
# an HTTP endpoint and reads back a pass rate. That endpoint is bundled at
# open_instruct/code_utils/api.py, so nothing external is required, but it has to be running before
# the first reward is computed or every rollout scores zero and the run looks like a failure to learn
# rather than a failure to connect.
#
# STARTED INSIDE THE ALLOCATION, ON LOCALHOST, deliberately. The alternative is a long-lived shared
# service, which would make one job's runaway program another job's timeout, and would put a code
# execution endpoint on a shared network. Here the service lives and dies with the job and listens
# only on 127.0.0.1.
#
# IT IS STILL EXECUTING MODEL-WRITTEN CODE. api.py runs each program in a subprocess with a timeout
# (code_max_execution_time, 1 second by default), which bounds runtime but is not a sandbox - it does
# not stop filesystem or network access. On a shared cluster that is a real consideration and the
# reason to prefer the Dockerfile at open_instruct/code/Dockerfile if this ever runs unattended.
set -euo pipefail

export EXP=${EXP:-rlvr_code}
export TRAIN_MIX=${TRAIN_MIX:-"allenai/rlvr-code-data-python-r1-format-filtered-keyword-filtered-filter-datecutoff-ngram-filtered 1.0"}
export EVAL_MIX=${EVAL_MIX:-$TRAIN_MIX}
export WANDB_PROJECT=${WANDB_PROJECT:-rlvr-verifiable}

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$SCRATCH/venv_oi}
PORT=${CODE_API_PORT:-1234}

# The executor, backgrounded, then waited for. `uvicorn` needs the repo on PYTHONPATH to import
# open_instruct.code_utils, and the venv activated for fastapi.
set +u
command -v module >/dev/null 2>&1 || source /usr/share/lmod/lmod/init/bash
source "$VENV/bin/activate"
set -u
cd "$REPO"
export PYTHONPATH=$REPO
mkdir -p logs

echo "starting the code executor on 127.0.0.1:$PORT"
uvicorn open_instruct.code_utils.api:app --host 127.0.0.1 --port "$PORT" \
    > "logs/code_api_${SLURM_JOB_ID:-$$}.log" 2>&1 &
API_PID=$!
# Kill it however this script exits, including on preemption, so a requeue does not find the port
# taken by the previous attempt's orphan.
trap 'kill $API_PID 2>/dev/null || true' EXIT

# Health-check rather than sleep: the reward silently returns zero if the endpoint is not up, and a
# fixed sleep would sometimes be too short on a loaded node.
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "executor healthy after ${i}s"
        break
    fi
    if ! kill -0 $API_PID 2>/dev/null; then
        echo "executor died during startup; its log follows" >&2
        cat "logs/code_api_${SLURM_JOB_ID:-$$}.log" >&2
        exit 1
    fi
    sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || {
    echo "executor never became healthy after 60s" >&2; exit 1; }

# Verified end to end before spending a GPU-hour: a program that should pass and one that should not.
python - <<'PY' || { echo "executor gave wrong answers; not starting training" >&2; exit 1; }
import os, requests
url = f"http://127.0.0.1:{os.environ.get('CODE_API_PORT', '1234')}/test_program"
good = requests.post(url, json={"program": "def add(a, b): return a + b",
                                "tests": ["assert add(1, 2) == 3"], "max_execution_time": 1.0}).json()
bad = requests.post(url, json={"program": "def add(a, b): return a - b",
                               "tests": ["assert add(1, 2) == 3"], "max_execution_time": 1.0}).json()
print("  passing program ->", good)
print("  failing program ->", bad)
assert sum(good["results"]) == 1, good
assert sum(bad["results"]) == 0, bad
print("  executor distinguishes pass from fail")
PY

# CodeVerifier appends /test_program itself, so hand it the base URL.
export CODE_API_URL="http://127.0.0.1:$PORT"
export INNER=projects/rlvr_verifiable/scripts/train.sh
exec bash "$REPO/projects/pedagogy_rm/scripts/train.sbatch"
