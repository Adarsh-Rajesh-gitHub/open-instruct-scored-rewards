#!/bin/bash
#SBATCH -J rlvr_math_olmoe
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/rlvr_math_olmoe_%j.out
#SBATCH -e logs/rlvr_math_olmoe_%j.err
#
# Verifiable-reward GRPO on OLMoE, same per-expert LoRA as the pedagogy run.
#
# ON \boxed{} - THE ANSWER IS THAT NOTHING NEEDS TO BE ADDED. MathVerifier tries four extractions
# in order: \boxed{...} first, then the Minerva form ("The final answer is X"), then the last
# $...$ span, then the whole completion normalised. So boxed is preferred but not required.
#
# And the dataset already teaches it. allenai/RLVR-GSM-MATH-IF-Mixed-Constraints prompts few-shot
# with worked solutions that end in \boxed{}, so the format is demonstrated rather than instructed.
# Adding "put your answer in \boxed{}" on top would be redundant and would change the prompt away
# from the one the published tulu3 result used. Verified by reading a row: the prompt carries
# worked examples ending in \boxed{[2,5)}, and ground_truth is the bare value, "8".
#
# THE REWARD IS ACCURACY, WHICH IS THE WHOLE REASON FOR THIS RUN. The pedagogy reward is a learned
# probe with a -0.60 length correlation and a 0.87-of-0.95 ceiling; here the reward is whether the
# final answer equals the reference. It cannot be gamed by writing shorter, and it needs no
# annotation. If per-expert LoRA on OLMoE works at all, it should show here first and unambiguously.
#
# One mix drives three verifiers: rows carry "MATH", "gsm8k" or an ifeval constraint in their
# `dataset` field and each is dispatched to its own verifier. Names are lowercased before lookup,
# so "MATH" resolves to the registered `math`. Checked against the registry - all three are present.
set -euo pipefail

if [ -z "${PEDAGOGY_SNAPSHOT:-}" ]; then
    snapshot="${TMPDIR:-/tmp}/rlvr_olmoe_${SLURM_JOB_ID:-$$}.sbatch"
    cp "${BASH_SOURCE[0]}" "$snapshot"; export PEDAGOGY_SNAPSHOT="$snapshot"; exec bash "$snapshot"
fi
echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

export POLICY=${POLICY:-allenai/OLMoE-1B-7B-0924-Instruct}
export EXP=${EXP:-rlvr_math_olmoe}
export WANDB_PROJECT=${WANDB_PROJECT:-rlvr-verifiable}
export TRAIN_MIX=${TRAIN_MIX:-"allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0"}
export EVAL_MIX=${EVAL_MIX:-"allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16"}

export LORA_R=${LORA_R:-8}
export LORA_ALPHA=${LORA_ALPHA:-32}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"q_proj k_proj v_proj o_proj"}
# PER-EXPERT LORA, WHICH NEEDED A WEIGHT-SYNC FIX BEFORE IT COULD RUN AT ALL.
# transformers 5 stores OLMoE's 64 experts as one stacked parameter per projection, and vLLM's
# OlmoeForCausalLM still expects the per-expert checkpoint names. The first two attempts here died
# at the first weight sync with
#     Call to collective_rpc method failed: 'layers.0.mlp.experts.gate_up_proj'
# and turning expert LoRA off did NOT help, which is the informative part: the sync ships every
# base weight regardless of what LoRA adapts, so the fused name broke it either way. The fix is
# _split_fused_experts in vllm_utils, covered by test_moe_weight_split against the real forward.
#
# Rank 2 on experts and 8 on attention follows PERFT, which finds expert-side adapters worth
# 69.42 against 62.86 for attention-only on this model - so this config is the reason to bother
# with OLMoE rather than an incidental setting.
export EXPERT_PARAMS=${EXPERT_PARAMS:-"mlp.experts.gate_up_proj mlp.experts.down_proj"}
export EXPERT_R=${EXPERT_R:-2}
export EXPERT_ALPHA=${EXPERT_ALPHA:-8}

export LR=${LR:-4e-5}
export MICRO_BATCH=${MICRO_BATCH:-2}
export EPISODES=${EPISODES:-51200}
export SAVE_FREQ=${SAVE_FREQ:-20}
# Maths needs room to reason before the answer, unlike a tutoring turn. No frozen probe encoder is
# resident here, so vLLM gets a larger share than the pedagogy run gives it.
export MAX_PROMPT=${MAX_PROMPT:-1024}
export RESPONSE_LEN=${RESPONSE_LEN:-1024}
export PACK_LEN=${PACK_LEN:-2048}
export VLLM_UTIL=${VLLM_UTIL:-0.55}

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
export SCRATCH REPO
mkdir -p "${CKPT_ROOT:-$SCRATCH/ckpt}"
export INNER=projects/rlvr_verifiable/scripts/train.sh

# FAIL LOUDLY IF THE SHARED WRAPPER CANNOT DISPATCH, BECAUSE THE FALLBACK IS SILENT AND WRONG.
# train.sbatch used to hardcode `bash projects/pedagogy_rm/scripts/train.sh`. Against that version
# INNER is simply ignored: the job starts, trains happily for hours, logs to the rlvr-verifiable
# project under the name rlvr_math_olmoe - and is a second pedagogy run on data/rl/train.jsonl with
# the pedagogy reward head. Nothing in the metrics says so. It cost an H100 for half an hour and was
# only caught by reading the run's group_scorer, which said `pedagogy:head=data/head.npz`.
#
# One grep turns that into a startup error. The check is on the deployed file rather than the repo
# because the two drift: this exact bug was a stale copy on the cluster while the fix sat locally.
WRAPPER=$REPO/projects/pedagogy_rm/scripts/train.sbatch
if ! grep -q 'INNER' "$WRAPPER"; then
    echo "FATAL: $WRAPPER does not honour INNER, so this would silently run pedagogy instead" >&2
    echo "       redeploy projects/pedagogy_rm/scripts/train.sbatch and resubmit" >&2
    exit 1
fi
exec bash "$WRAPPER"
