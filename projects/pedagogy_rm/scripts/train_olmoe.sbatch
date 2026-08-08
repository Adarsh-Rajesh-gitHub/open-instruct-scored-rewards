#!/bin/bash
#SBATCH -J pedagogy_olmoe
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_olmoe_%j.out
#SBATCH -e logs/pedagogy_olmoe_%j.err
#
# The pedagogy reward, unchanged, driving an OLMoE-1B-7B policy with per-expert LoRA.
#
# THE REWARD DOES NOT CHANGE AND THAT IS THE POINT. plugin.py loads its own frozen
# OLMo-2-7B-Instruct and scores the TEXT, not the policy's internals, so swapping the policy to a
# different model family touches no reward code and head5.npz still applies. What does change is
# that the probe now reads text from a family it was never fitted on, so expect it to be LESS
# trustworthy here than on OLMo, not equally so - `leak`'s length coupling was already 0.44 in
# distribution and 0.70 out of it. Read this run as "does the mechanism work", not "is this tutor
# better"; only a blind human comparison answers the second.
#
# THE LORA CONFIGURATION IS PERFT'S, MEASURED ON THIS EXACT MODEL. arXiv 2411.08212 sweeps PEFT
# placements on OLMoE-1B-7B and its PERFT-E variant is this setup - one LoRA per expert behind the
# frozen pretrained router. At matched activated parameters it scores 69.42 on commonsense against
# 62.86 for attention-only, 54.58 for a single shared adapter and 48.29 for router-only.
#
#   r=2 ON EXPERTS, NOT 1. PERFT's rank sweep is flat from 2 to 64 (68.79/69.42/68.56/69.29/
#   68.97/68.32) and never tested 1. PEFT's docs suggest r//num_experts, which is 1 here, but that
#   rule miscites its source: "LoRA Without Regret" divides by ACTIVE experts (8), not total (64).
#   Step time is dominated by materialising the delta and is nearly rank-independent, so r=2 sits
#   inside the measured range for free.
#
#   ALPHA IS SET EXPLICITLY. Scaling is alpha/r, so leaving alpha global while shrinking the expert
#   rank would scale experts 4x harder than attention (32/2 against 32/8). EXPERT_ALPHA=8 equalises
#   them at 4.0. PEFT's own MoE example has this asymmetry and does not mention it.
#
#   THE ROUTER IS FROZEN. mlp.gate is not in either target list. DenseMixer measured frozen against
#   trainable on OLMoE+LoRA at 32.76 and 32.85 - a 0.09 difference - and the reason is structural:
#   non-selected experts contribute no gradient to their router logits, so training the router can
#   sharpen existing routing but cannot redirect it.
#
#   NO HOT-EXPERT SUBSETTING, despite MoE-Sieve reporting 70-73% fewer expert parameters at parity
#   on this model. PEFT's fused path materialises the full delta for all 64 experts regardless of
#   how many carry adapters, so the saving does not exist here; it comes from the ModuleList
#   representation, which benchmarked 3.7x slower overall. The two papers also disagree on whether
#   the method helps at all (parity vs +3.1 on the same model and task).
#
# NO AUX LOAD-BALANCING LOSS. Not enabled by grpo_fast and deliberately left off: with the router
# frozen its gradient would fall entirely on the adapters, pushing them to equalise expert load,
# which is not what this run is for.
#
# WHAT TO WATCH, IN ORDER. First that reward moves at all. Then `objective/kl1_avg` against the
# dense arms' 0.25-0.30. Then, before trusting anything, expert-selection agreement between vLLM
# and the trainer: R3 (arXiv 2510.11370) measured ~10% router disagreement on a 30B MoE and had
# three of three GRPO runs collapse without correction. Our exposure is lower - frozen router,
# identical base weights on both sides, 16 layers rather than 48 - but it is not zero, because top-k
# routing is discontinuous and kernel numerics differ between vLLM's fused path and grouped_mm.
set -euo pipefail

if [ -z "${PEDAGOGY_SNAPSHOT:-}" ]; then
    snapshot="${TMPDIR:-/tmp}/olmoe_${SLURM_JOB_ID:-$$}.sbatch"
    cp "${BASH_SOURCE[0]}" "$snapshot"; export PEDAGOGY_SNAPSHOT="$snapshot"; exec bash "$snapshot"
fi
echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

export POLICY=${POLICY:-allenai/OLMoE-1B-7B-0924-Instruct}
export SCORER=pedagogy
export HEAD=${HEAD:-data/head5.npz}
export LENGTH_BAND=${LENGTH_BAND:-18-40}
export LENGTH_RAMP=${LENGTH_RAMP:-8-90}
export LENGTH_WEIGHT=${LENGTH_WEIGHT:-2.0}
export EXP=${EXP:-pedagogy_olmoe}
export WANDB_PROJECT=${WANDB_PROJECT:-pedagogy-rm}

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
export MICRO_BATCH=${MICRO_BATCH:-4}
export EPISODES=${EPISODES:-51200}
export SAVE_FREQ=${SAVE_FREQ:-10}
# The policy is a second 7B beside the frozen probe encoder, so vLLM gets a smaller share than the
# dense arms used. OLMoE activates 1.3B per token, so the trainer's activations are cheaper.
export VLLM_UTIL=${VLLM_UTIL:-0.40}

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
export SCRATCH REPO
mkdir -p "${CKPT_ROOT:-$SCRATCH/ckpt}"
exec bash "$REPO/projects/pedagogy_rm/scripts/train.sbatch"
