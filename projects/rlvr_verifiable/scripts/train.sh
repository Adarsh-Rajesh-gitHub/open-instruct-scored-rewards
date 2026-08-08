#!/bin/bash
# GRPO against open-instruct's built-in verifiers. Maths, code and factual QA, no new reward code.
#
# WHAT THIS DOES NOT CONTAIN, WHICH IS THE POINT. There is no reward model here, no plugin, no
# scorer. open-instruct already ships verifiers for every domain this project targets, and a row
# selects one by putting its name in the `dataset` field:
#
#     domain    verifier name    what it checks
#     maths     math             final answer against the reference, latex and sympy aware
#     maths     gsm8k            the integer after ####
#     maths     strict_math      as `math` but without the lenient fallbacks
#     code      code             runs the extracted python against assert-style tests
#     code      code_stdio       runs it against (stdin, expected stdout) pairs
#     factual   string_f1        token F1 against the reference answer
#     factual   string_matcher   exact match after normalisation
#
# So the whole reward path is `--apply_verifiable_reward True` plus data with the right `dataset`
# field. Compare projects/pedagogy_rm, which needed a plugin, a fitted head and a scorer registry
# because "was that good teaching" has no verifier and had to be learned.
#
# THE CLUSTER PLUMBING IS ALSO NOT HERE. projects/pedagogy_rm/scripts/train.sbatch is the Slurm
# wrapper and it now takes INNER, so the per-domain launchers in this directory point it at this
# file and inherit the snapshot-and-re-exec, the per-node flashinfer cache, the offline hub pin and
# the provenance tagging without copying any of it.
set -euo pipefail

MODE=${MODE:-lora}
EXP=${EXP:-rlvr}
POLICY=${POLICY:-allenai/OLMo-2-1124-7B-Instruct}

# Dataset mixer entries, as `name weight` pairs. Passed through verbatim, so these can be HF repo
# ids or local jsonl paths - grpo_fast.py resolves either.
TRAIN_MIX=${TRAIN_MIX:?set TRAIN_MIX, e.g. "allenai/RLVR-GSM 1.0"}
EVAL_MIX=${EVAL_MIX:-$TRAIN_MIX}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
EVAL_SPLIT=${EVAL_SPLIT:-train}

PROMPTS=${PROMPTS:-32}
SAMPLES=${SAMPLES:-8}
EPISODES=${EPISODES:-51200}
LR=${LR:-4e-5}
BETA=${BETA:-0.02}
MICRO_BATCH=${MICRO_BATCH:-4}
SEED=${SEED:-1}

# Verifiable tasks need room to reason before the answer, unlike a tutoring turn which is capped at
# a couple of sentences. 2048 in and out, packed to 4096, following the tulu3 RLVR recipe.
MAX_PROMPT=${MAX_PROMPT:-2048}
RESPONSE_LEN=${RESPONSE_LEN:-2048}
PACK_LEN=${PACK_LEN:-4096}

LORA_R=${LORA_R:-32}
# MIXTURE-OF-EXPERT KNOBS, empty for a dense policy so nothing changes for the existing arms.
# EXPERT_PARAMS names fused 3-D expert tensors that target_modules cannot reach; EXPERT_R and
# EXPERT_ALPHA give them their own rank and scaling. For OLMoE the settings PERFT measured are
#   EXPERT_PARAMS="mlp.experts.gate_up_proj mlp.experts.down_proj" EXPERT_R=2 EXPERT_ALPHA=8
# which is 16.8M trainable and keeps alpha/r equal to attention's.
EXPERT_PARAMS=${EXPERT_PARAMS:-}
EXPERT_R=${EXPERT_R:-}
EXPERT_ALPHA=${EXPERT_ALPHA:-}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-}
LORA_ALPHA=${LORA_ALPHA:-64}
LEARNERS=${LEARNERS:-1}
ENGINES=${ENGINES:-1}
TP=${TP:-1}
VLLM_UTIL=${VLLM_UTIL:-0.45}
COLOCATE=${COLOCATE:-1}
GRAD_CKPT=${GRAD_CKPT:-1}
EVAL_EVERY=${EVAL_EVERY:-10}
SAVE_FREQ=${SAVE_FREQ:-20}
OUTPUT_DIR=${OUTPUT_DIR:-output/$EXP}
CKPT_ROOT=${CKPT_ROOT:-}

tuning=()
if [ "$MODE" = full ]; then
    tuning+=(--learning_rate "${LR_FULL:-3e-7}" --deepspeed_stage 3)
    [ "${OFFLOAD:-0}" = 1 ] && tuning+=(--deepspeed_offload_optimizer)
elif [ "$MODE" = lora ]; then
    tuning+=(
        --learning_rate "$LR"
        --deepspeed_stage 2
        --use_peft
        --lora_r "$LORA_R"
        ${LORA_TARGET_MODULES:+--lora_target_modules $LORA_TARGET_MODULES}
        ${EXPERT_PARAMS:+--lora_target_parameters $EXPERT_PARAMS}
        ${EXPERT_R:+--lora_expert_rank "$EXPERT_R"}
        ${EXPERT_ALPHA:+--lora_expert_alpha "$EXPERT_ALPHA"}
        --lora_alpha "$LORA_ALPHA"
        --lora_dropout 0.0
    )
else
    echo "MODE must be full or lora, got '$MODE'" >&2
    exit 1
fi

[ "$COLOCATE" = 1 ] && tuning+=(--single_gpu_mode)
[ "$GRAD_CKPT" = 1 ] && tuning+=(--gradient_checkpointing)
tuning+=(--push_to_hub False)

if [ "${WANDB_MODE:-online}" != disabled ]; then
    tuning+=(--with_tracking --wandb_project "${WANDB_PROJECT:-rlvr-verifiable}")
    [ -n "${WANDB_ENTITY:-}" ] && tuning+=(--wandb_entity "$WANDB_ENTITY")
fi

[ -n "$CKPT_ROOT" ] && tuning+=(--checkpoint_state_dir "$CKPT_ROOT/$EXP" --checkpoint_state_freq "$SAVE_FREQ")

# The code verifier posts to an execution service. It is bundled - open_instruct/code_utils/api.py -
# and the code launcher starts it before calling this script, so all that is needed here is to pass
# the URL through. Left unset for the maths and factual domains, where nothing reads it.
[ -n "${CODE_API_URL:-}" ] && tuning+=(--code_api_url "$CODE_API_URL/test_program")

echo "=== $EXP ==="
echo "  policy    $POLICY  ($MODE)"
echo "  train mix $TRAIN_MIX"
echo "  eval mix  $EVAL_MIX"
echo "  lr $LR  beta $BETA  ${PROMPTS}x${SAMPLES} per step  ${EPISODES} episodes"

# shellcheck disable=SC2086  # the mixer args are intentionally word-split into name/weight pairs
python -u open_instruct/grpo_fast.py \
    --exp_name "$EXP" \
    --model_name_or_path "$POLICY" \
    --tokenizer_name_or_path "$POLICY" \
    --use_slow_tokenizer False \
    --dataset_mixer_list $TRAIN_MIX \
    --dataset_mixer_list_splits "$TRAIN_SPLIT" \
    --dataset_mixer_eval_list $EVAL_MIX \
    --dataset_mixer_eval_list_splits "$EVAL_SPLIT" \
    --apply_verifiable_reward True \
    --max_prompt_token_length "$MAX_PROMPT" \
    --response_length "$RESPONSE_LEN" \
    --pack_length "$PACK_LEN" \
    --num_unique_prompts_rollout "$PROMPTS" \
    --num_samples_per_prompt_rollout "$SAMPLES" \
    --temperature 1.0 \
    --beta "$BETA" \
    "${tuning[@]}" \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.03 \
    --total_episodes "$EPISODES" \
    --per_device_train_batch_size "$MICRO_BATCH" \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --num_learners_per_node "$LEARNERS" \
    --vllm_num_engines "$ENGINES" \
    --vllm_tensor_parallel_size "$TP" \
    --vllm_gpu_memory_utilization "$VLLM_UTIL" \
    --local_eval_every "$EVAL_EVERY" \
    --save_freq "$SAVE_FREQ" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED"
