"""Generate the on-policy corpus an EAGLE-3 draft for OLMoE is initialised from.

WHY THIS EXISTS AND WHY IT IS NOT "just generate some text". Draft initialisation is the single
largest lever on realised speedup that the paper measures. Its Table 3, at matched draft length
k=3, compares a draft trained on general chat data against one trained on the actual
post-training prompts:

    UltraChat init:  acceptance 2.88, speedup 1.51x   (RL-Zero)
    DAPO init:       acceptance 3.32, speedup 1.77x   (RL-Zero)

Same architecture, same k, same everything else. The paper's own reading: "Initialization
quality is not only about generic drafting ability, but about alignment with the rollout
distribution encountered during RL." So the corpus has to be what the *policy* says on the
*RLVR prompts*, not a general instruction-following set.

WHAT "ALIGNED" MEANS CONCRETELY. Every knob that shapes the token distribution has to match the
RL rollout: the checkpoint (the DPO model, which is what step 0 of RLVR generates with), the
prompt set (RLVR-GSM), the chat template and BOS handling, the temperature, and the response
length cap. The tokenizer and template are taken from open-instruct's own ``TokenizerConfig``
rather than reconstructed here, because a silently different template is exactly the class of
mismatch that invalidates a comparison after the GPU time is spent.

The defaults below therefore mirror the RLVR command in ``docs/olmo2.md``. Overriding them is
supported and is usually a mistake.

Sampling is deliberately *not* greedy. The draft has to model what the rollout actually
produces, and the rollout samples at temperature 1.0.

Usage (writes JSONL of {"messages": [user, assistant]} for the draft trainer):

    python projects/olmoe_specdec/generate_rollout_corpus.py \\
        --model_name_or_path allenai/OLMoE-1B-7B-0125-DPO \\
        --dtype bfloat16 \\
        --output_path "$EDULLM_OUTPUT_PREFIX/rollout_corpus.jsonl"
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import datasets
import vllm

from open_instruct import logger_utils
from open_instruct.dataset_transformation import TokenizerConfig

logger = logger_utils.setup_logger(__name__)

# Mirrors the RLVR recipe in docs/olmo2.md. See the module docstring on why these are not
# free parameters.
DEFAULT_MODEL = "allenai/OLMoE-1B-7B-0125-DPO"
DEFAULT_DATASET = "allenai/RLVR-GSM"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_RESPONSE_LENGTH = 2048
DEFAULT_MAX_PROMPT_TOKENS = 2048
DEFAULT_CHAT_TEMPLATE = "tulu"
# 4 rather than the rollout's 16. The draft is trained on tokens, not on groups, so what it
# needs is coverage of the response distribution; 16 samples of the same prompt mostly repeat
# each other's early tokens and cost 4x the generation time for little extra signal.
DEFAULT_SAMPLES_PER_PROMPT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--output_path", required=True, help="Destination .jsonl (or an s3:// prefix's local mount)")
    parser.add_argument("--samples_per_prompt", type=int, default=DEFAULT_SAMPLES_PER_PROMPT)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--response_length", type=int, default=DEFAULT_RESPONSE_LENGTH)
    parser.add_argument("--max_prompt_token_length", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument("--chat_template_name", default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--add_bos", action="store_true", default=True)
    parser.add_argument("--max_prompts", type=int, default=0, help="0 = all. Use a small value for a smoke test.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help=(
            "Named explicitly because the platform's bfloat16_not_in_the_hardware guard reads "
            "the text of the submitted command and cannot see a precision set in code."
        ),
    )
    return parser.parse_args()


def build_prompts(args: argparse.Namespace, tokenizer) -> tuple[list[str], list[list[dict]]]:
    """Render each RLVR prompt through the RL run's own chat template.

    Returns the rendered prompt strings and the message lists they came from, so the corpus can
    record the conversation rather than the rendered string -- draft trainers want messages, and
    a rendered string would bake in a template the trainer may apply again.
    """
    dataset = datasets.load_dataset(args.dataset, split=args.dataset_split)
    if args.max_prompts:
        dataset = dataset.select(range(min(args.max_prompts, len(dataset))))

    prompts: list[str] = []
    messages: list[list[dict]] = []
    skipped = 0
    for row in dataset:
        # RLVR-GSM carries a `messages` column whose last turn is the user's question.
        conversation = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
        rendered = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        if len(tokenizer(rendered)["input_ids"]) > args.max_prompt_token_length:
            # Dropped rather than truncated, to match what the RL data loader does: a truncated
            # prompt would put the draft on inputs the policy never sees.
            skipped += 1
            continue
        prompts.append(rendered)
        messages.append(conversation)

    logger.info(
        "prepared %d prompts from %s (%d skipped for exceeding %d prompt tokens)",
        len(prompts),
        args.dataset,
        skipped,
        args.max_prompt_token_length,
    )
    return prompts, messages


def main() -> None:
    args = parse_args()
    # Resolved to absolute up front: a relative path resolves against this process's cwd while a
    # subprocess may run with another, which has already cost one platform run.
    output_path = pathlib.Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer_config = TokenizerConfig(
        tokenizer_name_or_path=args.model_name_or_path,
        tokenizer_revision=args.model_revision,
        chat_template_name=args.chat_template_name,
        add_bos=args.add_bos,
    )
    tokenizer = tokenizer_config.tokenizer
    prompts, messages = build_prompts(args, tokenizer)
    if not prompts:
        raise SystemExit("no prompts survived preparation; check --dataset and --max_prompt_token_length")

    llm = vllm.LLM(
        model=args.model_name_or_path,
        revision=args.model_revision,
        tokenizer=args.model_name_or_path,
        dtype=args.dtype,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # No speculative_config here on purpose. This corpus is what a draft is trained *from*;
        # drafting while producing it would be circular, and there is no draft yet anyway.
    )
    sampling = vllm.SamplingParams(
        n=args.samples_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.response_length,
        seed=args.seed,
    )

    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - started

    written = 0
    total_tokens = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for conversation, output in zip(messages, outputs):
            for completion in output.outputs:
                text = completion.text
                if not text.strip():
                    # An empty completion teaches the draft nothing and would skew the
                    # length distribution it is fitted to.
                    continue
                record = {"messages": [*conversation, {"role": "assistant", "content": text}]}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                total_tokens += len(completion.token_ids)

    logger.info(
        "wrote %d conversations (%d generated tokens) to %s in %.1fs (%.0f tok/s)",
        written,
        total_tokens,
        output_path,
        elapsed,
        total_tokens / elapsed if elapsed > 0 else 0.0,
    )
    # Written beside the corpus rather than only logged, so a draft checkpoint can be traced back
    # to the exact distribution it was fitted to. Table 3's whole point is that this matters.
    manifest = output_path.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "model": args.model_name_or_path,
                "model_revision": args.model_revision,
                "dataset": f"{args.dataset}:{args.dataset_split}",
                "chat_template_name": args.chat_template_name,
                "add_bos": args.add_bos,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "response_length": args.response_length,
                "samples_per_prompt": args.samples_per_prompt,
                "seed": args.seed,
                "num_prompts": len(prompts),
                "num_conversations": written,
                "num_generated_tokens": total_tokens,
                "tokenizer_files_hash": tokenizer_config.tokenizer_files_hash,
                "generation_seconds": elapsed,
                "output_prefix_env": os.environ.get("EDULLM_OUTPUT_PREFIX"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote manifest to %s", manifest)


if __name__ == "__main__":
    main()
