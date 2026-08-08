"""Restore an upcycled checkpoint and run a teacher-generation smoke test."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from projects.dense_to_moe_upcycling.upcycled_mlp import (
    iter_upcycled_layers,
    restore_moe_checkpoint,
    routing_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-3B-Instruct"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--prompt",
        default=(
            "A student says: 'A heavier ball must fall faster because "
            "gravity pulls on it with more force.' Give one concise "
            "tutoring response that diagnoses the idea without simply "
            "stating the answer."
        ),
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    payload = restore_moe_checkpoint(model, args.checkpoint)
    manifest = payload["manifest"]
    del payload
    gc.collect()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise teacher. Diagnose the learner's "
                "reasoning and choose an instructional response."
            ),
        },
        {"role": "user", "content": args.prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(rendered, return_tensors="pt").to("cuda")
    for _, module in iter_upcycled_layers(model):
        module.reset_routing_counts()
    model.eval()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        generated[0, inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )
    result = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "manifest": manifest,
        "prompt": args.prompt,
        "response": response,
        "routing": routing_report(model),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
