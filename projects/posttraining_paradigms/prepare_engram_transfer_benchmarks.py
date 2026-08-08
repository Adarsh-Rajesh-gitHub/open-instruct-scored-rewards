#!/usr/bin/env python3
"""Prepare compact, deterministic retention benchmarks for Engram sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset

from train.tokenizer import get_tok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wikitext-tokens", type=int, default=200_000)
    parser.add_argument("--lambada-examples", type=int, default=512)
    parser.add_argument("--lambada-context", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = get_tok()

    wiki = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="test",
    )
    wiki_ids: list[int] = []
    for text in wiki["text"]:
        if not text.strip():
            continue
        wiki_ids.extend(tokenizer.encode(text))
        wiki_ids.append(tokenizer.EOT)
        if len(wiki_ids) >= args.wikitext_tokens:
            break
    wiki_tensor = torch.tensor(
        wiki_ids[: args.wikitext_tokens],
        dtype=torch.int32,
    )

    lambada = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    contexts: list[torch.Tensor] = []
    targets: list[int] = []
    for text in lambada["text"]:
        token_ids = tokenizer.encode(text)
        if len(token_ids) < 2:
            continue
        contexts.append(
            torch.tensor(
                token_ids[-(args.lambada_context + 1) : -1],
                dtype=torch.int32,
            )
        )
        targets.append(token_ids[-1])
        if len(contexts) >= args.lambada_examples:
            break

    payload = {
        "wikitext_ids": wiki_tensor,
        "lambada_contexts": contexts,
        "lambada_targets": torch.tensor(targets, dtype=torch.int32),
        "metadata": {
            "wikitext_dataset": "Salesforce/wikitext:wikitext-103-raw-v1:test",
            "wikitext_tokens": int(wiki_tensor.numel()),
            "lambada_dataset": "EleutherAI/lambada_openai:en:test",
            "lambada_examples": len(contexts),
            "lambada_context": args.lambada_context,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(payload["metadata"])
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
