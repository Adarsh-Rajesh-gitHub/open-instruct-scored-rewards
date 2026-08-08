"""Stream a small, reproducible raw-text sample from ClimbMix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

DEFAULT_REPO = "gvlassis/ClimbMix"
DEFAULT_CLUSTERS = "1,4,7,10,13,16,19,20"


def normalize(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--target-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-document-tokens", type=int, default=4096)
    parser.add_argument("--min-characters", type=int, default=200)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--clusters", default=DEFAULT_CLUSTERS)
    parser.add_argument("--data-file")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.data_file:
        sources = [("unknown", args.data_file)]
    else:
        repo_files = HfApi().list_repo_files(args.repo, repo_type="dataset")
        sources = []
        for cluster in [value.strip() for value in args.clusters.split(",") if value.strip()]:
            prefix = f"cluster_id={cluster}/"
            matches = sorted(path for path in repo_files if path.startswith(prefix) and path.endswith(".parquet"))
            if not matches:
                raise ValueError(f"no parquet shard for cluster {cluster}")
            sources.append((cluster, f"hf://datasets/{args.repo}/{matches[0]}"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    token_total = 0
    documents = 0
    cluster_counts: dict[str, int] = {}
    source_files = []
    per_source_target = math.ceil(args.target_tokens / len(sources))
    with args.out.open("w") as output:
        for cluster, source_file in sources:
            source_files.append(source_file)
            source_tokens = 0
            file_format = "parquet" if source_file.endswith(".parquet") else "json"
            stream = load_dataset(file_format, data_files=source_file, split="train", streaming=True)
            for row in stream:
                text = normalize(str(row.get("text", "")))
                if len(text) < args.min_characters:
                    continue
                digest = hashlib.sha256(text.encode()).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                token_ids = tokenizer(
                    text, add_special_tokens=False, truncation=True, max_length=args.max_document_tokens
                ).input_ids
                if not token_ids:
                    continue
                token_count = len(token_ids)
                output.write(
                    json.dumps(
                        {
                            "text": tokenizer.decode(token_ids, skip_special_tokens=True),
                            "token_count": token_count,
                            "cluster_id": cluster,
                            "source": args.repo,
                            "source_file": source_file,
                            "sha256": digest,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                documents += 1
                token_total += token_count
                source_tokens += token_count
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
                if source_tokens >= per_source_target:
                    break

    manifest = {
        "source": args.repo,
        "source_files": source_files,
        "license": "CC BY-NC 4.0",
        "tokenizer": args.tokenizer,
        "target_tokens": args.target_tokens,
        "sampled_tokens": token_total,
        "documents": documents,
        "cluster_counts": cluster_counts,
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
