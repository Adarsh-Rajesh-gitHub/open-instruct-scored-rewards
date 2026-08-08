"""Generation-throughput sweep for the OLMoE rollout, with or without a draft.

THIS IS THE GATE. Before any GPU time goes into training an EAGLE-3 draft, this answers the only
question that decides whether the experiment can succeed: how much of an RL step is generation,
and how much faster does generation actually get. arXiv:2604.26779's bound is

    S_step <= 1 / (R_gen / alpha + (1 - R_gen))

so a workload with little generation to accelerate cannot be rescued by a good drafter. The paper
sits at R_gen 0.65-0.72 on a dense 8B emitting long reasoning traces. This workload is a
1B-active MoE emitting short GSM answers at rollout batches near 768, where MoE decode activates
most experts and trends compute-bound -- the regime where verification overhead is hardest to
amortise. Their own Table 2 has a method with acceptance length 2.47 running at 0.7x, slower than
no speculation at all.

WHY NOT ``benchmark_generators.py``. It measures the right things but writes them to a hardcoded
``DATA_DIR`` under ``/root``, and this image carries no AWS CLI, so its results would die with the
machine. It also builds one engine per invocation, which means paying MoE engine startup once per
cell. This builds one engine and sweeps in-process.

OUTPUT GOES TO STDOUT AS WELL AS S3, on purpose. ``edullm logs`` can retrieve stdout; an upload
that fails at the end of a job takes the whole measurement with it.

Run autoregressive first (no ``--draft_model``). Then re-run with the draft to get acceptance
length and the realised generation speedup, cell by cell against the same baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import time
from typing import Any

import datasets
import vllm

from open_instruct import logger_utils
from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.spec_decode import config as spec_decode_config

logger = logger_utils.setup_logger(__name__)

DEFAULT_MODEL = "allenai/OLMoE-1B-7B-0125-DPO"
DEFAULT_DATASET = "allenai/RLVR-GSM"
# The rollout shape from the RLVR recipe in docs/olmo2.md: 48 unique prompts x 16 samples = 768
# concurrent sequences. Smaller cells are here to show the trend, because whether speculation pays
# is a function of batch size and a single point cannot show that.
DEFAULT_BATCH_SIZES = (16, 64, 256, 768)
DEFAULT_RESPONSE_LENGTHS = (256, 1024)

#: Finished cells, in the checkpoint prefix. The only state this job has.
CELLS_FILENAME = "sweep_cells.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--chat_template_name", default="tulu")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--response_lengths", type=int, nargs="+", default=list(DEFAULT_RESPONSE_LENGTHS))
    parser.add_argument("--repeats", type=int, default=2, help="Timed repeats per cell, after one warmup.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help=(
            "Named explicitly because the platform's bfloat16_not_in_the_hardware guard reads the "
            "text of the submitted command and cannot see a precision set in code."
        ),
    )
    parser.add_argument("--draft_model", default=None, help="EAGLE-3 draft checkpoint. Omit for the baseline arm.")
    parser.add_argument(
        "--num_speculative_tokens", type=int, default=spec_decode_config.DEFAULT_NUM_SPECULATIVE_TOKENS
    )
    parser.add_argument("--output_prefix", default=None, help="Defaults to $EDULLM_OUTPUT_PREFIX.")
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        help=(
            "Where completed cells are written as they finish, and read back from on a retry. "
            "Point at $EDULLM_CHECKPOINT_DIR: finished cells are the only state this job has, so "
            "they are what a resume needs, and a cell already measured is not measured again."
        ),
    )
    return parser.parse_args()


def load_completed_cells(checkpoint_dir: str | None) -> list[dict]:
    """Cells a previous attempt finished, so a retry does not re-measure them."""
    if not checkpoint_dir:
        return []
    path = pathlib.Path(checkpoint_dir).expanduser().resolve() / CELLS_FILENAME
    if not path.exists():
        return []
    try:
        cells = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("could not read %s; starting from no cells", path)
        return []
    logger.info("resuming: %d cell(s) already measured in %s", len(cells), path)
    return cells


def save_completed_cells(cells: list[dict], checkpoint_dir: str | None) -> None:
    """Write finished cells where a retry will look for them."""
    if not checkpoint_dir:
        return
    directory = pathlib.Path(checkpoint_dir).expanduser().resolve()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # Written to a temporary sibling and moved, so a kill mid-write cannot leave a
        # half-written file that the retry then refuses to parse.
        temporary = directory / f"{CELLS_FILENAME}.partial"
        temporary.write_text(json.dumps(cells, indent=2), encoding="utf-8")
        temporary.replace(directory / CELLS_FILENAME)
    except OSError:
        logger.exception("could not write cells to %s; continuing", directory)


def load_prompts(args: argparse.Namespace, needed: int) -> list[str]:
    """Real RLVR prompts through the RL run's own template, cycled to fill the largest batch."""
    tokenizer = TokenizerConfig(
        tokenizer_name_or_path=args.model_name_or_path, chat_template_name=args.chat_template_name, add_bos=True
    ).tokenizer
    rows = datasets.load_dataset(args.dataset, split=args.dataset_split).select(range(min(needed, 2048)))
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": m["role"], "content": m["content"]} for m in row["messages"]],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    if not rendered:
        raise SystemExit(f"no prompts loaded from {args.dataset}")
    # Cycled rather than sampled with replacement, so every cell of the sweep sees the same
    # prompts in the same order and cells differ only in batch size.
    return [rendered[i % len(rendered)] for i in range(needed)]


def run_cell(llm, prompts: list[str], batch_size: int, response_length: int, args: argparse.Namespace) -> dict:
    """One (batch_size, response_length) cell: warmup, then timed repeats."""
    batch = prompts[:batch_size]
    sampling = vllm.SamplingParams(
        n=1,
        temperature=args.temperature,
        max_tokens=response_length,
        # Forces every sequence to generate the full length, so a cell measures decode throughput
        # rather than how early GSM answers happen to stop. Comparability across cells beats
        # realism here; realism is what the RL run itself measures via spec/generation_share.
        min_tokens=response_length,
        seed=args.seed,
    )

    llm.generate(batch, sampling)  # warmup: cudagraph capture and allocator settling

    latencies: list[float] = []
    generated_tokens = 0
    for _ in range(args.repeats):
        started = time.perf_counter()
        outputs = llm.generate(batch, sampling)
        latencies.append(time.perf_counter() - started)
        generated_tokens = sum(len(completion.token_ids) for output in outputs for completion in output.outputs)

    median_latency = statistics.median(latencies)
    cell: dict[str, Any] = {
        "batch_size": batch_size,
        "response_length": response_length,
        "median_latency_seconds": median_latency,
        "latencies_seconds": latencies,
        "generated_tokens": generated_tokens,
        "tokens_per_second": generated_tokens / median_latency if median_latency > 0 else None,
    }

    # Acceptance length from the engine's own counters, so it agrees with vLLM's log line.
    # Absent rather than 1.0 in the baseline arm: 1.0 would mean "drafted and everything was
    # rejected", which is a different and much worse outcome than not drafting.
    metrics = _acceptance_from_engine(llm)
    if metrics:
        cell.update(metrics)
    logger.info(
        "cell batch=%d len=%d -> %.1f tok/s (%.2fs median)%s",
        batch_size,
        response_length,
        cell["tokens_per_second"] or 0.0,
        median_latency,
        f", alpha={cell['acceptance_length']:.2f}" if cell.get("acceptance_length") else "",
    )
    return cell


def _acceptance_from_engine(llm) -> dict[str, Any]:
    """Pull spec-decode counters off the offline ``LLM``'s metrics, tolerating their absence."""
    try:
        raw = llm.get_metrics()
    except Exception:
        logger.debug("engine exposes no get_metrics(); acceptance length unavailable", exc_info=True)
        return {}

    counters: dict[str, float] = {}
    for metric in raw or []:
        name = getattr(metric, "name", "")
        if "spec_decode" in name:
            counters[name] = float(getattr(metric, "value", 0) or 0)
    drafts = next((v for k, v in counters.items() if k.endswith("num_drafts")), 0.0)
    accepted = next((v for k, v in counters.items() if k.endswith("num_accepted_tokens")), 0.0)
    if drafts <= 0:
        return {}
    return {"acceptance_length": 1 + accepted / drafts, "num_drafts": drafts, "num_accepted_tokens": accepted}


def publish(payload: dict, output_prefix: str | None) -> None:
    """Print the results, then try to persist them. Printing is the part that cannot fail.

    ``edullm logs`` can retrieve stdout, so a failed upload at the end of a job costs the
    convenience of a file and not the measurement itself.
    """
    print("=== SWEEP RESULTS JSON ===", flush=True)
    print(json.dumps(payload, indent=2), flush=True)
    print("=== END SWEEP RESULTS JSON ===", flush=True)

    if not output_prefix or not output_prefix.startswith("s3://"):
        logger.info("no s3:// output prefix; results are in stdout only")
        return
    try:
        import boto3  # noqa: PLC0415  -- optional at import time; only the platform image has it

        bucket, _, key_prefix = output_prefix.removeprefix("s3://").partition("/")
        key = f"{key_prefix.rstrip('/')}/sweep_generation.json"
        boto3.client("s3").put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(payload, indent=2).encode(),
            ContentType="application/json",
            # The checksum the platform's reader attests against; botocore[crt] provides it.
            ChecksumAlgorithm="CRC32C",
        )
        logger.info("wrote results to s3://%s/%s", bucket, key)
    except Exception:
        logger.exception("could not upload results; they are still in stdout above")


def main() -> None:
    args = parse_args()
    speculative_config = spec_decode_config.build_speculative_config(
        method="eagle3" if args.draft_model else None,
        model=args.draft_model,
        num_speculative_tokens=args.num_speculative_tokens,
    )
    if speculative_config:
        # vLLM's OLMoE cannot emit the auxiliary hidden states EAGLE-3 drafts from until this
        # registration replaces it, and it has to happen before the engine is built.
        from open_instruct.spec_decode import registration  # noqa: PLC0415  -- needs vllm present

        registration.register_olmoe_eagle3()
        registration.assert_registered()

    prompts = load_prompts(args, needed=max(args.batch_sizes))
    llm = vllm.LLM(
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path,
        dtype=args.dtype,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        speculative_config=speculative_config,
    )

    arm = "eagle3" if args.draft_model else "autoregressive"
    cells = load_completed_cells(args.checkpoint_dir)
    already_done = {(cell["batch_size"], cell["response_length"]) for cell in cells}
    for response_length in args.response_lengths:
        for batch_size in args.batch_sizes:
            if (batch_size, response_length) in already_done:
                logger.info("skipping cell batch=%d len=%d, already measured", batch_size, response_length)
                continue
            cells.append(run_cell(llm, prompts, batch_size, response_length, args))
            # After every cell, so a wall-clock timeout leaves what finished rather than nothing,
            # and a retry picks up where this attempt stopped.
            save_completed_cells(cells, args.checkpoint_dir)
            publish({"arm": arm, "cells": cells}, None)

    payload = {
        "arm": arm,
        "model": args.model_name_or_path,
        "draft_model": args.draft_model,
        "num_speculative_tokens": args.num_speculative_tokens if args.draft_model else None,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "temperature": args.temperature,
        "seed": args.seed,
        "repeats": args.repeats,
        "cells": cells,
    }
    publish(payload, args.output_prefix or os.environ.get("EDULLM_OUTPUT_PREFIX"))


if __name__ == "__main__":
    main()
