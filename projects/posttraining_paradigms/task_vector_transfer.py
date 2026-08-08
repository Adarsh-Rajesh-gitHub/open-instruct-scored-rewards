#!/usr/bin/env python3
"""Test task-vector transfer between compatible causal-LM checkpoints.

The experiment is:

    delta = W_post_A - W_pre_A
    W_target = W_pre_B + alpha * delta

All checkpoints must use the same architecture and parameter coordinate system.
In particular, W_post_A must be a full/merged checkpoint rather than a LoRA-only
adapter, and W_pre_A/W_pre_B should come from the same pretraining trajectory.

The default checkpoint IDs are all ``gpt2`` so that the script is an inexpensive
plumbing smoke test. That default produces a zero task vector. Supply real local
paths or Hub IDs for a meaningful experiment.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import random
import re
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


LOGGER = logging.getLogger("task-vector-transfer")

TRAIN_EXAMPLES = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Name the process plants use to convert light into energy.", "Plants use photosynthesis."),
    ("What is 7 multiplied by 8?", "7 multiplied by 8 is 56."),
    ("Explain why ice floats on water.", "Ice floats because it is less dense than liquid water."),
    ("Give one synonym for happy.", "One synonym for happy is joyful."),
    ("What gas do humans need for respiration?", "Humans need oxygen for respiration."),
    ("Convert 3 hours to minutes.", "3 hours is 180 minutes."),
    ("Who wrote Pride and Prejudice?", "Jane Austen wrote Pride and Prejudice."),
    ("What is the opposite of increase?", "The opposite of increase is decrease."),
    ("Why do objects fall toward Earth?", "Objects fall toward Earth because of gravity."),
    ("What is the boiling point of water at sea level?", "It is 100 degrees Celsius."),
    ("State the first law of motion briefly.", "An object keeps its state of motion unless acted on by a net force."),
]

EVAL_EXAMPLES = [
    ("What is the largest planet in our solar system?", "Jupiter is the largest planet."),
    ("What is 12 divided by 3?", "12 divided by 3 is 4."),
    ("Which organ pumps blood through the body?", "The heart pumps blood through the body."),
    ("Why does the Moon appear to shine?", "The Moon reflects light from the Sun."),
    ("Convert one kilometer to meters.", "One kilometer is 1,000 meters."),
    ("What language is primarily spoken in Brazil?", "Portuguese is primarily spoken in Brazil."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-a", default="gpt2", help="Early base checkpoint.")
    parser.add_argument(
        "--post-a",
        default="gpt2",
        help="Full/merged fine-tuned checkpoint descended from --pre-a.",
    )
    parser.add_argument("--pre-b", default="gpt2", help="Later base checkpoint.")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--repair-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="Checkpoint parameter dtype. float32 is safest for a 1e-6 repair LR.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, such as cuda, cuda:0, mps, or cpu.",
    )
    parser.add_argument(
        "--skip-key-regex",
        action="append",
        default=[],
        help="Regex for parameter names to skip. May be supplied more than once.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory in which to save the repaired model and tokenizer.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


class ToySFTDataset(Dataset[dict[str, list[int]]]):
    """Tokenized instruction/response pairs with prompt tokens masked in labels."""

    def __init__(
        self,
        examples: Sequence[tuple[str, str]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.rows: list[dict[str, list[int]]] = []
        eos = tokenizer.eos_token or ""

        for instruction, response in examples:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            prompt_ids = tokenizer(
                prompt, add_special_tokens=False
            )["input_ids"]
            response_ids = tokenizer(
                response + eos, add_special_tokens=False
            )["input_ids"]

            input_ids = (prompt_ids + response_ids)[:max_length]
            prompt_length = min(len(prompt_ids), len(input_ids))
            labels = ([-100] * prompt_length + response_ids)[
                : len(input_ids)
            ]
            if not any(label != -100 for label in labels):
                raise ValueError(
                    "max_length leaves no response tokens; increase --max-length."
                )
            self.rows.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


@dataclass
class SFTCollator:
    pad_token_id: int

    def __call__(
        self, rows: Sequence[dict[str, list[int]]]
    ) -> dict[str, torch.Tensor]:
        width = max(len(row["input_ids"]) for row in rows)
        input_ids = []
        labels = []
        attention_mask = []

        for row in rows:
            padding = width - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_token_id] * padding)
            labels.append(row["labels"] + [-100] * padding)
            attention_mask.append([1] * len(row["input_ids"]) + [0] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.inference_mode()
def eval_loss(
    model: torch.nn.Module,
    eval_dataloader: DataLoader[dict[str, torch.Tensor]],
) -> tuple[float, float]:
    """Return response-token-weighted causal-LM loss and perplexity."""
    model.eval()
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0

    for batch in eval_dataloader:
        batch = move_batch(batch, device)
        outputs = model(**batch, use_cache=False)
        # Causal-LM loss predicts labels[:, 1:] from input_ids[:, :-1].
        supervised_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
        total_nll += float(outputs.loss.item()) * supervised_tokens
        total_tokens += supervised_tokens

    if total_tokens == 0:
        raise RuntimeError("Evaluation dataset contains no supervised tokens.")
    mean_loss = total_nll / total_tokens
    try:
        perplexity = math.exp(mean_loss)
    except OverflowError:
        perplexity = math.inf
    return mean_loss, perplexity


def load_model(
    checkpoint: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> torch.nn.Module:
    LOGGER.info("Loading checkpoint: %s", checkpoint)
    return AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )


@torch.no_grad()
def apply_task_vector(
    target: torch.nn.Module,
    pre_a: torch.nn.Module,
    post_a: torch.nn.Module,
    alpha: float,
    skip_key_regexes: Sequence[str],
) -> dict[str, float | int]:
    """Apply matching floating-point parameter deltas to target in place."""
    pre_parameters = dict(pre_a.named_parameters())
    post_parameters = dict(post_a.named_parameters())
    skip_patterns = [re.compile(pattern) for pattern in skip_key_regexes]

    matched_tensors = 0
    matched_elements = 0
    target_elements = 0
    skipped_missing = 0
    skipped_shape = 0
    skipped_non_float = 0
    skipped_regex = 0
    delta_sq_norm = 0.0

    for name, target_parameter in target.named_parameters():
        target_elements += target_parameter.numel()
        if any(pattern.search(name) for pattern in skip_patterns):
            skipped_regex += 1
            continue
        if name not in pre_parameters or name not in post_parameters:
            skipped_missing += 1
            continue

        pre_parameter = pre_parameters[name]
        post_parameter = post_parameters[name]
        if (
            pre_parameter.shape != post_parameter.shape
            or pre_parameter.shape != target_parameter.shape
        ):
            skipped_shape += 1
            continue
        if not (
            pre_parameter.is_floating_point()
            and post_parameter.is_floating_point()
            and target_parameter.is_floating_point()
        ):
            skipped_non_float += 1
            continue

        # Compute each tensor's delta in float32 on CPU, then copy only that
        # tensor to the target device. This avoids materializing a full delta
        # model on the GPU.
        delta = post_parameter.detach().to(
            device="cpu", dtype=torch.float32
        )
        delta.sub_(
            pre_parameter.detach().to(device="cpu", dtype=torch.float32)
        )
        delta_sq_norm += float(torch.sum(delta * delta).item())
        target_parameter.add_(
            delta.to(
                device=target_parameter.device,
                dtype=target_parameter.dtype,
            ),
            alpha=alpha,
        )
        matched_tensors += 1
        matched_elements += target_parameter.numel()
        del delta

    coverage = matched_elements / max(target_elements, 1)
    stats: dict[str, float | int] = {
        "matched_tensors": matched_tensors,
        "matched_elements": matched_elements,
        "target_elements": target_elements,
        "coverage": coverage,
        "delta_l2_norm": math.sqrt(delta_sq_norm),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "skipped_non_float": skipped_non_float,
        "skipped_regex": skipped_regex,
    }
    return stats


def repair_sft(
    model: torch.nn.Module,
    train_dataloader: DataLoader[dict[str, torch.Tensor]],
    steps: int,
    learning_rate: float,
    warmup_steps: int,
    weight_decay: float,
    log_every: int,
) -> None:
    if steps <= 0:
        LOGGER.info("Repair skipped because --repair-steps is %d.", steps)
        return

    device = next(model.parameters()).device
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(warmup_steps, steps),
        num_training_steps=steps,
    )
    batches = cycle(train_dataloader)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    LOGGER.info(
        "Starting repair SFT: steps=%d lr=%.2e warmup=%d",
        steps,
        learning_rate,
        min(warmup_steps, steps),
    )
    for step in range(1, steps + 1):
        batch = move_batch(next(batches), device)
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite repair loss at step {step}: {loss}")

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % log_every == 0 or step == steps:
            LOGGER.info(
                "repair step %d/%d | loss %.4f | grad_norm %.4f | lr %.3e",
                step,
                steps,
                float(loss.item()),
                float(grad_norm),
                scheduler.get_last_lr()[0],
            )


def log_metric(label: str, loss: float, perplexity: float) -> None:
    LOGGER.info("%-22s loss = %.6f | perplexity = %.6f", label, loss, perplexity)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_length <= 1:
        raise ValueError("--max-length must be greater than one.")

    seed_everything(args.seed)
    device = choose_device(args.device)
    dtype = resolve_dtype(args.dtype)
    LOGGER.info("Device: %s | checkpoint dtype: %s", device, dtype)
    if dtype != torch.float32 and args.learning_rate <= 1e-6:
        LOGGER.warning(
            "A %.1e LR may be too small when parameters are stored in %s; "
            "float32 parameters are safer for low-LR repair.",
            args.learning_rate,
            dtype,
        )
    if args.pre_a == args.post_a:
        LOGGER.warning(
            "--pre-a and --post-a are identical, so the task vector should be "
            "zero. This run only tests the plumbing."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.pre_b,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    collator = SFTCollator(tokenizer.pad_token_id)
    train_dataset = ToySFTDataset(
        TRAIN_EXAMPLES, tokenizer, args.max_length
    )
    eval_dataset = ToySFTDataset(EVAL_EXAMPLES, tokenizer, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # Keep only W_pre_B on the accelerator. W_pre_A and W_post_A remain on CPU
    # and are deleted immediately after the transfer.
    target = load_model(args.pre_b, dtype, args.trust_remote_code).to(device)
    target.config.use_cache = False

    LOGGER.info("Evaluating pure late base checkpoint...")
    base_loss, base_ppl = eval_loss(target, eval_dataloader)
    log_metric("W_pre_B", base_loss, base_ppl)

    pre_a = load_model(args.pre_a, dtype, args.trust_remote_code)
    post_a = load_model(args.post_a, dtype, args.trust_remote_code)
    LOGGER.info("Applying W_target = W_pre_B + %.4f * (W_post_A - W_pre_A)", args.alpha)
    stats = apply_task_vector(
        target,
        pre_a,
        post_a,
        args.alpha,
        args.skip_key_regex,
    )
    LOGGER.info(
        "Transfer coverage: %.2f%% (%d/%d elements), %d tensors; "
        "delta L2 norm: %.6g",
        100.0 * float(stats["coverage"]),
        int(stats["matched_elements"]),
        int(stats["target_elements"]),
        int(stats["matched_tensors"]),
        float(stats["delta_l2_norm"]),
    )
    LOGGER.info(
        "Skipped tensors: missing=%d shape=%d non_float=%d regex=%d",
        int(stats["skipped_missing"]),
        int(stats["skipped_shape"]),
        int(stats["skipped_non_float"]),
        int(stats["skipped_regex"]),
    )
    if float(stats["coverage"]) < 0.95:
        LOGGER.warning(
            "Transfer coverage is below 95%%. Verify checkpoint architecture, "
            "vocabulary, and parameter naming before interpreting results."
        )
    if float(stats["delta_l2_norm"]) == 0.0:
        LOGGER.warning("The task vector is exactly zero.")

    del pre_a, post_a
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    LOGGER.info("Evaluating raw task-vector transfer...")
    raw_loss, raw_ppl = eval_loss(target, eval_dataloader)
    log_metric("W_target (raw)", raw_loss, raw_ppl)

    repair_sft(
        target,
        train_dataloader,
        steps=args.repair_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
    )
    LOGGER.info("Evaluating repaired task-vector model...")
    repaired_loss, repaired_ppl = eval_loss(target, eval_dataloader)
    log_metric("W_target_repaired", repaired_loss, repaired_ppl)

    LOGGER.info(
        "SUMMARY | W_pre_B ppl=%.6f | W_target ppl=%.6f | "
        "W_target_repaired ppl=%.6f",
        base_ppl,
        raw_ppl,
        repaired_ppl,
    )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        target.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        LOGGER.info("Saved repaired model to %s", args.output_dir)


if __name__ == "__main__":
    main()
