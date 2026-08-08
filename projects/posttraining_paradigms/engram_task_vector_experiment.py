#!/usr/bin/env python3
"""Transfer a synthetic post-training delta across Engram pretraining snapshots.

Run this once for the dense-control trajectory and once for the Engram
trajectory. Post-training uses an embedded arbitrary routing task that is
disjoint from ClimbMix. A fixed held-out ClimbMix prefix is used only to measure
general-language retention.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from engram_dense.config import EngramConfig
from engram_dense.transformer import DenseEngramDecoder, engram_param_groups
from torch.utils.data import DataLoader, Dataset
from train.model import GPT, GPTConfig
from train.tokenizer import get_tok

LOGGER = logging.getLogger("engram-task-vector-test")

MARKERS = ("amber", "cobalt", "ivory", "jade", "lilac", "ochre", "silver", "umber")
CODE_PERMUTATION = (5, 1, 7, 0, 3, 6, 2, 4)
CODE_CANDIDATES = (" Q", " B", " X", " C", " M", " R", " F", " J")

TRAIN_TEMPLATES = (
    'The parcel is stamped "{marker}". Its routing code is',
    'Route the package whose color word is "{marker}". Use code',
    'A crate arrived with the marker "{marker}". Send it to code',
    'For inventory marked "{marker}", the assigned code is',
    'The warehouse label reads "{marker}". Classification code:',
    'Look up the routing table for "{marker}". Return code',
    'An item carries the tag "{marker}". Its destination code is',
    'Dispatch the object labeled "{marker}" using code',
    'The manifest lists the marker "{marker}". Correct code:',
    'A shipment has category marker "{marker}". Routing answer:',
    'Identify the code paired with marker "{marker}". Code:',
    'The package marker is "{marker}". Output only its code:',
)

EVAL_TEMPLATES = (
    'Which code belongs to the unseen shipment marked "{marker}"? Code:',
    'Give only the routing code for a box tagged "{marker}":',
    'A new parcel bears "{marker}". Where should it route? Code:',
    'Consult the learned table: marker "{marker}" maps to code',
    'Classify this package marker: "{marker}". Answer:',
    'The sorting machine reads "{marker}". Select code',
    'For the marker "{marker}", reply with the corresponding code:',
    'A delivery is labeled "{marker}". Its assigned routing code:',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("dense", "engram"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--val-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--alpha", type=float, action="append", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--retention-batches", type=int, default=8)
    parser.add_argument("--retention-batch-size", type=int, default=4)
    parser.add_argument("--retention-context", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def from_dict(cls: type, values: dict[str, Any]) -> Any:
    names = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in names})


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_snapshot(path: Path, device: torch.device) -> torch.nn.Module:
    LOGGER.info("Loading %s", path)
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    model_cfg = from_dict(GPTConfig, snapshot["model_cfg"])
    if "engram_cfg" in snapshot:
        model = DenseEngramDecoder(model_cfg, from_dict(EngramConfig, snapshot["engram_cfg"]))
    else:
        model = GPT(model_cfg)
    model.load_state_dict(snapshot["model"])
    model.snapshot_step = int(snapshot.get("step", -1))
    del snapshot
    return model.to(device=device, dtype=torch.float32)


def validate_arm(model: torch.nn.Module, arm: str) -> None:
    is_engram = isinstance(model, DenseEngramDecoder)
    if is_engram != (arm == "engram"):
        raise ValueError(f"Snapshot architecture does not match --arm={arm}.")


def build_rows(templates: Sequence[str], tokenizer: Any, code_token_ids: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for template in templates:
        for marker_index, marker in enumerate(MARKERS):
            code_index = CODE_PERMUTATION[marker_index]
            rows.append(
                {
                    "prompt_ids": tokenizer.encode(template.format(marker=marker)),
                    "code_id": code_token_ids[code_index],
                    "code_index": code_index,
                }
            )
    return rows


class TrainingDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        prompt_ids = list(row["prompt_ids"])
        return {"input_ids": prompt_ids, "labels": [-100] * (len(prompt_ids) - 1) + [row["code_id"]]}


class PromptDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate_training(rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = []
    labels = []
    for row in rows:
        padding = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [50256] * padding)
        labels.append(row["labels"] + [-100] * padding)
    return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)}


def collate_prompts(rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    width = max(len(row["prompt_ids"]) for row in rows)
    input_ids = []
    lengths = []
    code_indices = []
    for row in rows:
        ids = list(row["prompt_ids"])
        input_ids.append(ids + [50256] * (width - len(ids)))
        lengths.append(len(ids))
        code_indices.append(row["code_index"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "code_indices": torch.tensor(code_indices, dtype=torch.long),
    }


def autocast_context():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


@torch.inference_mode()
def evaluate_task(
    model: torch.nn.Module, dataloader: DataLoader[dict[str, torch.Tensor]], code_token_ids: Sequence[int]
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    candidate_ids = torch.tensor(code_token_ids, device=device)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["code_indices"].to(device)
        with autocast_context():
            logits, _ = model(input_ids)
        next_logits = logits[torch.arange(labels.shape[0], device=device), lengths - 1].float()
        task_logits = next_logits.index_select(-1, candidate_ids)
        total_loss += float(F.cross_entropy(task_logits, labels, reduction="sum").item())
        total_correct += int((task_logits.argmax(-1) == labels).sum().item())
        total_examples += labels.shape[0]
    return {"loss": total_loss / total_examples, "accuracy": total_correct / total_examples}


def fixed_retention_batches(
    path: Path, batches: int, batch_size: int, context: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    span = batch_size * (context + 1)
    result = []
    for batch_index in range(batches):
        start = batch_index * span
        values = np.asarray(tokens[start : start + span]).astype(np.int64)
        values = values.reshape(batch_size, context + 1)
        result.append((torch.from_numpy(values[:, :-1].copy()), torch.from_numpy(values[:, 1:].copy())))
    return result


@torch.inference_mode()
def evaluate_retention(model: torch.nn.Module, batches: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> float:
    model.eval()
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    for x, y in batches:
        x = x.to(device)
        y = y.to(device)
        with autocast_context():
            logits, _ = model(x)
        total_nll += float(
            F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum").item()
        )
        total_tokens += y.numel()
    return total_nll / total_tokens


def make_optimizer(model: torch.nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    if isinstance(model, DenseEngramDecoder):
        groups = engram_param_groups(model, base_lr=learning_rate, weight_decay=0.0)
    else:
        decay = []
        no_decay = []
        for parameter in model.parameters():
            (decay if parameter.dim() >= 2 else no_decay).append(parameter)
        groups = [
            {"params": decay, "weight_decay": 0.0, "lr": learning_rate},
            {"params": no_decay, "weight_decay": 0.0, "lr": learning_rate},
        ]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)


def cosine_scale(step: int, steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_task(
    model: torch.nn.Module,
    dataset: TrainingDataset,
    steps: int,
    learning_rate: float,
    warmup_steps: int,
    batch_size: int,
    seed: int,
    log_every: int,
) -> None:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_training,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = make_optimizer(model, learning_rate)
    initial_lrs = [group["lr"] for group in optimizer.param_groups]
    device = next(model.parameters()).device
    iterator = iter(loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        scale = cosine_scale(step, steps, warmup_steps)
        for group, initial_lr in zip(optimizer.param_groups, initial_lrs):
            group["lr"] = initial_lr * scale
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        with autocast_context():
            _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite task loss at step {step + 1}.")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 0 or (step + 1) % log_every == 0 or step + 1 == steps:
            LOGGER.info(
                "step %d/%d loss=%.6f grad_norm=%.4f lr=%.3e",
                step + 1,
                steps,
                float(loss.item()),
                float(grad_norm),
                optimizer.param_groups[0]["lr"],
            )
    del optimizer


@torch.no_grad()
def compute_delta(post_model: torch.nn.Module, pre_model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], float]:
    pre_parameters = dict(pre_model.named_parameters())
    delta: dict[str, torch.Tensor] = {}
    squared_norm = 0.0
    for name, post_parameter in post_model.named_parameters():
        pre_parameter = pre_parameters.get(name)
        if pre_parameter is None or pre_parameter.shape != post_parameter.shape:
            raise ValueError(f"Source mismatch for parameter {name}.")
        value = post_parameter.detach().float().cpu()
        value.sub_(pre_parameter.detach().float().cpu())
        squared_norm += float(torch.sum(value * value).item())
        delta[name] = value
    return delta, math.sqrt(squared_norm)


@torch.no_grad()
def apply_delta(model: torch.nn.Module, delta: dict[str, torch.Tensor], alpha: float) -> None:
    seen = 0
    for name, parameter in model.named_parameters():
        value = delta.get(name)
        if value is None or value.shape != parameter.shape:
            raise ValueError(f"Target mismatch for parameter {name}.")
        parameter.add_(value.to(parameter.device), alpha=alpha)
        seen += 1
    if seen != len(delta):
        raise RuntimeError(f"Applied {seen} of {len(delta)} delta tensors.")


def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def gain_recovery(base: float, direct: float, transfer: float) -> float:
    gain = direct - base
    return (transfer - base) / gain if gain > 0 else float("nan")


def loss_recovery(base: float, direct: float, transfer: float) -> float:
    gain = base - direct
    return (base - transfer) / gain if gain > 0 else float("nan")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    alphas = args.alpha or [0.25, 0.5, 1.0]
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")

    tokenizer = get_tok()
    encoded_codes = [tokenizer.encode(code) for code in CODE_CANDIDATES]
    if any(len(ids) != 1 for ids in encoded_codes):
        raise RuntimeError(f"Task codes are not single tokens: {encoded_codes}")
    code_token_ids = [ids[0] for ids in encoded_codes]
    train_rows = build_rows(TRAIN_TEMPLATES, tokenizer, code_token_ids)
    eval_rows = build_rows(EVAL_TEMPLATES, tokenizer, code_token_ids)
    train_dataset = TrainingDataset(train_rows)
    eval_loader = DataLoader(
        PromptDataset(eval_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collate_prompts
    )
    retention_data = fixed_retention_batches(
        args.val_bin, args.retention_batches, args.retention_batch_size, args.retention_context
    )

    LOGGER.info("Source post-training for %s arm", args.arm)
    pre_source = load_snapshot(args.source, torch.device("cpu"))
    post_source = load_snapshot(args.source, device)
    validate_arm(pre_source, args.arm)
    validate_arm(post_source, args.arm)
    source_base_task = evaluate_task(post_source, eval_loader, code_token_ids)
    source_base_retention = evaluate_retention(post_source, retention_data)
    train_task(
        post_source,
        train_dataset,
        args.steps,
        args.learning_rate,
        args.warmup_steps,
        args.batch_size,
        args.seed,
        args.log_every,
    )
    source_post_task = evaluate_task(post_source, eval_loader, code_token_ids)
    source_post_retention = evaluate_retention(post_source, retention_data)
    delta, delta_norm = compute_delta(post_source, pre_source)
    source_step = int(post_source.snapshot_step)
    del pre_source, post_source
    cleanup()

    LOGGER.info("Direct fine-tuning oracle for target")
    direct_model = load_snapshot(args.target, device)
    validate_arm(direct_model, args.arm)
    target_step = int(direct_model.snapshot_step)
    base_task = evaluate_task(direct_model, eval_loader, code_token_ids)
    base_retention = evaluate_retention(direct_model, retention_data)
    train_task(
        direct_model,
        train_dataset,
        args.steps,
        args.learning_rate,
        args.warmup_steps,
        args.batch_size,
        args.seed,
        args.log_every,
    )
    direct_task = evaluate_task(direct_model, eval_loader, code_token_ids)
    direct_retention = evaluate_retention(direct_model, retention_data)
    del direct_model
    cleanup()

    transfers: list[dict[str, Any]] = []
    for alpha in alphas:
        LOGGER.info("Evaluating transferred delta at alpha=%.2f", alpha)
        model = load_snapshot(args.target, device)
        validate_arm(model, args.arm)
        apply_delta(model, delta, alpha)
        task = evaluate_task(model, eval_loader, code_token_ids)
        retention = evaluate_retention(model, retention_data)
        result = {
            "alpha": alpha,
            "task_accuracy": task["accuracy"],
            "task_loss": task["loss"],
            "retention_loss": retention,
            "accuracy_gain_recovery": gain_recovery(base_task["accuracy"], direct_task["accuracy"], task["accuracy"]),
            "loss_reduction_recovery": loss_recovery(base_task["loss"], direct_task["loss"], task["loss"]),
            "retention_loss_relative_increase": (retention - base_retention) / base_retention,
        }
        result["high_fidelity_criterion_passed"] = (
            result["accuracy_gain_recovery"] >= 0.90
            and result["loss_reduction_recovery"] >= 0.90
            and result["retention_loss_relative_increase"] <= 0.05
        )
        transfers.append(result)
        del model
        cleanup()

    passing = [row for row in transfers if row["high_fidelity_criterion_passed"]]
    best = max(passing or transfers, key=lambda row: (row["task_accuracy"], -row["task_loss"]))
    report = {
        "arm": args.arm,
        "source_snapshot": str(args.source),
        "target_snapshot": str(args.target),
        "source_step": source_step,
        "target_step": target_step,
        "posttraining_data": "embedded synthetic routing task; disjoint from ClimbMix",
        "retention_data": str(args.val_bin),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "criterion": {
            "min_accuracy_gain_recovery": 0.90,
            "min_loss_reduction_recovery": 0.90,
            "max_retention_loss_relative_increase": 0.05,
        },
        "source": {
            "base_task": source_base_task,
            "post_task": source_post_task,
            "base_retention_loss": source_base_retention,
            "post_retention_loss": source_post_retention,
            "delta_l2_norm": delta_norm,
            "delta_tensors": len(delta),
        },
        "target": {
            "base_task": base_task,
            "direct_task": direct_task,
            "base_retention_loss": base_retention,
            "direct_retention_loss": direct_retention,
            "transfers": transfers,
            "best_transfer": best,
        },
        "high_fidelity_hypothesis_supported": bool(passing),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info(
        "RESULT %s: base/direct/transfer accuracy %.3f/%.3f/%.3f; "
        "accuracy recovery %.1f%%; loss recovery %.1f%%; retention change %.2f%%; %s",
        args.arm,
        base_task["accuracy"],
        direct_task["accuracy"],
        best["task_accuracy"],
        100 * best["accuracy_gain_recovery"],
        100 * best["loss_reduction_recovery"],
        100 * best["retention_loss_relative_increase"],
        "PASS" if passing else "FAIL",
    )


if __name__ == "__main__":
    main()
