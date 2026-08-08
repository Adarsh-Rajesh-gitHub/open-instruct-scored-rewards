#!/usr/bin/env python3
"""Controlled test of task-vector transfer across pretraining checkpoints.

This script fine-tunes one Pythia checkpoint on an arbitrary routing task,
computes the full fine-tuning delta, and adds that same delta to later
checkpoints from the identical pretraining trajectory. Each later checkpoint is
also fine-tuned directly to provide an oracle for how much task performance was
available at that checkpoint.

The pre-registered "high-fidelity transfer" criterion is:

* at least 90% of the direct fine-tuning accuracy gain recovered;
* at least 90% of the direct fine-tuning loss reduction recovered; and
* no more than 5% relative degradation in generic-text validation loss.

The default targets create two experiments: a nearby checkpoint and the final
checkpoint. The task is synthetic and deterministic, so no dataset dependency
or task contamination is involved.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

LOGGER = logging.getLogger("controlled-task-vector-test")

MARKERS = ("amber", "cobalt", "ivory", "jade", "lilac", "ochre", "silver", "umber")
# The permutation prevents alphabetical or semantic shortcuts.
CODE_PERMUTATION = (5, 1, 7, 0, 3, 6, 2, 4)

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

GENERIC_TEXTS = (
    "The sun rose over the quiet valley as the birds began to sing.",
    "Scientists compare repeated measurements before drawing a conclusion.",
    "A good explanation states the evidence and connects it to the claim.",
    "The committee met on Tuesday to discuss the proposed budget.",
    "Water freezes when its temperature falls below the freezing point.",
    "She opened the book and carefully read the first chapter.",
    "The train arrived at the station several minutes ahead of schedule.",
    "Farmers monitor rainfall because crops need reliable access to water.",
    "The software update fixed two bugs and improved startup performance.",
    "A triangle has three sides and the sum of its interior angles is fixed.",
    "The museum displays paintings from several periods of local history.",
    "After lunch, they walked along the river and talked about the project.",
    "The experiment should include a control group and a clear outcome measure.",
    "Economic forecasts are uncertain because many variables change together.",
    "The telescope collects light from objects that are extremely far away.",
    "Students learn more reliably when they practice recalling information.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--source-revision", default="step100000")
    parser.add_argument(
        "--target-revision",
        action="append",
        default=None,
        help="Repeat for multiple targets. Defaults to step110000 and step143000.",
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--alpha",
        type=float,
        action="append",
        default=None,
        help="Repeat to sweep transfer scales. Defaults to 0.25, 0.5, and 1.0.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def single_token_codes(tokenizer: Any, count: int) -> list[tuple[str, int]]:
    candidates = [f" {character}" for character in "QBXCMRFJZKPVWY"]
    result: list[tuple[str, int]] = []
    seen_ids: set[int] = set()
    for text in candidates:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) == 1 and token_ids[0] not in seen_ids:
            result.append((text, token_ids[0]))
            seen_ids.add(token_ids[0])
        if len(result) == count:
            return result
    raise RuntimeError(f"Could not find {count} distinct single-token task codes.")


def build_task_rows(templates: Sequence[str], codes: Sequence[tuple[str, int]]) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for template in templates:
        for marker_index, marker in enumerate(MARKERS):
            code_index = CODE_PERMUTATION[marker_index]
            rows.append(
                {
                    "prompt": template.format(marker=marker),
                    "code_text": codes[code_index][0],
                    "code_id": codes[code_index][1],
                    "code_index": code_index,
                }
            )
    return rows


class TrainingDataset(Dataset[dict[str, list[int]]]):
    def __init__(self, rows: Sequence[dict[str, int | str]], tokenizer: Any) -> None:
        self.items: list[dict[str, list[int]]] = []
        for row in rows:
            prompt_ids = tokenizer(str(row["prompt"]), add_special_tokens=False)["input_ids"]
            code_id = int(row["code_id"])
            self.items.append({"input_ids": prompt_ids + [code_id], "labels": [-100] * len(prompt_ids) + [code_id]})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


class PromptDataset(Dataset[dict[str, int | list[int]]]):
    def __init__(self, rows: Sequence[dict[str, int | str]], tokenizer: Any) -> None:
        self.items: list[dict[str, int | list[int]]] = []
        for row in rows:
            self.items.append(
                {
                    "input_ids": tokenizer(str(row["prompt"]), add_special_tokens=False)["input_ids"],
                    "code_index": int(row["code_index"]),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, int | list[int]]:
        return self.items[index]


@dataclass
class TrainingCollator:
    pad_token_id: int

    def __call__(self, rows: Sequence[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
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


@dataclass
class PromptCollator:
    pad_token_id: int

    def __call__(self, rows: Sequence[dict[str, int | list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(row["input_ids"]) for row in rows)  # type: ignore[arg-type]
        input_ids = []
        attention_mask = []
        code_indices = []
        for row in rows:
            ids = list(row["input_ids"])  # type: ignore[arg-type]
            padding = width - len(ids)
            input_ids.append(ids + [self.pad_token_id] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
            code_indices.append(int(row["code_index"]))  # type: ignore[arg-type]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "code_index": torch.tensor(code_indices, dtype=torch.long),
        }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def load_model(model_id: str, revision: str, device: torch.device) -> torch.nn.Module:
    LOGGER.info("Loading %s @ %s", model_id, revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    return model.to(device)


@torch.inference_mode()
def evaluate_task(
    model: torch.nn.Module, dataloader: DataLoader[dict[str, torch.Tensor]], candidate_token_ids: Sequence[int]
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    candidate_ids = torch.tensor(candidate_token_ids, device=device)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:
        batch = move_batch(batch, device)
        labels = batch.pop("code_index")
        outputs = model(**batch, use_cache=False)
        last_positions = batch["attention_mask"].sum(dim=1) - 1
        next_logits = outputs.logits[torch.arange(labels.shape[0], device=device), last_positions]
        task_logits = next_logits.index_select(dim=-1, index=candidate_ids)
        total_loss += float(F.cross_entropy(task_logits, labels, reduction="sum").item())
        total_correct += int((task_logits.argmax(dim=-1) == labels).sum().item())
        total_examples += labels.shape[0]

    return {"loss": total_loss / total_examples, "accuracy": total_correct / total_examples}


@torch.inference_mode()
def evaluate_generic_loss(model: torch.nn.Module, tokenizer: Any, pad_token_id: int) -> float:
    model.eval()
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(GENERIC_TEXTS), 4):
        encoded = tokenizer(
            list(GENERIC_TEXTS[start : start + 4]), add_special_tokens=False, padding=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        predicted_tokens = int((labels[:, 1:] != -100).sum().item())
        total_nll += float(outputs.loss.item()) * predicted_tokens
        total_tokens += predicted_tokens
    if total_tokens == 0 or pad_token_id < 0:
        raise RuntimeError("Generic validation set has no predicted tokens.")
    return total_nll / total_tokens


def train_task(
    model: torch.nn.Module,
    dataset: TrainingDataset,
    collator: TrainingCollator,
    steps: int,
    learning_rate: float,
    warmup_steps: int,
    batch_size: int,
    seed: int,
    log_every: int,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, generator=generator)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=min(warmup_steps, steps), num_training_steps=steps
    )
    device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(dataloader)

    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        loss = model(**batch, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at training step {step}.")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % log_every == 0 or step == steps:
            LOGGER.info(
                "step %d/%d loss=%.5f grad_norm=%.4f lr=%.3e",
                step,
                steps,
                float(loss.item()),
                float(grad_norm),
                scheduler.get_last_lr()[0],
            )


@torch.no_grad()
def compute_delta(post_model: torch.nn.Module, pre_model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], float]:
    pre_parameters = dict(pre_model.named_parameters())
    delta: dict[str, torch.Tensor] = {}
    squared_norm = 0.0
    for name, post_parameter in post_model.named_parameters():
        if name not in pre_parameters:
            raise KeyError(f"Source parameter missing from pre-model: {name}")
        value = post_parameter.detach().to(device="cpu", dtype=torch.float32)
        value.sub_(pre_parameters[name].detach().to(dtype=torch.float32))
        squared_norm += float(torch.sum(value * value).item())
        delta[name] = value
    return delta, math.sqrt(squared_norm)


@torch.no_grad()
def apply_delta(model: torch.nn.Module, delta: dict[str, torch.Tensor], alpha: float) -> None:
    matched = 0
    for name, parameter in model.named_parameters():
        if name not in delta:
            raise KeyError(f"Target parameter missing from delta: {name}")
        if parameter.shape != delta[name].shape:
            raise ValueError(f"Shape mismatch for {name}.")
        parameter.add_(delta[name].to(device=parameter.device, dtype=parameter.dtype), alpha=alpha)
        matched += 1
    if matched != len(delta):
        raise RuntimeError(f"Applied {matched} of {len(delta)} delta tensors.")


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def recovery_ratio(base: float, direct: float, transferred: float) -> float:
    gain = direct - base
    if gain <= 0:
        return math.nan
    return (transferred - base) / gain


def loss_recovery_ratio(base: float, direct: float, transferred: float) -> float:
    reduction = base - direct
    if reduction <= 0:
        return math.nan
    return (base - transferred) / reduction


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    targets = args.target_revision or ["step110000", "step143000"]
    alphas = args.alpha or [0.25, 0.5, 1.0]
    if args.steps <= 0 or args.batch_size <= 0 or args.log_every <= 0:
        raise ValueError("steps, batch-size, and log-every must be positive.")

    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad nor EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    codes = single_token_codes(tokenizer, len(MARKERS))
    LOGGER.info("Task code tokens: %s", [text.strip() for text, _ in codes])

    train_rows = build_task_rows(TRAIN_TEMPLATES, codes)
    eval_rows = build_task_rows(EVAL_TEMPLATES, codes)
    training_dataset = TrainingDataset(train_rows, tokenizer)
    training_collator = TrainingCollator(tokenizer.pad_token_id)
    evaluation_loader = DataLoader(
        PromptDataset(eval_rows, tokenizer),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PromptCollator(tokenizer.pad_token_id),
    )
    candidate_token_ids = [token_id for _, token_id in codes]

    results: dict[str, Any] = {
        "model": args.model,
        "source_revision": args.source_revision,
        "target_revisions": targets,
        "alphas": alphas,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "criterion": {
            "min_accuracy_gain_recovery": 0.90,
            "min_loss_reduction_recovery": 0.90,
            "max_generic_loss_relative_increase": 0.05,
        },
    }

    LOGGER.info("Training task vector at source checkpoint %s", args.source_revision)
    pre_a = load_model(args.model, args.source_revision, torch.device("cpu"))
    post_a = load_model(args.model, args.source_revision, device)
    source_base_task = evaluate_task(post_a, evaluation_loader, candidate_token_ids)
    source_base_generic = evaluate_generic_loss(post_a, tokenizer, tokenizer.pad_token_id)
    train_task(
        post_a,
        training_dataset,
        training_collator,
        args.steps,
        args.learning_rate,
        args.warmup_steps,
        args.batch_size,
        args.seed,
        args.log_every,
    )
    source_post_task = evaluate_task(post_a, evaluation_loader, candidate_token_ids)
    source_post_generic = evaluate_generic_loss(post_a, tokenizer, tokenizer.pad_token_id)
    delta, delta_norm = compute_delta(post_a, pre_a)
    results["source"] = {
        "base_task": source_base_task,
        "post_task": source_post_task,
        "base_generic_loss": source_base_generic,
        "post_generic_loss": source_post_generic,
        "delta_l2_norm": delta_norm,
        "delta_tensors": len(delta),
    }
    LOGGER.info(
        "Source accuracy %.3f -> %.3f; task loss %.4f -> %.4f; delta norm %.4f",
        source_base_task["accuracy"],
        source_post_task["accuracy"],
        source_base_task["loss"],
        source_post_task["loss"],
        delta_norm,
    )
    del post_a, pre_a
    cleanup_memory()

    target_results: list[dict[str, Any]] = []
    for target_revision in targets:
        LOGGER.info("Direct-oracle run for target %s", target_revision)
        direct_model = load_model(args.model, target_revision, device)
        base_task = evaluate_task(direct_model, evaluation_loader, candidate_token_ids)
        base_generic = evaluate_generic_loss(direct_model, tokenizer, tokenizer.pad_token_id)
        train_task(
            direct_model,
            training_dataset,
            training_collator,
            args.steps,
            args.learning_rate,
            args.warmup_steps,
            args.batch_size,
            args.seed,
            args.log_every,
        )
        direct_task = evaluate_task(direct_model, evaluation_loader, candidate_token_ids)
        direct_generic = evaluate_generic_loss(direct_model, tokenizer, tokenizer.pad_token_id)
        del direct_model
        cleanup_memory()

        transfers: list[dict[str, float | bool]] = []
        for alpha in alphas:
            LOGGER.info("Transfer run target=%s alpha=%.3f", target_revision, alpha)
            transferred_model = load_model(args.model, target_revision, device)
            apply_delta(transferred_model, delta, alpha)
            transfer_task = evaluate_task(transferred_model, evaluation_loader, candidate_token_ids)
            transfer_generic = evaluate_generic_loss(transferred_model, tokenizer, tokenizer.pad_token_id)
            transfer_result: dict[str, float | bool] = {
                "alpha": alpha,
                "task_accuracy": transfer_task["accuracy"],
                "task_loss": transfer_task["loss"],
                "generic_loss": transfer_generic,
                "accuracy_gain_recovery": recovery_ratio(
                    base_task["accuracy"], direct_task["accuracy"], transfer_task["accuracy"]
                ),
                "loss_reduction_recovery": loss_recovery_ratio(
                    base_task["loss"], direct_task["loss"], transfer_task["loss"]
                ),
                "generic_loss_relative_increase": (transfer_generic - base_generic) / base_generic,
            }
            transfer_result["high_fidelity_criterion_passed"] = (
                float(transfer_result["accuracy_gain_recovery"]) >= 0.90
                and float(transfer_result["loss_reduction_recovery"]) >= 0.90
                and float(transfer_result["generic_loss_relative_increase"]) <= 0.05
            )
            transfers.append(transfer_result)
            del transferred_model
            cleanup_memory()

        passing_transfers = [row for row in transfers if bool(row["high_fidelity_criterion_passed"])]
        best = max(
            passing_transfers or transfers, key=lambda row: (float(row["task_accuracy"]), -float(row["task_loss"]))
        )
        passed = bool(passing_transfers)
        target_result = {
            "revision": target_revision,
            "base_task": base_task,
            "direct_task": direct_task,
            "base_generic_loss": base_generic,
            "direct_generic_loss": direct_generic,
            "transfers": transfers,
            "best_transfer": best,
            "high_fidelity_criterion_passed": passed,
        }
        target_results.append(target_result)
        LOGGER.info(
            "Target %s: base/direct/best-transfer accuracy %.3f/%.3f/%.3f; "
            "accuracy recovery %.1f%%; loss recovery %.1f%%; criterion=%s",
            target_revision,
            base_task["accuracy"],
            direct_task["accuracy"],
            best["task_accuracy"],
            100.0 * float(best["accuracy_gain_recovery"]),
            100.0 * float(best["loss_reduction_recovery"]),
            "PASS" if passed else "FAIL",
        )

    results["targets"] = target_results
    results["verdict"] = (
        "validated_in_this_control"
        if all(row["high_fidelity_criterion_passed"] for row in target_results)
        else "high_fidelity_hypothesis_not_supported"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    LOGGER.info("VERDICT: %s", results["verdict"])
    LOGGER.info("Wrote results to %s", args.output)


if __name__ == "__main__":
    main()
