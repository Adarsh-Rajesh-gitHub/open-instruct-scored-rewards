#!/usr/bin/env python3
"""Checkpoint-age, scaling, interpolation, repair, and retention sweep."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from scripts.task_vector_transfer import (
    CODE_CANDIDATES,
    EVAL_TEMPLATES,
    TRAIN_TEMPLATES,
    PromptDataset,
    TrainingDataset,
    apply_delta,
    autocast_context,
    build_rows,
    cleanup,
    collate_prompts,
    collate_training,
    compute_delta,
    evaluate_retention,
    evaluate_task,
    fixed_retention_batches,
    load_snapshot,
    make_optimizer,
    seed_everything,
    train_task,
    validate_arm,
)
from torch.utils.data import DataLoader
from train.tokenizer import get_tok

LOGGER = logging.getLogger("engram-transfer-sweep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("dense", "engram"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--climbmix-val", type=Path, required=True)
    parser.add_argument("--benchmark-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--repair-learning-rate", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--alpha", type=float, action="append", default=None)
    parser.add_argument("--merge-lambda", type=float, action="append", default=None)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().float().cpu().clone() for name, parameter in model.named_parameters()}


@torch.no_grad()
def interpolate_with_posttrained_early(
    model: torch.nn.Module, posttrained_early: dict[str, torch.Tensor], weight: float
) -> None:
    seen = 0
    for name, parameter in model.named_parameters():
        early_parameter = posttrained_early.get(name)
        if early_parameter is None or early_parameter.shape != parameter.shape:
            raise ValueError(f"Interpolation mismatch for {name}.")
        parameter.mul_(1.0 - weight)
        parameter.add_(early_parameter.to(parameter.device), alpha=weight)
        seen += 1
    if seen != len(posttrained_early):
        raise RuntimeError(f"Interpolated {seen} of {len(posttrained_early)} tensors.")


def load_external_benchmarks(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "wikitext_ids": payload["wikitext_ids"].long(),
        "lambada_contexts": [context.long() for context in payload["lambada_contexts"]],
        "lambada_targets": payload["lambada_targets"].long(),
        "metadata": payload["metadata"],
    }


@torch.inference_mode()
def evaluate_wikitext(
    model: torch.nn.Module, token_ids: torch.Tensor, batches: int = 16, batch_size: int = 4, context: int = 256
) -> float:
    model.eval()
    device = next(model.parameters()).device
    required = batches * batch_size * (context + 1)
    if token_ids.numel() < required:
        raise ValueError("WikiText benchmark tensor is too short.")
    total_nll = 0.0
    total_tokens = 0
    cursor = 0
    for _ in range(batches):
        values = token_ids[cursor : cursor + batch_size * (context + 1)]
        values = values.reshape(batch_size, context + 1)
        cursor += batch_size * (context + 1)
        x = values[:, :-1].to(device)
        y = values[:, 1:].to(device)
        with autocast_context():
            logits, _ = model(x)
        total_nll += float(
            F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum").item()
        )
        total_tokens += y.numel()
    return total_nll / total_tokens


@torch.inference_mode()
def evaluate_lambada(
    model: torch.nn.Module, contexts: Sequence[torch.Tensor], targets: torch.Tensor, batch_size: int = 16
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    total_nll = 0.0
    total_correct = 0
    total_examples = 0
    for start in range(0, len(contexts), batch_size):
        rows = contexts[start : start + batch_size]
        width = max(int(row.numel()) for row in rows)
        x = torch.full((len(rows), width), 50256, dtype=torch.long)
        lengths = torch.empty(len(rows), dtype=torch.long)
        for index, row in enumerate(rows):
            x[index, : row.numel()] = row
            lengths[index] = row.numel()
        y = targets[start : start + len(rows)]
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.to(device)
        with autocast_context():
            logits, _ = model(x)
        next_logits = logits[torch.arange(len(rows), device=device), lengths - 1].float()
        total_nll += float(F.cross_entropy(next_logits, y, reduction="sum").item())
        total_correct += int((next_logits.argmax(-1) == y).sum().item())
        total_examples += len(rows)
    return {"loss": total_nll / total_examples, "accuracy": total_correct / total_examples}


def evaluate_general(
    model: torch.nn.Module, climbmix_batches: Sequence[tuple[torch.Tensor, torch.Tensor]], benchmarks: dict[str, Any]
) -> dict[str, float]:
    lambada = evaluate_lambada(model, benchmarks["lambada_contexts"], benchmarks["lambada_targets"])
    return {
        "climbmix_loss": evaluate_retention(model, climbmix_batches),
        "wikitext_loss": evaluate_wikitext(model, benchmarks["wikitext_ids"]),
        "lambada_loss": lambada["loss"],
        "lambada_accuracy": lambada["accuracy"],
    }


def evaluate_variant(
    model: torch.nn.Module,
    select_loader: DataLoader[dict[str, torch.Tensor]],
    test_loader: DataLoader[dict[str, torch.Tensor]],
    code_token_ids: Sequence[int],
    climbmix_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    benchmarks: dict[str, Any],
) -> dict[str, Any]:
    return {
        "selection_task": evaluate_task(model, select_loader, code_token_ids),
        "test_task": evaluate_task(model, test_loader, code_token_ids),
        "general": evaluate_general(model, climbmix_batches, benchmarks),
    }


def relative_forgetting(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    return {
        "climbmix_loss_relative_change": (candidate["climbmix_loss"] - baseline["climbmix_loss"])
        / baseline["climbmix_loss"],
        "wikitext_loss_relative_change": (candidate["wikitext_loss"] - baseline["wikitext_loss"])
        / baseline["wikitext_loss"],
        "lambada_loss_relative_change": (candidate["lambada_loss"] - baseline["lambada_loss"])
        / baseline["lambada_loss"],
        "lambada_accuracy_change": (candidate["lambada_accuracy"] - baseline["lambada_accuracy"]),
    }


def train_steps_with_existing_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader[dict[str, torch.Tensor]],
    iterator: Any,
    steps: int,
) -> Any:
    device = next(model.parameters()).device
    model.train()
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("Non-finite repair loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return iterator


def repair_curve(
    model: torch.nn.Module,
    train_dataset: TrainingDataset,
    select_loader: DataLoader[dict[str, torch.Tensor]],
    test_loader: DataLoader[dict[str, torch.Tensor]],
    code_token_ids: Sequence[int],
    climbmix_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    benchmarks: dict[str, Any],
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    checkpoints = (0, 5, 10, 20, 40, 80)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_training,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = make_optimizer(model, learning_rate)
    iterator = iter(loader)
    result = []
    completed = 0
    for checkpoint in checkpoints:
        if checkpoint > completed:
            iterator = train_steps_with_existing_optimizer(model, optimizer, loader, iterator, checkpoint - completed)
            completed = checkpoint
        metrics = evaluate_variant(model, select_loader, test_loader, code_token_ids, climbmix_batches, benchmarks)
        metrics["repair_steps"] = checkpoint
        result.append(metrics)
    del optimizer
    return result


def choose_best(rows: Sequence[dict[str, Any]], coefficient_name: str) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row["metrics"]["selection_task"]["accuracy"],
            -row["metrics"]["selection_task"]["loss"],
            -abs(float(row[coefficient_name])),
        ),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    alphas = args.alpha or [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    merge_weights = args.merge_lambda or [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")

    tokenizer = get_tok()
    encoded_codes = [tokenizer.encode(code) for code in CODE_CANDIDATES]
    if any(len(ids) != 1 for ids in encoded_codes):
        raise RuntimeError(f"Task codes are not single tokens: {encoded_codes}")
    code_token_ids = [ids[0] for ids in encoded_codes]
    train_rows = build_rows(TRAIN_TEMPLATES, tokenizer, code_token_ids)
    selection_rows = build_rows(EVAL_TEMPLATES[:4], tokenizer, code_token_ids)
    test_rows = build_rows(EVAL_TEMPLATES[4:], tokenizer, code_token_ids)
    train_dataset = TrainingDataset(train_rows)
    select_loader = DataLoader(
        PromptDataset(selection_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collate_prompts
    )
    test_loader = DataLoader(
        PromptDataset(test_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collate_prompts
    )
    climbmix_batches = fixed_retention_batches(args.climbmix_val, batches=8, batch_size=4, context=256)
    benchmarks = load_external_benchmarks(args.benchmark_file)

    LOGGER.info("Fine-tuning source %s for %s", args.source, args.arm)
    pre_source = load_snapshot(args.source, torch.device("cpu"))
    post_source = load_snapshot(args.source, device)
    validate_arm(pre_source, args.arm)
    validate_arm(post_source, args.arm)
    source_before = evaluate_variant(
        post_source, select_loader, test_loader, code_token_ids, climbmix_batches, benchmarks
    )
    train_task(
        post_source,
        train_dataset,
        args.steps,
        args.learning_rate,
        warmup_steps=8,
        batch_size=args.batch_size,
        seed=args.seed,
        log_every=args.log_every,
    )
    source_after = evaluate_variant(
        post_source, select_loader, test_loader, code_token_ids, climbmix_batches, benchmarks
    )
    delta, delta_norm = compute_delta(post_source, pre_source)
    post_source_state = state_dict_cpu(post_source)
    source_step = int(post_source.snapshot_step)
    del pre_source, post_source
    cleanup()

    LOGGER.info("Building final-checkpoint baselines")
    direct_final = load_snapshot(args.target, device)
    validate_arm(direct_final, args.arm)
    target_step = int(direct_final.snapshot_step)
    final_before = evaluate_variant(
        direct_final, select_loader, test_loader, code_token_ids, climbmix_batches, benchmarks
    )
    train_task(
        direct_final,
        train_dataset,
        args.steps,
        args.learning_rate,
        warmup_steps=8,
        batch_size=args.batch_size,
        seed=args.seed,
        log_every=args.log_every,
    )
    final_after = evaluate_variant(
        direct_final, select_loader, test_loader, code_token_ids, climbmix_batches, benchmarks
    )
    del direct_final
    cleanup()

    transfer_rows: list[dict[str, Any]] = []
    for alpha in alphas:
        LOGGER.info("Task-vector alpha %.2f", alpha)
        model = load_snapshot(args.target, device)
        validate_arm(model, args.arm)
        apply_delta(model, delta, alpha)
        metrics = evaluate_variant(model, select_loader, test_loader, code_token_ids, climbmix_batches, benchmarks)
        transfer_rows.append(
            {
                "alpha": alpha,
                "metrics": metrics,
                "forgetting_vs_final": relative_forgetting(final_before["general"], metrics["general"]),
            }
        )
        del model
        cleanup()

    interpolation_rows: list[dict[str, Any]] = []
    for merge_weight in merge_weights:
        LOGGER.info("Full-weight interpolation lambda %.2f", merge_weight)
        model = load_snapshot(args.target, device)
        validate_arm(model, args.arm)
        interpolate_with_posttrained_early(model, post_source_state, merge_weight)
        metrics = evaluate_variant(model, select_loader, test_loader, code_token_ids, climbmix_batches, benchmarks)
        interpolation_rows.append(
            {
                "lambda": merge_weight,
                "metrics": metrics,
                "forgetting_vs_final": relative_forgetting(final_before["general"], metrics["general"]),
            }
        )
        del model
        cleanup()

    best_transfer = choose_best(transfer_rows, "alpha")
    interior_interpolations = [row for row in interpolation_rows if 0.0 < float(row["lambda"]) < 1.0]
    if not interior_interpolations:
        raise RuntimeError("At least one interpolation coefficient must be inside (0, 1).")
    best_interpolation = choose_best(interior_interpolations, "lambda")
    repair: dict[str, Any] = {}
    for name, coefficient, mode in (
        ("task_vector", best_transfer["alpha"], "task_vector"),
        ("interpolation", best_interpolation["lambda"], "interpolation"),
    ):
        LOGGER.info("Repairing best %s variant", name)
        model = load_snapshot(args.target, device)
        validate_arm(model, args.arm)
        if mode == "task_vector":
            apply_delta(model, delta, float(coefficient))
        else:
            interpolate_with_posttrained_early(model, post_source_state, float(coefficient))
        curve = repair_curve(
            model,
            train_dataset,
            select_loader,
            test_loader,
            code_token_ids,
            climbmix_batches,
            benchmarks,
            args.repair_learning_rate,
            args.batch_size,
            args.seed,
        )
        for point in curve:
            point["forgetting_vs_final"] = relative_forgetting(final_before["general"], point["general"])
        repair[name] = {"starting_coefficient": coefficient, "curve": curve}
        del model
        cleanup()

    report = {
        "arm": args.arm,
        "source_snapshot": str(args.source),
        "target_snapshot": str(args.target),
        "source_step": source_step,
        "target_step": target_step,
        "checkpoint_age_steps": target_step - source_step,
        "posttraining_data": "synthetic routing task, disjoint from ClimbMix",
        "task_train_examples": len(train_rows),
        "task_selection_examples": len(selection_rows),
        "task_test_examples": len(test_rows),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "repair_learning_rate": args.repair_learning_rate,
        "seed": args.seed,
        "external_benchmarks": benchmarks["metadata"],
        "source_before": source_before,
        "source_after": source_after,
        "delta_l2_norm": delta_norm,
        "delta_tensors": len(delta),
        "final_before": final_before,
        "final_after_direct": final_after,
        "task_vector_sweep": transfer_rows,
        "best_task_vector": best_transfer,
        "interpolation_sweep": interpolation_rows,
        "best_interpolation": best_interpolation,
        "repair": repair,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info(
        "DONE %s source=%d: early %.3f->%.3f, final %.3f->%.3f, best transfer %.3f at alpha %.2f",
        args.arm,
        source_step,
        source_before["test_task"]["accuracy"],
        source_after["test_task"]["accuracy"],
        final_before["test_task"]["accuracy"],
        final_after["test_task"]["accuracy"],
        best_transfer["metrics"]["test_task"]["accuracy"],
        best_transfer["alpha"],
    )

    del delta, post_source_state
    gc.collect()


if __name__ == "__main__":
    main()
