"""Continue-train a function-preserving partial Qwen MoE on raw text."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from projects.dense_to_moe_upcycling.upcycled_mlp import (
    combined_router_loss,
    iter_upcycled_layers,
    parse_layer_spec,
    routing_report,
    save_moe_checkpoint,
    set_only_moe_trainable,
    upcycle_qwen_layers,
)


class TokenBlockDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        sequence_length: int,
        max_blocks: int = 0,
    ):
        blocks = []
        buffer: list[int] = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            text = json.loads(line).get("text", "")
            token_ids = tokenizer(
                text, add_special_tokens=False
            ).input_ids
            if tokenizer.eos_token_id is not None:
                token_ids.append(tokenizer.eos_token_id)
            buffer.extend(token_ids)
            while len(buffer) >= sequence_length:
                blocks.append(
                    torch.tensor(
                        buffer[:sequence_length], dtype=torch.long
                    )
                )
                del buffer[:sequence_length]
                if max_blocks > 0 and len(blocks) >= max_blocks:
                    self.blocks = blocks
                    return
        self.blocks = blocks
        if not self.blocks:
            raise ValueError("data produced no complete token blocks")

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.blocks[index]


@torch.inference_mode()
def evaluate(model, blocks: list[torch.Tensor], device: str) -> float:
    model.eval()
    losses = []
    for block in blocks:
        input_ids = block.unsqueeze(0).to(device)
        losses.append(
            float(model(input_ids=input_ids, labels=input_ids).loss)
        )
    model.train()
    return float(sum(losses) / len(losses))


def split_parameters(model):
    routers = []
    experts = []
    for _, module in iter_upcycled_layers(model):
        routers.extend(module.router.parameters())
        experts.extend(module.experts.parameters())
    return routers, experts


def parameter_grad_norm(parameters) -> float:
    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        squared = value if squared is None else squared + value
    return 0.0 if squared is None else float(squared.sqrt())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-3B-Instruct"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", default="last2")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--router-std", type=float, default=0.02)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--max-blocks", type=int, default=4096)
    parser.add_argument("--eval-blocks", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--expert-lr", type=float, default=2e-6)
    parser.add_argument("--router-lr", type=float, default=1e-4)
    parser.add_argument("--load-balance-coef", type=float, default=0.01)
    parser.add_argument("--router-z-coef", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = TokenBlockDataset(
        args.data,
        tokenizer,
        args.sequence_length,
        max_blocks=args.max_blocks,
    )
    if len(dataset) <= args.eval_blocks:
        raise ValueError("need more token blocks than --eval-blocks")
    train_blocks = dataset.blocks[: -args.eval_blocks]
    eval_blocks = dataset.blocks[-args.eval_blocks :]
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_blocks,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(args.device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    probe = eval_blocks[0].unsqueeze(0).to(args.device)
    model.eval()
    with torch.inference_mode():
        dense_output = model(input_ids=probe, labels=probe)
        dense_loss = float(dense_output.loss)
        dense_logits = dense_output.logits.float().cpu()

    layer_indices = parse_layer_spec(
        args.layers, len(model.model.layers)
    )
    manifest = upcycle_qwen_layers(
        model,
        layer_indices,
        num_experts=args.num_experts,
        top_k=args.top_k,
        router_std=args.router_std,
    )
    with torch.inference_mode():
        upcycled_output = model(input_ids=probe, labels=probe)
        initial_loss = float(upcycled_output.loss)
        difference = upcycled_output.logits.float().cpu() - dense_logits
        equivalence = {
            "dense_loss": dense_loss,
            "upcycled_loss": initial_loss,
            "max_logit_difference": float(difference.abs().max()),
            "mean_logit_difference": float(difference.abs().mean()),
        }
    del dense_output, dense_logits, upcycled_output, difference, probe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    parameter_counts = set_only_moe_trainable(model)
    routers, experts = split_parameters(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": routers, "lr": args.router_lr},
            {"params": experts, "lr": args.expert_lr},
        ],
        weight_decay=0.01,
    )
    total_optimizer_steps = max(1, args.steps)
    warmup = max(1, int(0.05 * total_optimizer_steps))

    def learning_rate_scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(
            1, total_optimizer_steps - warmup
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    initial_eval_loss = evaluate(model, eval_blocks, args.device)
    for _, module in iter_upcycled_layers(model):
        module.reset_routing_counts()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    logs = []
    start_time = time.time()
    for optimizer_step in range(args.steps):
        step_loss = 0.0
        step_lm = 0.0
        step_load = 0.0
        step_z = 0.0
        for _ in range(args.gradient_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            input_ids = batch.to(args.device)
            output = model(input_ids=input_ids, labels=input_ids)
            load_loss, z_loss = combined_router_loss(model)
            loss = (
                output.loss
                + args.load_balance_coef * load_loss
                + args.router_z_coef * z_loss
            )
            (loss / args.gradient_accumulation).backward()
            step_loss += float(loss.detach())
            step_lm += float(output.loss.detach())
            step_load += float(load_loss.detach())
            step_z += float(z_loss.detach())
        router_grad_norm = parameter_grad_norm(routers)
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            args.max_grad_norm,
        )
        scale = learning_rate_scale(optimizer_step)
        for group, base_lr in zip(
            optimizer.param_groups,
            (args.router_lr, args.expert_lr),
            strict=True,
        ):
            group["lr"] = base_lr * scale
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {
            "step": optimizer_step + 1,
            "loss": step_loss / args.gradient_accumulation,
            "lm_loss": step_lm / args.gradient_accumulation,
            "load_balance_loss": step_load / args.gradient_accumulation,
            "router_z_loss": step_z / args.gradient_accumulation,
            "router_lr": optimizer.param_groups[0]["lr"],
            "expert_lr": optimizer.param_groups[1]["lr"],
            "router_grad_norm": router_grad_norm,
            "total_grad_norm": float(total_grad_norm),
            "elapsed_seconds": time.time() - start_time,
        }
        logs.append(record)
        print(json.dumps(record), flush=True)

    final_eval_loss = evaluate(model, eval_blocks, args.device)
    results = {
        "base_model": args.model,
        "data": str(args.data),
        "data_blocks": len(dataset),
        "train_blocks": len(train_blocks),
        "eval_blocks": len(eval_blocks),
        "sequence_length": args.sequence_length,
        "steps": args.steps,
        "manifest": manifest,
        "parameter_counts": parameter_counts,
        "equivalence": equivalence,
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": final_eval_loss,
        "routing": routing_report(model),
        "last_train_record": logs[-1] if logs else None,
    }
    (args.out_dir / "train_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in logs)
    )
    (args.out_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    save_moe_checkpoint(
        model,
        args.out_dir / "moe_checkpoint.pt",
        base_model=args.model,
        manifest=manifest,
        extra={
            "results": results,
            "tokenizer": args.model,
        },
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
