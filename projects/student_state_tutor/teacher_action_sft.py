"""LoRA-SFT a teacher to map mastery graph states to structured actions."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from projects.student_state_tutor.mastery_state import (
    EvidenceEvent,
    MasteryGraphState,
)
from projects.student_state_tutor.teacher_action_ablation import (
    SYSTEM,
    load_banks,
    render_prompt,
)


ACTION_COUNTS = {
    "diagnostic": ((0, 0),),
    "worked_example": ((0, 3), (0, 4), (1, 6)),
    "contrast_case": ((1, 2), (1, 3), (2, 3)),
    "retrieval_practice": ((1, 1), (2, 1), (3, 2), (4, 2)),
    "stop": ((3, 0), (4, 0)),
}


def add_count_events(
    graph: MasteryGraphState,
    concept_id: str,
    successes: int,
    failures: int,
) -> None:
    for index, correct in enumerate([True] * successes + [False] * failures):
        graph.observe(
            EvidenceEvent(
                learner_id=graph.learner_id,
                concept_id=concept_id,
                item_id=f"{concept_id}_train_{index}",
                correct=correct,
                timestamp=f"2026-08-06T00:{index:02d}:00+00:00",
                source="synthetic_sft_rule",
            )
        )


def build_sft_examples(
    banks: list[dict],
    examples_per_action: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    concept_ids = [bank["concept_id"] for bank in banks]
    bank_map = {bank["concept_id"]: bank for bank in banks}
    candidates = [
        {
            "concept_id": concept_id,
            "title": bank_map[concept_id]["title"],
            "item_id": (
                f"{concept_id}:"
                f"{bank_map[concept_id]['questions'][0]['question_id']}"
            ),
        }
        for concept_id in concept_ids
    ]
    examples = []
    for action, target_count_options in ACTION_COUNTS.items():
        for index in range(examples_per_action):
            target = concept_ids[index % len(concept_ids)]
            learner_id = f"sft_learner_{rng.getrandbits(64):016x}"
            graph = MasteryGraphState(
                learner_id=learner_id,
                concept_ids=concept_ids,
            )
            if action == "stop":
                for rank, concept_id in enumerate(concept_ids):
                    add_count_events(
                        graph,
                        concept_id,
                        4 + rank + rng.randint(0, 1),
                        0,
                    )
                target = graph.weakest()
            else:
                target_counts = rng.choice(target_count_options)
                add_count_events(graph, target, *target_counts)
                for concept_id in concept_ids:
                    if concept_id == target:
                        continue
                    add_count_events(
                        graph,
                        concept_id,
                        4 + rng.randint(0, 2),
                        rng.randint(0, 1),
                    )
            target = graph.weakest()
            item_id = next(
                row["item_id"]
                for row in candidates
                if row["concept_id"] == target
            )
            decision = {
                "target_node": target,
                "action": action,
                "item_id": None if action == "stop" else item_id,
                "stop": action == "stop",
            }
            case = {
                "case_id": graph.learner_id,
                "learner_id": graph.learner_id,
                "candidates": candidates,
                "events": [],
                "latest_event": {},
                "graph_view": graph.local_view(),
                "shuffled_graph_view": graph.local_view(),
                "oracle_target": target,
                "oracle_action": action,
                "shuffled_visible_target": target,
            }
            examples.append(
                {
                    "case": case,
                    "condition": "graph",
                    "decision": decision,
                }
            )
    rng.shuffle(examples)
    return examples


class ActionDataset(Dataset):
    def __init__(self, tokenizer, examples: list[dict], max_length: int):
        self.rows = []
        for example in examples:
            prompt = render_prompt(
                tokenizer, example["case"], example["condition"]
            )
            response = json.dumps(example["decision"], separators=(",", ":"))
            prompt_ids = tokenizer(
                prompt, add_special_tokens=False
            ).input_ids
            response_ids = tokenizer(
                response + tokenizer.eos_token,
                add_special_tokens=False,
            ).input_ids
            input_ids = (prompt_ids + response_ids)[-max_length:]
            prompt_kept = max(0, len(input_ids) - len(response_ids))
            labels = [-100] * prompt_kept + input_ids[prompt_kept:]
            self.rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class ActionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        width = max(len(row["input_ids"]) for row in rows)
        input_ids = []
        attention_mask = []
        labels = []
        for row in rows:
            padding = width - len(row["input_ids"])
            input_ids.append(
                row["input_ids"] + [self.pad_token_id] * padding
            )
            attention_mask.append(row["attention_mask"] + [0] * padding)
            labels.append(row["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_mask, dtype=torch.long
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--teacher-model", default="Qwen/Qwen2.5-3B-Instruct"
    )
    parser.add_argument("--examples-per-action", type=int, default=96)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    banks = load_banks(args.banks)
    examples = build_sft_examples(
        banks, args.examples_per_action, args.seed
    )
    (args.out_dir / "training_examples.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "prompt_case": example["case"],
                    "decision": example["decision"],
                }
            )
            + "\n"
            for example in examples
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = ActionDataset(tokenizer, examples, args.max_length)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        dtype=torch.bfloat16,
    ).to("cuda")
    model.config.use_cache = False
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    collator = ActionCollator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    epochs = int(args.epochs)
    optimizer_steps_per_epoch = (
        len(loader) + args.gradient_accumulation - 1
    ) // args.gradient_accumulation
    total_optimizer_steps = optimizer_steps_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.05 * total_optimizer_steps)),
        num_training_steps=total_optimizer_steps,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    optimizer_step = 0
    for epoch in range(epochs):
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to("cuda") for key, value in batch.items()}
            loss = model(**batch).loss
            (loss / args.gradient_accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            should_step = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == len(loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if optimizer_step % 5 == 0:
                    recent = losses[-5 * args.gradient_accumulation :]
                    print(
                        f"epoch={epoch + 1} step={optimizer_step}/"
                        f"{total_optimizer_steps} loss={sum(recent) / len(recent):.4f}",
                        flush=True,
                    )
        checkpoint = args.out_dir / "checkpoints" / f"epoch_{epoch + 1}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint)

    adapter_dir = args.out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    metrics = {
        "train_loss": float(sum(losses) / len(losses)),
        "optimizer_steps": optimizer_step,
        "epochs": epochs,
        "examples": len(examples),
        "action_counts": dict(
            Counter(example["decision"]["action"] for example in examples)
        ),
        "base_model": args.teacher_model,
        "system_prompt": SYSTEM,
    }
    (args.out_dir / "train_results.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
