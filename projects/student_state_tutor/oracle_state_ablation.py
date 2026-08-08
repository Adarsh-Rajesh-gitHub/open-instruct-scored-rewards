"""Does an explicit student belief improve a tutor's intervention?

This is a decision-value gate, not a representation probe. It compares tutor
messages generated from:

1. the question and student utterance;
2. the same context plus the student's gold wrong belief; and
3. the same context plus a different wrong belief from the same item.

A frozen student model then scores every answer choice after reading only the
tutor message. The primary comparison is paired oracle-vs-transcript gain in
gold probability and gold-vs-believed margin.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CONDITIONS = ("transcript", "oracle", "counterfactual")
TUTOR_SYSTEM = """You are a patient tutor helping a middle-school student.
Respond directly to the student's stated confusion. Give one targeted conceptual
hint or diagnostic question in at most two sentences. Do not reveal the answer,
name an option letter, quote an answer option, or say which option is wrong."""


def format_choices(choices: list[str]) -> str:
    return "\n".join(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices))


def extract_student_opening(completion: str) -> str | None:
    match = re.search(r"(?:^|\n)Student:\s*(.*?)(?:\nTutor:|$)", completion, flags=re.DOTALL)
    if not match:
        return None
    opening = " ".join(match.group(1).strip().split())
    return opening or None


def load_examples(path: Path, limit: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    candidates = []
    seen_questions = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("subject") == "math":
            continue
        question = row.get("prompt")
        choices = row.get("choices")
        gold_idx = row.get("gold_idx")
        belief = row.get("student_believed")
        opening = extract_student_opening(row.get("completion", ""))
        if not question or not choices or opening is None or belief not in choices:
            continue
        belief_idx = choices.index(belief)
        if belief_idx == gold_idx or question in seen_questions:
            continue
        counterfactual_indices = [
            index for index in range(len(choices)) if index not in (gold_idx, belief_idx)
        ]
        if not counterfactual_indices:
            continue
        seen_questions.add(question)
        counterfactual_idx = rng.choice(counterfactual_indices)
        candidates.append(
            {
                "question_id": f"q{len(candidates):04d}",
                "question": question,
                "choices": choices,
                "gold_idx": gold_idx,
                "belief_idx": belief_idx,
                "belief": belief,
                "counterfactual_idx": counterfactual_idx,
                "counterfactual_belief": choices[counterfactual_idx],
                "student_opening": opening,
                "subject": row.get("subject", "unknown"),
                "grade": row.get("grade"),
            }
        )
    rng.shuffle(candidates)
    return candidates[:limit] if limit > 0 else candidates


def tutor_user_prompt(example: dict, condition: str) -> str:
    diagnostic = ""
    if condition == "oracle":
        diagnostic = (
            "\nPrivate diagnostic: the student currently believes "
            f'"{example["belief"]}". Use this to target the misconception, but do not '
            "repeat or quote the belief.\n"
        )
    elif condition == "counterfactual":
        diagnostic = (
            "\nPrivate diagnostic: the student currently believes "
            f'"{example["counterfactual_belief"]}". Use this to target the misconception, '
            "but do not repeat or quote the belief.\n"
        )
    return (
        f"Question:\n{example['question']}\n"
        f"{format_choices(example['choices'])}\n\n"
        f"Student said:\n{example['student_opening']}\n"
        f"{diagnostic}\nWrite the tutor's next message."
    )


def render_tutor_prompt(tokenizer, example: dict, condition: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": TUTOR_SYSTEM},
            {"role": "user", "content": tutor_user_prompt(example, condition)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def generate_tutor_responses(args, examples: list[dict]) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, dtype=torch.bfloat16
    ).to(args.device)
    model.eval()

    requests = [
        (index, condition, render_tutor_prompt(tokenizer, example, condition))
        for index, example in enumerate(examples)
        for condition in CONDITIONS
    ]
    responses: list[dict[str, str]] = [dict() for _ in examples]
    for start in range(0, len(requests), args.generation_batch_size):
        batch = requests[start : start + args.generation_batch_size]
        encoded = tokenizer(
            [prompt for _, _, prompt in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(args.device)
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_width = encoded.input_ids.shape[1]
        decoded = tokenizer.batch_decode(
            generated[:, prompt_width:], skip_special_tokens=True
        )
        for (example_index, condition, _), response in zip(
            batch, decoded, strict=True
        ):
            responses[example_index][condition] = response.strip()
        print(
            f"generated {min(start + len(batch), len(requests))}/{len(requests)}",
            flush=True,
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [
        {**example, "tutor_responses": response_map}
        for example, response_map in zip(examples, responses, strict=True)
    ]


def make_score_record(question: str, choices: list[str], hint: str) -> dict:
    head = f"Fact: {hint}\n" if hint else ""
    return {"prompt": f"{head}Question: {question}\nAnswer:", "choices": choices}


@torch.inference_mode()
def score_records(args, records: list[dict]) -> list[list[float]]:
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.student_model, dtype=torch.bfloat16
    ).to(args.device)
    model.eval()

    flattened = []
    for record_index, record in enumerate(records):
        prompt_ids = tokenizer(
            record["prompt"], add_special_tokens=True
        ).input_ids
        for choice_index, choice in enumerate(record["choices"]):
            choice_ids = tokenizer(
                " " + choice, add_special_tokens=False
            ).input_ids
            flattened.append(
                (record_index, choice_index, prompt_ids + choice_ids, len(prompt_ids))
            )

    scored = [[float("nan")] * len(record["choices"]) for record in records]
    for start in range(0, len(flattened), args.scoring_batch_size):
        batch = flattened[start : start + args.scoring_batch_size]
        max_length = max(len(input_ids) for _, _, input_ids, _ in batch)
        input_ids = torch.full(
            (len(batch), max_length),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=args.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        choice_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row_index, (_, _, sequence, prompt_length) in enumerate(batch):
            offset = max_length - len(sequence)
            input_ids[row_index, offset:] = torch.tensor(
                sequence, dtype=torch.long, device=args.device
            )
            attention_mask[row_index, offset:] = 1
            choice_mask[row_index, offset + prompt_length :] = True

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probabilities = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        targets = input_ids[:, 1:]
        token_log_probabilities = log_probabilities.gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        shifted_choice_mask = choice_mask[:, 1:]
        sequence_scores = (
            (token_log_probabilities * shifted_choice_mask).sum(dim=1)
            / shifted_choice_mask.sum(dim=1).clamp_min(1)
        )
        for (record_index, choice_index, _, _), score in zip(
            batch, sequence_scores.tolist(), strict=True
        ):
            scored[record_index][choice_index] = score
        print(
            f"scored {min(start + len(batch), len(flattened))}/{len(flattened)} choices",
            flush=True,
        )
    return scored


def softmax(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    shifted = array - array.max()
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def leaks_answer(response: str, example: dict) -> bool:
    normalized_response = normalize_text(response)
    gold = normalize_text(example["choices"][example["gold_idx"]])
    if len(gold) >= 4 and gold in normalized_response:
        return True
    return bool(
        re.search(
            r"\b(?:answer|option|choice)\s*(?:is\s*)?[abcd]\b",
            response,
            flags=re.IGNORECASE,
        )
    )


def attach_scores(rows: list[dict], score_sets: list[list[float]]) -> None:
    cursor = 0
    for row in rows:
        baseline_scores = score_sets[cursor]
        cursor += 1
        baseline_probabilities = softmax(baseline_scores)
        row["baseline"] = {
            "scores": baseline_scores,
            "gold_probability": float(baseline_probabilities[row["gold_idx"]]),
            "belief_margin": float(
                baseline_scores[row["gold_idx"]]
                - baseline_scores[row["belief_idx"]]
            ),
            "solved": int(np.argmax(baseline_scores) == row["gold_idx"]),
        }
        row["condition_scores"] = {}
        for condition in CONDITIONS:
            scores = score_sets[cursor]
            cursor += 1
            probabilities = softmax(scores)
            response = row["tutor_responses"][condition]
            row["condition_scores"][condition] = {
                "scores": scores,
                "gold_probability": float(probabilities[row["gold_idx"]]),
                "belief_margin": float(
                    scores[row["gold_idx"]] - scores[row["belief_idx"]]
                ),
                "solved": int(np.argmax(scores) == row["gold_idx"]),
                "leaked": leaks_answer(response, row),
            }


def mean_metric(rows: list[dict], condition: str, metric: str, nonleaky: bool) -> float:
    values = [
        row["condition_scores"][condition][metric]
        for row in rows
        if not nonleaky or not row["condition_scores"][condition]["leaked"]
    ]
    return float(np.mean(values)) if values else float("nan")


def paired_difference(
    rows: list[dict], left: str, right: str, metric: str
) -> np.ndarray:
    return np.asarray(
        [
            row["condition_scores"][left][metric]
            - row["condition_scores"][right][metric]
            for row in rows
        ],
        dtype=np.float64,
    )


def bootstrap_interval(values: np.ndarray, seed: int, samples: int = 5000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(rows: list[dict], seed: int) -> dict:
    summary = {
        "n": len(rows),
        "subjects": dict(Counter(row["subject"] for row in rows)),
        "baseline": {
            metric: float(np.mean([row["baseline"][metric] for row in rows]))
            for metric in ("gold_probability", "belief_margin", "solved")
        },
        "conditions": {},
        "paired_comparisons": {},
    }
    for condition in CONDITIONS:
        summary["conditions"][condition] = {
            metric: mean_metric(rows, condition, metric, nonleaky=False)
            for metric in ("gold_probability", "belief_margin", "solved", "leaked")
        }
        summary["conditions"][condition]["nonleaky"] = {
            metric: mean_metric(rows, condition, metric, nonleaky=True)
            for metric in ("gold_probability", "belief_margin", "solved")
        }

    for left, right in (
        ("oracle", "transcript"),
        ("oracle", "counterfactual"),
        ("transcript", "counterfactual"),
    ):
        name = f"{left}_minus_{right}"
        summary["paired_comparisons"][name] = {}
        for metric in ("gold_probability", "belief_margin", "solved"):
            differences = paired_difference(rows, left, right, metric)
            summary["paired_comparisons"][name][metric] = {
                "mean": float(differences.mean()),
                "95_ci": bootstrap_interval(differences, seed),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generation-batch-size", type=int, default=24)
    parser.add_argument("--scoring-batch-size", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--force-generation", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    generations_path = args.out_dir / "generations.jsonl"
    scored_path = args.out_dir / "scored.jsonl"
    results_path = args.out_dir / "results.json"

    if generations_path.exists() and not args.force_generation:
        rows = [
            json.loads(line)
            for line in generations_path.read_text().splitlines()
            if line.strip()
        ]
        print(f"loaded {len(rows)} cached generations", flush=True)
    else:
        examples = load_examples(args.traces, args.limit, args.seed)
        if not examples:
            raise ValueError("no qualifying non-math examples found")
        print(
            f"loaded {len(examples)} examples: "
            f"{dict(Counter(example['subject'] for example in examples))}",
            flush=True,
        )
        rows = generate_tutor_responses(args, examples)
        generations_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    score_records_input = []
    for row in rows:
        score_records_input.append(
            make_score_record(row["question"], row["choices"], "")
        )
        for condition in CONDITIONS:
            score_records_input.append(
                make_score_record(
                    row["question"],
                    row["choices"],
                    row["tutor_responses"][condition],
                )
            )
    scores = score_records(args, score_records_input)
    attach_scores(rows, scores)
    scored_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    results = summarize(rows, args.seed)
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
