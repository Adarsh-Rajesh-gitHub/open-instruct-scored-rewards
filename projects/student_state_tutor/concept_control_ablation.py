"""Separate student-specific relation value from generic concept scaffolding."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from projects.student_state_tutor import graph_state_ablation as graph
from projects.student_state_tutor import oracle_state_ablation as base


NEW_CONDITIONS = ("concept_only", "shuffled_concept")


def concept_tutor_prompt(tokenizer, row: dict, concept: str) -> str:
    user = (
        f"Question:\n{row['question']}\n"
        f"{base.format_choices(row['choices'])}\n\n"
        f"Student said:\n{row['student_opening']}\n\n"
        "Private concept scaffold:\n"
        f"concept = {concept}\n\n"
        "Use the scaffold only to select a useful intervention. Do not repeat "
        "its phrasing. Write the tutor's next message."
    )
    return graph.render_chat(tokenizer, base.TUTOR_SYSTEM, user)


@torch.inference_mode()
def generate_controls(args, rows: list[dict]) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, dtype=torch.bfloat16
    ).to(args.device)
    model.eval()

    concepts = [row["graph_actual"]["concept"] for row in rows]
    shifted_concepts = concepts[1:] + concepts[:1]
    requests = []
    for index, (row, shifted_concept) in enumerate(
        zip(rows, shifted_concepts, strict=True)
    ):
        requests.append(
            (
                index,
                "concept_only",
                concept_tutor_prompt(tokenizer, row, row["graph_actual"]["concept"]),
            )
        )
        requests.append(
            (
                index,
                "shuffled_concept",
                concept_tutor_prompt(tokenizer, row, shifted_concept),
            )
        )
    outputs = graph.batch_generate(
        model,
        tokenizer,
        requests,
        args.generation_batch_size,
        args.max_input_tokens,
        args.max_new_tokens,
        args.device,
    )
    for row, output_map, shifted_concept in zip(
        rows, outputs, shifted_concepts, strict=True
    ):
        row["concept_control_responses"] = output_map
        row["shuffled_concept"] = shifted_concept

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def attach_scores(rows: list[dict], score_sets: list[list[float]]) -> None:
    for row_index, row in enumerate(rows):
        row["concept_control_scores"] = {}
        for condition_index, condition in enumerate(NEW_CONDITIONS):
            scores = score_sets[row_index * len(NEW_CONDITIONS) + condition_index]
            probabilities = base.softmax(scores)
            response = row["concept_control_responses"][condition]
            row["concept_control_scores"][condition] = {
                "scores": scores,
                "gold_probability": float(probabilities[row["gold_idx"]]),
                "belief_margin": float(
                    scores[row["gold_idx"]] - scores[row["belief_idx"]]
                ),
                "solved": int(np.argmax(scores) == row["gold_idx"]),
                "leaked": base.leaks_answer(response, row),
            }


def metric(row: dict, condition: str, name: str) -> float:
    if condition == "transcript":
        return float(row["condition_scores"]["transcript"][name])
    if condition == "graph_actual":
        return float(row["graph_condition_scores"]["graph_actual"][name])
    return float(row["concept_control_scores"][condition][name])


def summarize(rows: list[dict], seed: int) -> dict:
    conditions = ("transcript", "graph_actual") + NEW_CONDITIONS
    result = {
        "n": len(rows),
        "subjects": dict(Counter(row["subject"] for row in rows)),
        "conditions": {},
        "paired_comparisons": {},
    }
    for condition in conditions:
        result["conditions"][condition] = {
            name: float(np.mean([metric(row, condition, name) for row in rows]))
            for name in ("gold_probability", "belief_margin", "solved", "leaked")
        }
    for left, right in (
        ("graph_actual", "concept_only"),
        ("concept_only", "transcript"),
        ("concept_only", "shuffled_concept"),
    ):
        result["paired_comparisons"][f"{left}_minus_{right}"] = {}
        for name in ("gold_probability", "belief_margin", "solved"):
            differences = np.asarray(
                [
                    metric(row, left, name) - metric(row, right, name)
                    for row in rows
                ],
                dtype=np.float64,
            )
            result["paired_comparisons"][f"{left}_minus_{right}"][name] = {
                "mean": float(differences.mean()),
                "95_ci": base.bootstrap_interval(differences, seed),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generation-batch-size", type=int, default=24)
    parser.add_argument("--scoring-batch-size", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--force-generation", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
    else:
        rows = [
            json.loads(line)
            for line in args.input.read_text().splitlines()
            if line.strip()
        ]
        rows = generate_controls(args, rows)
        generations_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    score_inputs = [
        base.make_score_record(
            row["question"],
            row["choices"],
            row["concept_control_responses"][condition],
        )
        for row in rows
        for condition in NEW_CONDITIONS
    ]
    score_sets = base.score_records(args, score_inputs)
    attach_scores(rows, score_sets)
    scored_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    results = summarize(rows, args.seed)
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
