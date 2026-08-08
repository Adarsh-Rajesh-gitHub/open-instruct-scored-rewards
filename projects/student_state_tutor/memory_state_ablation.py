"""Test student state as persistent memory across multiple tutoring turns."""

from __future__ import annotations

import argparse
import gc
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from projects.student_state_tutor import graph_state_ablation as graph
from projects.student_state_tutor import oracle_state_ablation as base


CONDITIONS = ("full_history", "last_only", "last_concept", "last_graph")


def dialogue_turns(completion: str) -> list[tuple[str, str]]:
    turns = []
    for match in re.finditer(
        r"(?ms)^(Student|Tutor):\s*(.*?)(?=^(?:Student|Tutor):|\Z)",
        completion,
    ):
        text = " ".join(match.group(2).strip().split())
        if text:
            turns.append((match.group(1), text))
    return turns


def load_memory_examples(
    traces_path: Path, graph_path: Path, limit: int
) -> list[dict]:
    graph_rows = {
        row["question"]: row
        for row in (
            json.loads(line)
            for line in graph_path.read_text().splitlines()
            if line.strip()
        )
    }
    best_by_question = {}
    for line in traces_path.read_text().splitlines():
        if not line.strip():
            continue
        trace = json.loads(line)
        question = trace.get("prompt")
        if question not in graph_rows or float(trace.get("solved", 1.0)) != 0.0:
            continue
        turns = dialogue_turns(trace.get("completion", ""))
        student_turns = [text for role, text in turns if role == "Student"]
        tutor_turns = [text for role, text in turns if role == "Tutor"]
        if len(student_turns) < 2 or not tutor_turns or turns[-1][0] != "Student":
            continue
        candidate = {
            **graph_rows[question],
            "full_history": trace["completion"].strip(),
            "last_student_turn": student_turns[-1],
            "past_tutor_context": "\n".join(tutor_turns),
            "history_student_turns": len(student_turns),
        }
        previous = best_by_question.get(question)
        if previous is None or candidate["history_student_turns"] > previous["history_student_turns"]:
            best_by_question[question] = candidate
    examples = sorted(
        best_by_question.values(),
        key=lambda row: (-row["history_student_turns"], row["question"]),
    )
    return examples[:limit] if limit > 0 else examples


def tutor_prompt(tokenizer, row: dict, condition: str) -> str:
    question = (
        f"Question:\n{row['question']}\n"
        f"{base.format_choices(row['choices'])}\n\n"
    )
    if condition == "full_history":
        context = f"Conversation so far:\n{row['full_history']}\n\n"
    else:
        context = f"Student's latest message:\n{row['last_student_turn']}\n\n"
    state = ""
    if condition == "last_concept":
        state = (
            "Private persistent student-state memory:\n"
            f"concept = {row['graph_actual']['concept']}\n\n"
        )
    elif condition == "last_graph":
        state = (
            "Private persistent student-state memory:\n"
            f"{graph.state_text(row['graph_actual'])}\n\n"
        )
    user = (
        question
        + context
        + state
        + "Write the tutor's next message. Do not repeat the private memory."
    )
    return graph.render_chat(tokenizer, base.TUTOR_SYSTEM, user)


@torch.inference_mode()
def generate_responses(args, rows: list[dict]) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, dtype=torch.bfloat16
    ).to(args.device)
    model.eval()
    requests = [
        (index, condition, tutor_prompt(tokenizer, row, condition))
        for index, row in enumerate(rows)
        for condition in CONDITIONS
    ]
    outputs = graph.batch_generate(
        model,
        tokenizer,
        requests,
        args.generation_batch_size,
        args.max_input_tokens,
        args.max_new_tokens,
        args.device,
    )
    for row, output_map in zip(rows, outputs, strict=True):
        row["memory_tutor_responses"] = output_map
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def attach_scores(rows: list[dict], score_sets: list[list[float]]) -> None:
    cursor = 0
    for row in rows:
        baseline_scores = score_sets[cursor]
        cursor += 1
        baseline_probabilities = base.softmax(baseline_scores)
        row["memory_baseline"] = {
            "scores": baseline_scores,
            "gold_probability": float(baseline_probabilities[row["gold_idx"]]),
            "belief_margin": float(
                baseline_scores[row["gold_idx"]]
                - baseline_scores[row["belief_idx"]]
            ),
            "solved": int(np.argmax(baseline_scores) == row["gold_idx"]),
        }
        row["memory_condition_scores"] = {}
        for condition in CONDITIONS:
            scores = score_sets[cursor]
            cursor += 1
            probabilities = base.softmax(scores)
            response = row["memory_tutor_responses"][condition]
            row["memory_condition_scores"][condition] = {
                "scores": scores,
                "gold_probability": float(probabilities[row["gold_idx"]]),
                "belief_margin": float(
                    scores[row["gold_idx"]] - scores[row["belief_idx"]]
                ),
                "solved": int(np.argmax(scores) == row["gold_idx"]),
                "leaked": base.leaks_answer(response, row),
            }


def summarize(rows: list[dict], seed: int) -> dict:
    result = {
        "n": len(rows),
        "subjects": dict(Counter(row["subject"] for row in rows)),
        "mean_student_turns": float(
            np.mean([row["history_student_turns"] for row in rows])
        ),
        "baseline": {
            name: float(np.mean([row["memory_baseline"][name] for row in rows]))
            for name in ("gold_probability", "belief_margin", "solved")
        },
        "conditions": {},
        "paired_comparisons": {},
    }
    for condition in CONDITIONS:
        result["conditions"][condition] = {
            name: float(
                np.mean(
                    [
                        row["memory_condition_scores"][condition][name]
                        for row in rows
                    ]
                )
            )
            for name in ("gold_probability", "belief_margin", "solved", "leaked")
        }
    for left, right in (
        ("last_graph", "last_concept"),
        ("last_graph", "last_only"),
        ("last_graph", "full_history"),
        ("full_history", "last_only"),
    ):
        comparison = {}
        for name in ("gold_probability", "belief_margin", "solved"):
            differences = np.asarray(
                [
                    row["memory_condition_scores"][left][name]
                    - row["memory_condition_scores"][right][name]
                    for row in rows
                ],
                dtype=np.float64,
            )
            comparison[name] = {
                "mean": float(differences.mean()),
                "95_ci": base.bootstrap_interval(differences, seed),
            }
        result["paired_comparisons"][f"{left}_minus_{right}"] = comparison
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--graph-input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generation-batch-size", type=int, default=24)
    parser.add_argument("--scoring-batch-size", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
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
        rows = load_memory_examples(args.traces, args.graph_input, args.limit)
        if not rows:
            raise ValueError("no unresolved multi-turn examples found")
        print(
            f"loaded {len(rows)} examples: "
            f"{dict(Counter(row['subject'] for row in rows))}",
            flush=True,
        )
        rows = generate_responses(args, rows)
        generations_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    score_inputs = []
    for row in rows:
        score_inputs.append(
            base.make_score_record(
                row["question"], row["choices"], row["past_tutor_context"]
            )
        )
        for condition in CONDITIONS:
            combined_hint = (
                row["past_tutor_context"]
                + "\n"
                + row["memory_tutor_responses"][condition]
            )
            score_inputs.append(
                base.make_score_record(
                    row["question"], row["choices"], combined_hint
                )
            )
    score_sets = base.score_records(args, score_inputs)
    attach_scores(rows, score_sets)
    scored_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    results = summarize(rows, args.seed)
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
