"""Evaluate graph-conditioned teacher actions on held-out isomorphic items."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from projects.student_state_tutor.oracle_state_ablation import (
    bootstrap_interval,
    leaks_answer,
    make_score_record,
    score_records,
    softmax,
)
from projects.student_state_tutor.teacher_action_ablation import CONDITIONS


GENERATION_SYSTEM = """You are a patient physics tutor.
Realize the supplied structured action as one concise intervention of at most
three sentences. Help the student reason about the selected concept. Do not
reveal an answer, quote an answer choice, name an option letter, or solve the
held-out transfer problem. If the action targets an unrelated concept, stay
faithful to that selected action rather than silently changing it."""


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def flatten_questions(banks: list[dict]) -> list[dict]:
    rows = []
    for bank in banks:
        for question in bank["questions"]:
            rows.append(
                {
                    "concept_id": bank["concept_id"],
                    "bank_title": bank["title"],
                    "bank_description": bank.get("description", ""),
                    **question,
                }
            )
    return rows


def attach_baseline_scores(questions: list[dict], scores: list[list[float]]) -> None:
    for question, values in zip(questions, scores, strict=True):
        probabilities = softmax(values)
        predicted = int(np.argmax(values))
        question["baseline"] = {
            "scores": values,
            "predicted_idx": predicted,
            "gold_probability": float(probabilities[question["gold_idx"]]),
            "solved": int(predicted == question["gold_idx"]),
        }


def screen_questions(args, banks: list[dict]) -> list[dict]:
    questions = flatten_questions(banks)
    records = [
        make_score_record(row["question"], row["choices"], "")
        for row in questions
    ]
    scores = score_records(args, records)
    attach_baseline_scores(questions, scores)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return questions


def build_transfer_rows(
    banks: list[dict],
    action_rows: list[dict],
    screened_questions: list[dict],
    limit: int,
) -> tuple[list[dict], dict]:
    bank_map = {bank["concept_id"]: bank for bank in banks}
    wrong_by_concept: dict[str, list[dict]] = {}
    for concept_id in bank_map:
        wrong_by_concept[concept_id] = [
            row
            for row in screened_questions
            if row["concept_id"] == concept_id and not row["baseline"]["solved"]
        ]
    eligible = {
        concept_id
        for concept_id, rows in wrong_by_concept.items()
        if len(rows) >= 2
    }
    output = []
    for action_row in action_rows:
        concept_id = action_row["oracle_target"]
        if concept_id not in eligible:
            continue
        wrong = wrong_by_concept[concept_id]
        offset = len(output) % len(wrong)
        diagnostic = wrong[offset]
        transfer = wrong[(offset + 1) % len(wrong)]
        if diagnostic["question_id"] == transfer["question_id"]:
            continue
        belief_idx = diagnostic["baseline"]["predicted_idx"]
        output.append(
            {
                "case_id": action_row["case_id"],
                "concept_id": concept_id,
                "bank_title": bank_map[concept_id]["title"],
                "diagnostic": diagnostic,
                "transfer": transfer,
                "belief_idx": belief_idx,
                "student_belief": diagnostic["choices"][belief_idx],
                "decisions": action_row["decisions"],
                "candidate_banks": {
                    bank["concept_id"]: {
                        "title": bank["title"],
                        "description": bank.get("description", ""),
                        "practice_question": bank["questions"][0]["question"],
                    }
                    for bank in banks
                },
            }
        )
        if limit > 0 and len(output) >= limit:
            break
    stats = {
        "screened_questions": len(screened_questions),
        "baseline_solved": sum(
            row["baseline"]["solved"] for row in screened_questions
        ),
        "wrong_by_concept": {
            concept_id: len(rows)
            for concept_id, rows in wrong_by_concept.items()
        },
        "eligible_concepts": sorted(eligible),
        "transfer_rows": len(output),
    }
    return output, stats


def generation_user_prompt(row: dict, condition: str) -> str:
    decision = row["decisions"][condition]
    target = decision.get("target_node")
    selected_bank = row["candidate_banks"].get(target, {})
    return (
        f"Student diagnostic question:\n{row['diagnostic']['question']}\n\n"
        f"Student answered:\n{row['student_belief']}\n\n"
        f"Structured policy decision:\n"
        f"{json.dumps({key: decision.get(key) for key in ('target_node', 'action', 'item_id', 'stop')}, indent=2)}\n\n"
        f"Selected concept title:\n{selected_bank.get('title', 'invalid selection')}\n\n"
        f"Selected concept description:\n{selected_bank.get('description', '')}\n\n"
        f"Representative practice question for the selected concept:\n"
        f"{selected_bank.get('practice_question', '')}\n\n"
        "Write the tutor intervention only."
    )


def render_prompt(tokenizer, row: dict, condition: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": GENERATION_SYSTEM},
            {"role": "user", "content": generation_user_prompt(row, condition)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def generate_interventions(args, rows: list[dict]) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
    ).to(args.device)
    if args.teacher_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.teacher_adapter)
    model.eval()
    requests = [
        (row_index, condition, render_prompt(tokenizer, row, condition))
        for row_index, row in enumerate(rows)
        for condition in CONDITIONS
    ]
    for row in rows:
        row["interventions"] = {}
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
        for (row_index, condition, _), text in zip(batch, decoded, strict=True):
            rows[row_index]["interventions"][condition] = text.strip()
        print(
            f"generated {min(start + len(batch), len(requests))}/{len(requests)} interventions",
            flush=True,
        )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def canonical_intervention(row: dict) -> str:
    feedback = row["diagnostic"].get("feedback", "").strip()
    if feedback:
        return feedback
    return (
        f"Review the core principle behind {row['bank_title']}. "
        "Explain why it applies before attempting a new example."
    )


def score_transfer(args, rows: list[dict]) -> None:
    records = []
    for row in rows:
        transfer = row["transfer"]
        records.append(
            make_score_record(transfer["question"], transfer["choices"], "")
        )
        for condition in CONDITIONS:
            records.append(
                make_score_record(
                    transfer["question"],
                    transfer["choices"],
                    row["interventions"][condition],
                )
            )
        row["interventions"]["canonical"] = canonical_intervention(row)
        records.append(
            make_score_record(
                transfer["question"],
                transfer["choices"],
                row["interventions"]["canonical"],
            )
        )
    score_sets = score_records(args, records)
    cursor = 0
    for row in rows:
        transfer = row["transfer"]
        baseline_values = score_sets[cursor]
        cursor += 1
        baseline_probabilities = softmax(baseline_values)
        row["transfer_baseline"] = {
            "scores": baseline_values,
            "gold_probability": float(
                baseline_probabilities[transfer["gold_idx"]]
            ),
            "solved": int(
                np.argmax(baseline_values) == transfer["gold_idx"]
            ),
        }
        row["transfer_scores"] = {}
        for condition in (*CONDITIONS, "canonical"):
            values = score_sets[cursor]
            cursor += 1
            probabilities = softmax(values)
            response = row["interventions"][condition]
            leak_example = {
                "choices": transfer["choices"],
                "gold_idx": transfer["gold_idx"],
            }
            row["transfer_scores"][condition] = {
                "scores": values,
                "gold_probability": float(
                    probabilities[transfer["gold_idx"]]
                ),
                "solved": int(np.argmax(values) == transfer["gold_idx"]),
                "leaked": leaks_answer(response, leak_example),
            }


def paired(rows: list[dict], left: str, right: str, metric: str) -> np.ndarray:
    return np.asarray(
        [
            row["transfer_scores"][left][metric]
            - row["transfer_scores"][right][metric]
            for row in rows
        ],
        dtype=np.float64,
    )


def summarize(rows: list[dict], screen_stats: dict, seed: int) -> dict:
    conditions = {}
    for condition in (*CONDITIONS, "canonical"):
        conditions[condition] = {
            metric: float(
                np.mean(
                    [
                        row["transfer_scores"][condition][metric]
                        for row in rows
                    ]
                )
            )
            for metric in ("gold_probability", "solved", "leaked")
        }
    comparisons = {}
    for right in ("latest", "full_history", "concept_only", "shuffled_graph", "canonical"):
        name = f"graph_minus_{right}"
        comparisons[name] = {}
        for metric in ("gold_probability", "solved"):
            values = paired(rows, "graph", right, metric)
            comparisons[name][metric] = {
                "mean": float(values.mean()),
                "95_ci": bootstrap_interval(values, seed),
            }
    graph_by_concept = {}
    for concept_id in sorted({row["concept_id"] for row in rows}):
        selected = [row for row in rows if row["concept_id"] == concept_id]
        graph_by_concept[concept_id] = {
            "n": len(selected),
            "graph_gold_probability": float(
                np.mean(
                    [
                        row["transfer_scores"]["graph"]["gold_probability"]
                        for row in selected
                    ]
                )
            ),
            "canonical_gold_probability": float(
                np.mean(
                    [
                        row["transfer_scores"]["canonical"]["gold_probability"]
                        for row in selected
                    ]
                )
            ),
        }
    graph_vs_latest = comparisons["graph_minus_latest"]["gold_probability"]
    graph_vs_canonical = comparisons["graph_minus_canonical"]["gold_probability"]
    gate = {
        "enough_rows": len(rows) >= 16,
        "beats_latest": graph_vs_latest["mean"] > 0.0
        and graph_vs_latest["95_ci"][0] >= 0.0,
        "beats_canonical": graph_vs_canonical["mean"] > 0.0,
        "low_leakage": conditions["graph"]["leaked"] <= 0.05,
        "concept_coverage": len(graph_by_concept) >= 3,
    }
    return {
        "n": len(rows),
        "screening": screen_stats,
        "baseline_transfer_solve_rate": float(
            np.mean([row["transfer_baseline"]["solved"] for row in rows])
        ),
        "conditions": conditions,
        "paired_comparisons": comparisons,
        "by_concept": graph_by_concept,
        "gate_checks": gate,
        "gate_passed": all(gate.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--action-rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--teacher-model", default="Qwen/Qwen2.5-3B-Instruct"
    )
    parser.add_argument("--teacher-adapter")
    parser.add_argument(
        "--student-model", default="Qwen/Qwen2.5-0.5B-Instruct"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--scoring-batch-size", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    banks = load_jsonl(args.banks)
    action_rows = load_jsonl(args.action_rows)
    screened = screen_questions(args, banks)
    rows, screen_stats = build_transfer_rows(
        banks, action_rows, screened, args.limit
    )
    if not rows:
        raise ValueError(
            "no transfer rows: student needs at least two baseline errors per bank"
        )
    generate_interventions(args, rows)
    score_transfer(args, rows)
    results = summarize(rows, screen_stats, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "scored.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (args.out_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
