"""Test an abstract graph-style student state without answer-option wording.

This follows the failed raw-belief ablation. A frozen diagnostician first maps
the contrast between a wrong and correct choice into:

    concept -> mistaken_relation

The tutor receives that abstract relation, never the wrong answer text. We
compare actual and within-item counterfactual relations against the cached
transcript-only tutor response.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from projects.student_state_tutor import oracle_state_ablation as base
from transformers import AutoModelForCausalLM, AutoTokenizer

GRAPH_CONDITIONS = ("graph_actual", "graph_counterfactual")
DIAGNOSTIC_SYSTEM = """You label conceptual student misconceptions.
Return exactly two short lines:
CONCEPT: <2-6 word concept>
MISTAKEN_RELATION: <an abstract claim describing the conceptual confusion>

The relation must generalize to a new question. Do not quote or closely
paraphrase either answer choice, use option letters, give the correct answer, or
write tutoring advice."""


def diagnostic_prompt(row: dict, belief_key: str) -> str:
    return (
        f"Question:\n{row['question']}\n\n"
        f"Correct choice used only to identify the contrast:\n"
        f"{row['choices'][row['gold_idx']]}\n\n"
        f"Student's wrong choice used only to identify the contrast:\n"
        f"{row[belief_key]}\n\n"
        "Label the underlying conceptual confusion."
    )


def parse_graph_state(text: str) -> dict | None:
    concept = re.search(r"(?im)^\s*CONCEPT\s*:\s*(.+?)\s*$", text)
    relation = re.search(r"(?im)^\s*MISTAKEN_RELATION\s*:\s*(.+?)\s*$", text)
    if not concept or not relation:
        return None
    return {
        "concept": concept.group(1).strip().strip("\"'`"),
        "mistaken_relation": relation.group(1).strip().strip("\"'`"),
        "raw": text.strip(),
    }


def state_text(state: dict) -> str:
    return f"concept = {state['concept']}\nstudent_belief_relation = {state['mistaken_relation']}"


def has_option_overlap(state: dict | None, row: dict) -> bool:
    if state is None:
        return True
    normalized_state = base.normalize_text(state_text(state))
    for choice_index in (row["gold_idx"], row["belief_idx"], row["counterfactual_idx"]):
        choice = base.normalize_text(row["choices"][choice_index])
        if len(choice) >= 8 and choice in normalized_state:
            return True
    return False


def render_chat(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def batch_generate(
    model,
    tokenizer,
    requests: list[tuple[int, str, str]],
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    device: str,
) -> list[dict[str, str]]:
    responses: list[dict[str, str]] = [dict() for _ in range(max(r[0] for r in requests) + 1)]
    for start in range(0, len(requests), batch_size):
        batch = requests[start : start + batch_size]
        encoded = tokenizer(
            [prompt for _, _, prompt in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to(device)
        generated = model.generate(
            **encoded, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id
        )
        prompt_width = encoded.input_ids.shape[1]
        decoded = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
        for (row_index, key, _), response in zip(batch, decoded, strict=True):
            responses[row_index][key] = response.strip()
        print(f"generated {min(start + len(batch), len(requests))}/{len(requests)}", flush=True)
    return responses


def graph_tutor_prompt(tokenizer, row: dict, state: dict) -> str:
    private_state = state_text(state)
    user = (
        f"Question:\n{row['question']}\n"
        f"{base.format_choices(row['choices'])}\n\n"
        f"Student said:\n{row['student_opening']}\n\n"
        "Private abstract student-state graph:\n"
        f"{private_state}\n\n"
        "Use the graph only to select a useful intervention. Do not repeat its "
        "phrasing. Write the tutor's next message."
    )
    return render_chat(tokenizer, base.TUTOR_SYSTEM, user)


def generate_graphs_and_tutors(args, rows: list[dict]) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.teacher_model, dtype=torch.bfloat16).to(args.device)
    model.eval()

    diagnostic_requests = []
    for index, row in enumerate(rows):
        for key, belief_key in (("actual", "belief"), ("counterfactual", "counterfactual_belief")):
            prompt = render_chat(tokenizer, DIAGNOSTIC_SYSTEM, diagnostic_prompt(row, belief_key))
            diagnostic_requests.append((index, key, prompt))
    diagnostic_outputs = batch_generate(
        model,
        tokenizer,
        diagnostic_requests,
        args.generation_batch_size,
        args.max_input_tokens,
        args.graph_max_new_tokens,
        args.device,
    )

    valid_rows = []
    for row, output_map in zip(rows, diagnostic_outputs, strict=True):
        graph_actual = parse_graph_state(output_map["actual"])
        graph_counterfactual = parse_graph_state(output_map["counterfactual"])
        enriched = {
            **row,
            "graph_actual": graph_actual,
            "graph_counterfactual": graph_counterfactual,
            "graph_actual_overlap": has_option_overlap(graph_actual, row),
            "graph_counterfactual_overlap": has_option_overlap(graph_counterfactual, row),
        }
        if (
            graph_actual is not None
            and graph_counterfactual is not None
            and not enriched["graph_actual_overlap"]
            and not enriched["graph_counterfactual_overlap"]
        ):
            valid_rows.append(enriched)

    print(f"valid graph pairs {len(valid_rows)}/{len(rows)}", flush=True)
    if not valid_rows:
        raise ValueError("no valid graph-state pairs")

    tutor_requests = []
    for index, row in enumerate(valid_rows):
        for condition in GRAPH_CONDITIONS:
            tutor_requests.append((index, condition, graph_tutor_prompt(tokenizer, row, row[condition])))
    tutor_outputs = batch_generate(
        model,
        tokenizer,
        tutor_requests,
        args.generation_batch_size,
        args.max_input_tokens,
        args.tutor_max_new_tokens,
        args.device,
    )
    for row, output_map in zip(valid_rows, tutor_outputs, strict=True):
        row["graph_tutor_responses"] = output_map

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return valid_rows


def attach_graph_scores(rows: list[dict], score_sets: list[list[float]]) -> None:
    for row_index, row in enumerate(rows):
        row["graph_condition_scores"] = {}
        for condition_index, condition in enumerate(GRAPH_CONDITIONS):
            scores = score_sets[row_index * len(GRAPH_CONDITIONS) + condition_index]
            probabilities = base.softmax(scores)
            response = row["graph_tutor_responses"][condition]
            row["graph_condition_scores"][condition] = {
                "scores": scores,
                "gold_probability": float(probabilities[row["gold_idx"]]),
                "belief_margin": float(scores[row["gold_idx"]] - scores[row["belief_idx"]]),
                "solved": int(np.argmax(scores) == row["gold_idx"]),
                "leaked": base.leaks_answer(response, row),
            }


def metric(row: dict, condition: str, name: str) -> float:
    if condition == "transcript":
        return float(row["condition_scores"]["transcript"][name])
    return float(row["graph_condition_scores"][condition][name])


def paired(rows: list[dict], left: str, right: str, name: str) -> np.ndarray:
    return np.asarray([metric(row, left, name) - metric(row, right, name) for row in rows], dtype=np.float64)


def summarize(rows: list[dict], seed: int) -> dict:
    conditions = ("transcript",) + GRAPH_CONDITIONS
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
    for left, right in (("graph_actual", "transcript"), ("graph_actual", "graph_counterfactual")):
        comparison = {}
        for name in ("gold_probability", "belief_margin", "solved"):
            differences = paired(rows, left, right, name)
            comparison[name] = {"mean": float(differences.mean()), "95_ci": base.bootstrap_interval(differences, seed)}
        result["paired_comparisons"][f"{left}_minus_{right}"] = comparison
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
    parser.add_argument("--graph-max-new-tokens", type=int, default=64)
    parser.add_argument("--tutor-max-new-tokens", type=int, default=96)
    parser.add_argument("--force-generation", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    generations_path = args.out_dir / "generations.jsonl"
    scored_path = args.out_dir / "scored.jsonl"
    results_path = args.out_dir / "results.json"

    if generations_path.exists() and not args.force_generation:
        rows = [json.loads(line) for line in generations_path.read_text().splitlines() if line.strip()]
        print(f"loaded {len(rows)} graph generations", flush=True)
    else:
        source_rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
        rows = generate_graphs_and_tutors(args, source_rows)
        generations_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    score_inputs = []
    for row in rows:
        for condition in GRAPH_CONDITIONS:
            score_inputs.append(
                base.make_score_record(row["question"], row["choices"], row["graph_tutor_responses"][condition])
            )
    score_sets = base.score_records(args, score_inputs)
    attach_graph_scores(rows, score_sets)
    scored_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    results = summarize(rows, args.seed)
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
