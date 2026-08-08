"""Test whether a frozen teacher uses a compact mastery graph correctly."""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from projects.student_state_tutor.mastery_state import EvidenceEvent, MasteryGraphState
from transformers import AutoModelForCausalLM, AutoTokenizer

CONDITIONS = ("latest", "full_history", "concept_only", "graph", "shuffled_graph")
ACTIONS = ("diagnostic", "worked_example", "contrast_case", "retrieval_practice", "stop")
SYSTEM = """You are selecting the next action for a physics tutor.
Return exactly one JSON object with keys target_node, action, item_id, and stop.
Use only the supplied student evidence. Choose one candidate concept and an item
from that concept. Do not solve any physics problem or add explanatory text.

Action policy:
- diagnostic: no unaided evidence exists for the target.
- worked_example: repeated unaided failures and posterior mean below 0.30.
- contrast_case: posterior mean from 0.30 through 0.49.
- retrieval_practice: posterior mean from 0.50 through 0.74.
- stop: all concepts have posterior mean at least 0.75.
Target the lowest posterior mean; break ties by wider interval."""


def load_banks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def action_for_belief(mean: float, unaided_n: int) -> str:
    if unaided_n == 0:
        return "diagnostic"
    if mean < 0.30:
        return "worked_example"
    if mean < 0.50:
        return "contrast_case"
    if mean < 0.75:
        return "retrieval_practice"
    return "stop"


def add_events(
    graph: MasteryGraphState, concept_id: str, successes: int, failures: int, event_order: list[dict]
) -> None:
    outcomes = [True] * successes + [False] * failures
    for index, correct in enumerate(outcomes):
        event = EvidenceEvent(
            learner_id=graph.learner_id,
            concept_id=concept_id,
            item_id=f"{concept_id}_diagnostic_{index}",
            correct=correct,
            timestamp=f"2026-08-06T00:{len(event_order):02d}:00+00:00",
        )
        graph.observe(event)
        event_order.append(
            {"concept_id": concept_id, "item_id": event.item_id, "correct": correct, "assistance": "unaided"}
        )


def build_cases(banks: list[dict], replicates: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    concept_ids = [bank["concept_id"] for bank in banks]
    bank_map = {bank["concept_id"]: bank for bank in banks}
    cases = []
    profiles = ((0, 4), (1, 3), (2, 3), (3, 2))
    for target_index, target in enumerate(concept_ids):
        for replicate in range(replicates):
            graph = MasteryGraphState(
                learner_id=f"learner_{target_index:02d}_{replicate:03d}", concept_ids=concept_ids
            )
            events = []
            target_successes, target_failures = profiles[replicate % len(profiles)]
            add_events(graph, target, target_successes, target_failures, events)
            for concept_id in concept_ids:
                if concept_id == target:
                    continue
                successes = 3 + rng.randint(0, 2)
                failures = rng.randint(0, 1)
                add_events(graph, concept_id, successes, failures, events)
            rng.shuffle(events)

            actual_target = graph.weakest()
            actual_belief = graph.belief(actual_target)
            shuffled_ids = concept_ids[:]
            for _ in range(20):
                rng.shuffle(shuffled_ids)
                if shuffled_ids != concept_ids:
                    break
            shuffled_nodes = []
            for source_id, destination_id in zip(concept_ids, shuffled_ids):
                node = graph.belief(source_id).to_view()
                node["concept_id"] = destination_id
                shuffled_nodes.append(node)
            shuffled_target = min(
                shuffled_nodes,
                key=lambda node: (node["mean"], -(node["interval"][1] - node["interval"][0]), node["concept_id"]),
            )["concept_id"]

            candidates = [
                {
                    "concept_id": concept_id,
                    "title": bank_map[concept_id]["title"],
                    "item_id": (f"{concept_id}:{bank_map[concept_id]['questions'][0]['question_id']}"),
                }
                for concept_id in concept_ids
            ]
            cases.append(
                {
                    "case_id": f"case_{target_index:02d}_{replicate:03d}",
                    "learner_id": graph.learner_id,
                    "candidates": candidates,
                    "events": events,
                    "latest_event": events[-1],
                    "graph_view": graph.local_view(),
                    "shuffled_graph_view": {
                        "learner_id": graph.learner_id,
                        "nodes": shuffled_nodes,
                        "event_count": len(events),
                    },
                    "oracle_target": actual_target,
                    "oracle_action": action_for_belief(actual_belief.mean, actual_belief.unaided_n),
                    "shuffled_visible_target": shuffled_target,
                }
            )
    rng.shuffle(cases)
    return cases


def candidate_text(case: dict) -> str:
    return json.dumps(case["candidates"], indent=2)


def user_prompt(case: dict, condition: str) -> str:
    if condition == "latest":
        evidence = {"latest_event": case["latest_event"]}
    elif condition == "full_history":
        evidence = {"history": case["events"]}
    elif condition == "concept_only":
        evidence = {
            "concepts": [{"concept_id": row["concept_id"], "title": row["title"]} for row in case["candidates"]]
        }
    elif condition == "graph":
        evidence = case["graph_view"]
    elif condition == "shuffled_graph":
        evidence = case["shuffled_graph_view"]
    else:
        raise ValueError(f"unknown condition {condition}")
    return (
        f"Candidate concepts and items:\n{candidate_text(case)}\n\n"
        f"Student evidence:\n{json.dumps(evidence, indent=2)}\n\n"
        "Choose the next tutoring action. Return JSON only."
    )


def render_prompt(tokenizer, case: dict, condition: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(case, condition)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_decision(text: str, case: dict) -> dict:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match is None:
        return {"valid": False, "raw": text}
    try:
        decision = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"valid": False, "raw": text}
    concept_to_items = {row["concept_id"]: {row["item_id"]} for row in case["candidates"]}
    target = decision.get("target_node")
    action = decision.get("action")
    item_id = decision.get("item_id")
    stop = decision.get("stop")
    valid = (
        target in concept_to_items
        and action in ACTIONS
        and isinstance(stop, bool)
        and ((action == "stop" and stop) or (action != "stop" and not stop and item_id in concept_to_items[target]))
    )
    return {"valid": valid, "target_node": target, "action": action, "item_id": item_id, "stop": stop, "raw": text}


@torch.inference_mode()
def generate_decisions(args, cases: list[dict]) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, dtype=torch.bfloat16 if args.device == "cuda" else torch.float32
    ).to(args.device)
    if args.teacher_adapter:
        from peft import PeftModel  # noqa: PLC0415

        model = PeftModel.from_pretrained(model, args.teacher_adapter)
    model.eval()
    requests = [
        (case_index, condition, render_prompt(tokenizer, case, condition))
        for case_index, case in enumerate(cases)
        for condition in CONDITIONS
    ]
    outputs = [dict() for _ in cases]
    for start in range(0, len(requests), args.batch_size):
        batch = requests[start : start + args.batch_size]
        encoded = tokenizer(
            [prompt for _, _, prompt in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(args.device)
        generated = model.generate(
            **encoded, do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id
        )
        prompt_width = encoded.input_ids.shape[1]
        decoded = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
        for (case_index, condition, _), text in zip(batch, decoded, strict=True):
            outputs[case_index][condition] = parse_decision(text.strip(), cases[case_index])
        print(f"generated {min(start + len(batch), len(requests))}/{len(requests)}", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [{**case, "decisions": decisions} for case, decisions in zip(cases, outputs, strict=True)]


def mean(values: list[bool]) -> float:
    return float(np.mean(values)) if values else float("nan")


def summarize(rows: list[dict]) -> dict:
    conditions = {}
    for condition in CONDITIONS:
        decisions = [row["decisions"][condition] for row in rows]
        valid = [decision["valid"] for decision in decisions]
        target_correct = [
            decision["target_node"] == row["oracle_target"]
            for row, decision in zip(rows, decisions, strict=True)
            if decision["valid"]
        ]
        action_correct = [
            decision["action"] == row["oracle_action"]
            for row, decision in zip(rows, decisions, strict=True)
            if decision["valid"] and decision["target_node"] == row["oracle_target"]
        ]
        conditions[condition] = {
            "valid_rate": mean(valid),
            "target_accuracy": mean(target_correct),
            "action_accuracy_given_target": mean(action_correct),
        }
    shuffled_follow = [
        row["decisions"]["shuffled_graph"]["target_node"] == row["shuffled_visible_target"]
        for row in rows
        if row["decisions"]["shuffled_graph"]["valid"]
    ]
    changed_cases = [
        row
        for row in rows
        if row["shuffled_visible_target"] != row["oracle_target"]
        and row["decisions"]["graph"]["valid"]
        and row["decisions"]["shuffled_graph"]["valid"]
    ]
    sensitivity = [
        row["decisions"]["graph"]["target_node"] != row["decisions"]["shuffled_graph"]["target_node"]
        for row in changed_cases
    ]
    gate = {
        "valid": conditions["graph"]["valid_rate"] >= 0.95,
        "target": conditions["graph"]["target_accuracy"] >= 0.80,
        "action": conditions["graph"]["action_accuracy_given_target"] >= 0.60,
        "beats_latest": (conditions["graph"]["target_accuracy"] - conditions["latest"]["target_accuracy"] >= 0.20),
        "uses_visible_state": mean(shuffled_follow) >= 0.80 and mean(sensitivity) >= 0.70,
    }
    return {
        "cases": len(rows),
        "conditions": conditions,
        "shuffled_visible_target_accuracy": mean(shuffled_follow),
        "graph_shuffled_target_sensitivity": mean(sensitivity),
        "gate_checks": gate,
        "gate_passed": all(gate.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--teacher-adapter")
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    banks = load_banks(args.banks)
    cases = build_cases(banks, args.replicates, args.seed)
    rows = generate_decisions(args, cases)
    summary = summarize(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (args.out_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
