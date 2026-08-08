"""Pilot: can clean observer hidden states classify naturally selected math errors?

The student is never told which error to make. Each multiple-choice distractor is
generated from a known error transformation, and the selected option is labelled
after generation. The observer gets a fresh forward pass over only the problem
and emitted response; it never sees the student prompt or generation activations.

Example:
    python -m projects.student_state_tutor.misconception_probe \
        --out-dir projects/student_state_tutor/runs/pilot \
        --n-problems 200 --samples-per-problem 3
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path


LABELS = ("correct", "sign_error", "operation_error", "off_by_one")
STUDENT_SYSTEM = """You are a student solving a short math multiple-choice problem.
Try the problem yourself. Explain your thinking briefly, then finish with
"Answer: X" where X is A, B, C, or D. Do not role-play a particular mistake."""
STUDENT_SYSTEM_FREE = """You are a student solving a short math problem.
Try the problem yourself without using a calculator. Start with "Answer: N"
where N is your integer answer, then explain your thinking in one short sentence.
Do not role-play a particular mistake."""
OBSERVER_SYSTEM = """You are a tutor reading a student's work. Read the problem
and the student's response carefully before deciding how to help."""


def make_problem(rng: random.Random, index: int) -> dict:
    """Make one problem whose distractors have known, class-consistent causes."""
    family = index % 4
    while True:
        a = rng.randint(3, 18)
        b = rng.randint(2, 12)
        if family == 0:
            stem = f"Evaluate {a} + (-{b})."
            values = {
                "correct": a - b,
                "sign_error": a + b,
                "operation_error": a * b,
                "off_by_one": a - b + 1,
            }
        elif family == 1:
            stem = f"Evaluate {a} - (-{b})."
            values = {
                "correct": a + b,
                "sign_error": a - b,
                "operation_error": a * b,
                "off_by_one": a + b + 1,
            }
        elif family == 2:
            stem = f"Evaluate (-{a}) × (-{b})."
            values = {
                "correct": a * b,
                "sign_error": -(a * b),
                "operation_error": a + b,
                "off_by_one": a * b + 1,
            }
        else:
            stem = f"Evaluate (-{a}) × {b}."
            values = {
                "correct": -(a * b),
                "sign_error": a * b,
                "operation_error": -(a + b),
                "off_by_one": -(a * b) - 1,
            }
        if len(set(values.values())) == len(LABELS):
            break

    ordered = list(values.items())
    rng.shuffle(ordered)
    letters = "ABCD"
    options = [{"letter": letters[i], "label": label, "value": value} for i, (label, value) in enumerate(ordered)]
    option_text = "\n".join(f"{o['letter']}. {o['value']}" for o in options)
    return {
        "problem_id": f"p{index:04d}",
        "family": family,
        "stem": stem,
        "options": options,
        "question": f"{stem}\n\n{option_text}",
        "free_question": stem,
    }


def parse_choice(response: str, problem: dict) -> str | None:
    """Return the selected option letter without using a model judge."""
    matches = re.findall(
        r"(?i)(?:(?:correct\s+)?answer|choice)\s*(?:is\s*)?[:=-]?\s*\(?([A-D])\)?",
        response,
    )
    if matches:
        return matches[-1].upper()

    final_line = response.strip().splitlines()[-1] if response.strip() else ""
    letter = re.fullmatch(r"\s*\(?([A-Da-d])\)?[.)]?\s*", final_line)
    if letter:
        return letter.group(1).upper()

    number_matches = re.findall(r"(?i)(?:answer|result|equals|=)\s*(?:is\s*)?[:=-]?\s*(-?\d+)", response)
    if number_matches:
        value = int(number_matches[-1])
        candidates = [o["letter"] for o in problem["options"] if o["value"] == value]
        if len(candidates) == 1:
            return candidates[0]
    return None


def option_label(problem: dict, letter: str | None) -> str | None:
    if letter is None:
        return None
    return next((o["label"] for o in problem["options"] if o["letter"] == letter), None)


def load_model(model_name: str, requested_device: str = "auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = requested_device
    # Keep non-CUDA execution in fp32. Some PyTorch/transformers combinations
    # still produce sampling NaNs on MPS; use --device cpu for that failure.
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device).eval()
    return model, tokenizer, device


def batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def generate_rows(args, model, tokenizer, device, problems: list[dict]) -> list[dict]:
    import torch

    prompts = []
    for problem in problems:
        if args.task_format == "free_response":
            system = STUDENT_SYSTEM_FREE
            question = problem["free_question"]
        else:
            system = STUDENT_SYSTEM
            question = problem["question"]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for sample_index in range(args.samples_per_problem):
            prompts.append((problem, question, sample_index, rendered))

    rows = []
    torch.manual_seed(args.seed)
    for batch_index, batch in enumerate(batched(prompts, args.generation_batch_size)):
        texts = [rendered for _, _, _, rendered in batch]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_width = encoded["input_ids"].shape[1]
        responses = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
        for (problem, question, sample_index, _), response in zip(batch, responses, strict=True):
            choice = parse_choice(response, problem)
            rows.append(
                {
                    **problem,
                    "question": question,
                    "sample_index": sample_index,
                    "response": response.strip(),
                    "choice": choice,
                    "label": option_label(problem, choice),
                }
            )
        if (batch_index + 1) % 10 == 0:
            print(f"generated {len(rows)}/{len(prompts)}")
    return rows


def observer_text(row: dict) -> str:
    return f"Problem:\n{row['question']}\n\nStudent response:\n{row['response']}"


def extract_hidden(args, model, tokenizer, device, rows: list[dict], out_path: Path) -> None:
    import numpy as np
    import torch

    usable = [row for row in rows if row["label"] in LABELS]
    n_layers = model.config.num_hidden_layers
    layers = sorted({round(fraction * n_layers) for fraction in (0.25, 0.5, 0.75, 1.0)})
    outputs = [[] for _ in layers]

    for batch_index, batch in enumerate(batched(usable, args.extraction_batch_size)):
        rendered = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": OBSERVER_SYSTEM},
                    {"role": "user", "content": observer_text(row)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            states = model(**encoded, output_hidden_states=True).hidden_states
        for output, layer in zip(outputs, layers, strict=True):
            output.extend(states[layer][:, -1].float().cpu().numpy())
        if (batch_index + 1) % 10 == 0:
            print(f"extracted {min((batch_index + 1) * args.extraction_batch_size, len(usable))}/{len(usable)}")

    np.savez_compressed(
        out_path,
        hidden=np.stack([np.stack(output) for output in outputs], axis=1).astype(np.float16),
        layers=np.asarray(layers),
        labels=np.asarray([row["label"] for row in usable]),
        groups=np.asarray([row["problem_id"] for row in usable]),
        texts=np.asarray([observer_text(row) for row in usable]),
        responses=np.asarray([row["response"] for row in usable]),
    )
    print(f"wrote {out_path}: {len(usable)} examples x {len(layers)} layers")


def simple_features(text: str) -> list[float]:
    words = text.split()
    response = text.partition("Student response:\n")[2]
    response_words = response.split()
    return [
        math.log1p(len(words)),
        math.log1p(len(response_words)),
        math.log1p(len(text)),
        response.count("?"),
        response.count("-"),
        response.count("+"),
        response.count("×"),
        sum(char.isdigit() for char in response),
        float("answer" in response.lower()),
        float(any(token in response.lower() for token in ("negative", "minus", "sign"))),
    ]


def make_folds(labels, groups, class_names, seed: int):
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    class_group_counts = [
        len(set(groups[index] for index, label in enumerate(labels) if label == class_name)) for class_name in class_names
    ]
    n_splits = min(5, min(class_group_counts))
    if n_splits < 2:
        raise ValueError(f"need at least two problem groups per class, got {dict(zip(class_names, class_group_counts))}")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(labels)), labels, groups))


def score_predictions(labels, predictions, class_names) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=class_names, zero_division=0
    )
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=class_names, average="macro", zero_division=0)),
    }
    result["per_class"] = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(class_names)
    }
    return result


def fit_dense(features, labels, folds):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    predictions = np.empty(len(labels), dtype=object)
    for train, test in folds:
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=0),
        )
        classifier.fit(features[train], labels[train])
        predictions[test] = classifier.predict(features[test])
    return predictions


def fit_tfidf(texts, labels, folds):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, make_pipeline

    predictions = np.empty(len(labels), dtype=object)
    for train, test in folds:
        vectorizer = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=20_000)),
            ]
        )
        classifier = make_pipeline(
            vectorizer,
            LogisticRegression(C=2.0, class_weight="balanced", max_iter=3000, random_state=0),
        )
        classifier.fit(texts[train], labels[train])
        predictions[test] = classifier.predict(texts[test])
    return predictions


def cluster_bootstrap_balanced_accuracy(
    labels, predictions, groups, class_names, seed: int, samples: int = 2000
) -> list[float]:
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    scores = []
    for _ in range(samples):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_rows[group] for group in selected])
        if len(set(labels[indices])) == len(class_names):
            scores.append(float(balanced_accuracy_score(labels[indices], predictions[indices])))
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def evaluate(args, hidden_path: Path, out_path: Path) -> dict:
    import numpy as np

    blob = np.load(hidden_path, allow_pickle=False)
    hidden = blob["hidden"].astype(np.float32)
    layers = [int(value) for value in blob["layers"]]
    labels = blob["labels"].astype(str)
    groups = blob["groups"].astype(str)
    texts = blob["texts"].astype(str)
    counts = Counter(labels)
    class_names = [label for label in LABELS if counts[label] > 0]
    folds = make_folds(labels, groups, class_names, args.seed)
    results: dict[str, dict] = {}

    majority = counts.most_common(1)[0][0]
    majority_predictions = np.asarray([majority] * len(labels))
    results["majority"] = score_predictions(labels, majority_predictions, class_names)

    simple = np.asarray([simple_features(text) for text in texts], dtype=np.float32)
    simple_predictions = fit_dense(simple, labels, folds)
    results["surface_simple"] = score_predictions(labels, simple_predictions, class_names)

    tfidf_predictions = fit_tfidf(texts, labels, folds)
    results["surface_tfidf"] = score_predictions(labels, tfidf_predictions, class_names)

    hidden_predictions = {}
    for layer_index, layer in enumerate(layers):
        predictions = fit_dense(hidden[:, layer_index, :], labels, folds)
        hidden_predictions[layer] = predictions
        results[f"hidden_layer_{layer}"] = score_predictions(labels, predictions, class_names)

    primary_layer = min(layers, key=lambda layer: abs(layer - max(layers) / 2))
    primary_predictions = hidden_predictions[primary_layer]
    ci = cluster_bootstrap_balanced_accuracy(labels, primary_predictions, groups, class_names, args.seed)
    results[f"hidden_layer_{primary_layer}"]["balanced_accuracy_95_ci"] = ci

    best_surface = max(results["surface_simple"]["balanced_accuracy"], results["surface_tfidf"]["balanced_accuracy"])
    results[f"hidden_layer_{primary_layer}"]["gain_over_best_surface"] = (
        results[f"hidden_layer_{primary_layer}"]["balanced_accuracy"] - best_surface
    )

    payload = {
        "model": args.model,
        "n_classified": len(labels),
        "class_counts": dict(counts),
        "classes_evaluated": class_names,
        "n_problem_groups": len(set(groups)),
        "primary_layer": primary_layer,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--task-format", default="free_response", choices=("free_response", "choices"))
    parser.add_argument("--n-problems", type=int, default=200)
    parser.add_argument("--samples-per-problem", type=int, default=3)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--extraction-batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "rows.jsonl"
    hidden_path = args.out_dir / "hidden.npz"
    results_path = args.out_dir / "results.json"

    rng = random.Random(args.seed)
    problems = [make_problem(rng, index) for index in range(args.n_problems)]
    model, tokenizer, device = load_model(args.model, args.device)
    print(f"model={args.model} device={device}")

    if rows_path.exists() and not args.force:
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        print(f"loaded {len(rows)} rows from {rows_path}")
        changed = 0
        for row in rows:
            choice = parse_choice(row["response"], row)
            label = option_label(row, choice)
            if choice != row.get("choice") or label != row.get("label"):
                row["choice"] = choice
                row["label"] = label
                changed += 1
        if changed:
            rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            hidden_path.unlink(missing_ok=True)
            results_path.unlink(missing_ok=True)
            print(f"relabelled {changed} rows and invalidated cached features")
    else:
        rows = generate_rows(args, model, tokenizer, device, problems)
        rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(f"wrote {rows_path}")

    counts = Counter(row["label"] or "unclassified" for row in rows)
    print(f"class counts: {dict(counts)}")

    if not hidden_path.exists() or args.force:
        extract_hidden(args, model, tokenizer, device, rows, hidden_path)
    else:
        print(f"using existing {hidden_path}")

    evaluate(args, hidden_path, results_path)


if __name__ == "__main__":
    main()
