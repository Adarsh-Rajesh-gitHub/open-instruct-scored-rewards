"""Validate a persistent mastery state on real human physics responses.

For each random item split, half of every physics inventory is observed as
student history and the other half is held out. We compare prediction of held-
out correctness from:

1. item difficulty + global student ability;
2. the same features + per-inventory mastery state; and
3. the same features + mastery over randomly permuted item groups.

The per-inventory state is a Beta-Bernoulli posterior updated only from observed
history responses. Improvement over both controls establishes predictive value
for a coarse, persistent student-state graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyreadr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

INSTRUMENTS = ("FCI", "FMCE", "RRMCS", "FMCI", "MWCS", "TCE", "STPFASL")
MODELS = ("global", "mastery", "shuffled")


def read_rda_frame(path: Path):
    objects = pyreadr.read_r(str(path))
    if len(objects) != 1:
        raise ValueError(f"expected one object in {path}, found {len(objects)}")
    return next(iter(objects.values()))


def load_correctness(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    matrices = []
    domains = []
    item_names = []
    n_students = None
    for domain_index, instrument in enumerate(INSTRUMENTS):
        responses = read_rda_frame(data_dir / f"{instrument}.rda")
        keys = read_rda_frame(data_dir / f"{instrument}key.rda")
        if n_students is None:
            n_students = len(responses)
        elif len(responses) != n_students:
            raise ValueError("SPHERE instruments do not share a student axis")
        key_values = keys.iloc[0].astype(str).to_numpy()
        response_values = responses.astype(str).to_numpy()
        if response_values.shape[1] != len(key_values):
            raise ValueError(f"key width mismatch for {instrument}")
        matrices.append((response_values == key_values[None, :]).astype(np.float32))
        domains.extend([domain_index] * response_values.shape[1])
        item_names.extend(responses.columns.astype(str).tolist())
    return (np.concatenate(matrices, axis=1), np.asarray(domains, dtype=np.int64), item_names)


def posterior_mean(correct: np.ndarray, axis: int) -> np.ndarray:
    """Uniform Beta(1,1) posterior mean for binary observations."""
    count = correct.shape[axis]
    return (correct.sum(axis=axis) + 1.0) / (count + 2.0)


def split_items(domains: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    history = []
    target = []
    for domain in np.unique(domains):
        indices = np.flatnonzero(domains == domain)
        shuffled = rng.permutation(indices)
        cut = max(2, len(indices) // 2)
        history.extend(shuffled[:cut])
        target.extend(shuffled[cut:])
    return np.asarray(sorted(history)), np.asarray(sorted(target))


def student_states(
    correctness: np.ndarray,
    domains: np.ndarray,
    history_items: np.ndarray,
    target_items: np.ndarray,
    pseudo_domains: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history = correctness[:, history_items]
    global_state = posterior_mean(history, axis=1)
    mastery = np.empty((len(correctness), len(target_items)), dtype=np.float32)
    shuffled = np.empty_like(mastery)
    for column, item in enumerate(target_items):
        same_domain = history_items[domains[history_items] == domains[item]]
        same_pseudo_domain = history_items[pseudo_domains[history_items] == pseudo_domains[item]]
        mastery[:, column] = posterior_mean(correctness[:, same_domain], axis=1)
        if len(same_pseudo_domain):
            shuffled[:, column] = posterior_mean(correctness[:, same_pseudo_domain], axis=1)
        else:
            shuffled[:, column] = global_state
    return global_state, mastery, shuffled


def row_features(
    student_indices: np.ndarray,
    target_items: np.ndarray,
    item_difficulty: np.ndarray,
    global_state: np.ndarray,
    mastery: np.ndarray,
    shuffled: np.ndarray,
) -> dict[str, np.ndarray]:
    n_students = len(student_indices)
    n_targets = len(target_items)
    item_feature = np.tile(item_difficulty[None, :], (n_students, 1)).reshape(-1)
    global_feature = np.repeat(global_state[student_indices], n_targets)
    base = np.column_stack([item_feature, global_feature])
    return {
        "global": base,
        "mastery": np.column_stack([base, mastery[student_indices].reshape(-1)]),
        "shuffled": np.column_stack([base, shuffled[student_indices].reshape(-1)]),
    }


def evaluate_predictions(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
    }


def run_seed(correctness: np.ndarray, domains: np.ndarray, seed: int, folds: int) -> dict:
    rng = np.random.default_rng(seed)
    history_items, target_items = split_items(domains, rng)
    pseudo_domains = rng.permutation(domains)
    global_state, mastery, shuffled = student_states(correctness, domains, history_items, target_items, pseudo_domains)
    labels_matrix = correctness[:, target_items].astype(np.int64)
    predictions = {model: np.empty(labels_matrix.size, dtype=np.float64) for model in MODELS}
    labels = labels_matrix.reshape(-1)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)

    for train_students, test_students in splitter.split(correctness):
        item_difficulty = posterior_mean(correctness[train_students][:, target_items], axis=0)
        train_features = row_features(train_students, target_items, item_difficulty, global_state, mastery, shuffled)
        test_features = row_features(test_students, target_items, item_difficulty, global_state, mastery, shuffled)
        train_labels = labels_matrix[train_students].reshape(-1)
        test_flat_indices = (
            test_students[:, None] * len(target_items) + np.arange(len(target_items))[None, :]
        ).reshape(-1)
        for model_name in MODELS:
            classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
            classifier.fit(train_features[model_name], train_labels)
            predictions[model_name][test_flat_indices] = classifier.predict_proba(test_features[model_name])[:, 1]

    return {
        "seed": seed,
        "history_items": len(history_items),
        "target_items": len(target_items),
        "metrics": {
            model_name: evaluate_predictions(labels, probabilities)
            for model_name, probabilities in predictions.items()
        },
    }


def paired_summary(seed_results: list[dict], left: str, right: str) -> dict:
    output = {}
    for metric in ("auc", "log_loss", "brier", "accuracy"):
        differences = np.asarray(
            [result["metrics"][left][metric] - result["metrics"][right][metric] for result in seed_results],
            dtype=np.float64,
        )
        output[metric] = {
            "mean": float(differences.mean()),
            "95_interval_across_item_splits": [
                float(np.percentile(differences, 2.5)),
                float(np.percentile(differences, 97.5)),
            ],
        }
    return output


def summarize(correctness: np.ndarray, domains: np.ndarray, item_names: list[str], seed_results: list[dict]) -> dict:
    mean_metrics = {}
    for model_name in MODELS:
        mean_metrics[model_name] = {
            metric: float(np.mean([result["metrics"][model_name][metric] for result in seed_results]))
            for metric in ("auc", "log_loss", "brier", "accuracy")
        }
    return {
        "students": int(correctness.shape[0]),
        "items": len(item_names),
        "domains": {instrument: int(np.sum(domains == index)) for index, instrument in enumerate(INSTRUMENTS)},
        "seeds": len(seed_results),
        "mean_metrics": mean_metrics,
        "mastery_minus_global": paired_summary(seed_results, "mastery", "global"),
        "mastery_minus_shuffled": paired_summary(seed_results, "mastery", "shuffled"),
        "seed_results": seed_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    correctness, domains, item_names = load_correctness(args.data_dir)
    seed_results = [run_seed(correctness, domains, seed, args.folds) for seed in range(args.seeds)]
    result = summarize(correctness, domains, item_names, seed_results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
