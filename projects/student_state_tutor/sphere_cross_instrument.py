"""Test whether human mastery state transfers across related instruments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from projects.student_state_tutor import sphere_mastery_prediction as sphere
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RELATED_DIRECTIONS = (
    ("FCI", "FMCE"),
    ("FMCE", "FCI"),
    ("FCI", "RRMCS"),
    ("RRMCS", "FCI"),
    ("TCE", "STPFASL"),
    ("STPFASL", "TCE"),
)
MODELS = ("global", "related", "unrelated", "shuffled")


def domain_items(domains: np.ndarray, name: str) -> np.ndarray:
    return np.flatnonzero(domains == sphere.INSTRUMENTS.index(name))


def build_features(
    student_indices: np.ndarray,
    target_items: np.ndarray,
    item_difficulty: np.ndarray,
    global_state: np.ndarray,
    related_state: np.ndarray,
    unrelated_state: np.ndarray,
    shuffled_state: np.ndarray,
) -> dict[str, np.ndarray]:
    n_targets = len(target_items)
    item_feature = np.tile(item_difficulty[None, :], (len(student_indices), 1)).reshape(-1)
    global_feature = np.repeat(global_state[student_indices], n_targets)
    base = np.column_stack([item_feature, global_feature])

    def add_state(state):
        return np.column_stack([base, np.repeat(state[student_indices], n_targets)])

    return {
        "global": base,
        "related": add_state(related_state),
        "unrelated": add_state(unrelated_state),
        "shuffled": add_state(shuffled_state),
    }


def run_direction(
    correctness: np.ndarray, domains: np.ndarray, donor_name: str, target_name: str, seed: int, folds: int
) -> dict:
    rng = np.random.default_rng(seed)
    donor_items = domain_items(domains, donor_name)
    target_items = domain_items(domains, target_name)
    excluded = {sphere.INSTRUMENTS.index(donor_name), sphere.INSTRUMENTS.index(target_name)}
    unrelated_names = [name for index, name in enumerate(sphere.INSTRUMENTS) if index not in excluded]
    unrelated_name = unrelated_names[seed % len(unrelated_names)]
    unrelated_items = domain_items(domains, unrelated_name)
    global_items = np.flatnonzero(~np.isin(domains, list(excluded)))

    global_state = sphere.posterior_mean(correctness[:, global_items], axis=1)
    related_state = sphere.posterior_mean(correctness[:, donor_items], axis=1)
    unrelated_state = sphere.posterior_mean(correctness[:, unrelated_items], axis=1)
    shuffled_state = related_state[rng.permutation(len(related_state))]
    target_labels = correctness[:, target_items].astype(np.int64)
    flat_labels = target_labels.reshape(-1)
    predictions = {model: np.empty(flat_labels.size, dtype=np.float64) for model in MODELS}

    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_students, test_students in splitter.split(correctness):
        item_difficulty = sphere.posterior_mean(correctness[train_students][:, target_items], axis=0)
        train_features = build_features(
            train_students, target_items, item_difficulty, global_state, related_state, unrelated_state, shuffled_state
        )
        test_features = build_features(
            test_students, target_items, item_difficulty, global_state, related_state, unrelated_state, shuffled_state
        )
        train_labels = target_labels[train_students].reshape(-1)
        test_indices = (test_students[:, None] * len(target_items) + np.arange(len(target_items))[None, :]).reshape(-1)
        for model_name in MODELS:
            classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
            classifier.fit(train_features[model_name], train_labels)
            predictions[model_name][test_indices] = classifier.predict_proba(test_features[model_name])[:, 1]

    return {
        "donor": donor_name,
        "target": target_name,
        "unrelated": unrelated_name,
        "seed": seed,
        "metrics": {
            model: sphere.evaluate_predictions(flat_labels, probabilities)
            for model, probabilities in predictions.items()
        },
    }


def comparison(results: list[dict], left: str, right: str) -> dict:
    output = {}
    for metric in ("auc", "log_loss", "brier", "accuracy"):
        differences = np.asarray(
            [result["metrics"][left][metric] - result["metrics"][right][metric] for result in results]
        )
        output[metric] = {
            "mean": float(differences.mean()),
            "95_interval_across_splits": [
                float(np.percentile(differences, 2.5)),
                float(np.percentile(differences, 97.5)),
            ],
        }
    return output


def summarize(results: list[dict]) -> dict:
    by_direction = {}
    for donor, target in RELATED_DIRECTIONS:
        selected = [row for row in results if row["donor"] == donor and row["target"] == target]
        by_direction[f"{donor}_to_{target}"] = {
            "related_minus_global": comparison(selected, "related", "global"),
            "related_minus_unrelated": comparison(selected, "related", "unrelated"),
            "related_minus_shuffled": comparison(selected, "related", "shuffled"),
        }
    return {
        "directions": by_direction,
        "pooled": {
            "related_minus_global": comparison(results, "related", "global"),
            "related_minus_unrelated": comparison(results, "related", "unrelated"),
            "related_minus_shuffled": comparison(results, "related", "shuffled"),
        },
        "runs": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    correctness, domains, _ = sphere.load_correctness(args.data_dir)
    results = [
        run_direction(correctness, domains, donor, target, seed, args.folds)
        for donor, target in RELATED_DIRECTIONS
        for seed in range(args.seeds)
    ]
    summary = summarize(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
