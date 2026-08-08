"""Measure how quickly a human mastery graph becomes action-usable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from projects.student_state_tutor import sphere_curriculum_value as curriculum
from projects.student_state_tutor import sphere_mastery_prediction as sphere


def split_with_k_history(domains: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    history = []
    target = []
    for domain in np.unique(domains):
        indices = rng.permutation(np.flatnonzero(domains == domain))
        if k >= len(indices):
            raise ValueError(f"k={k} leaves no target items in domain {domain}")
        history.extend(indices[:k])
        target.extend(indices[k:])
    return np.asarray(sorted(history)), np.asarray(sorted(target))


def run(correctness: np.ndarray, domains: np.ndarray, k: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    history, target = split_with_k_history(domains, k, rng)
    history_state, heldout_state = curriculum.domain_state_matrices(correctness, domains, history, target)
    predicted_weak = np.argmin(history_state, axis=1)
    predicted_strong = np.argmax(history_state, axis=1)
    oracle_weak = np.argmin(heldout_state, axis=1)
    random_domain = rng.integers(0, len(sphere.INSTRUMENTS), len(correctness))
    selected = curriculum.row_pick(heldout_state, predicted_weak)
    random_selected = curriculum.row_pick(heldout_state, random_domain)
    strong_selected = curriculum.row_pick(heldout_state, predicted_strong)
    oracle_selected = curriculum.row_pick(heldout_state, oracle_weak)
    correlations = curriculum.rank_correlation(history_state, heldout_state)
    return {
        "k": k,
        "seed": seed,
        "weak_domain_top1": float(np.mean(predicted_weak == oracle_weak)),
        "rank_correlation": float(correlations.mean()),
        "random_minus_selected": float((random_selected - selected).mean()),
        "strong_minus_weak": float((strong_selected - selected).mean()),
        "oracle_regret": float((selected - oracle_selected).mean()),
    }


def summarize(results: list[dict], ks: list[int]) -> dict:
    metrics = ("weak_domain_top1", "rank_correlation", "random_minus_selected", "strong_minus_weak", "oracle_regret")
    by_k = {}
    for k in ks:
        selected = [row for row in results if row["k"] == k]
        by_k[str(k)] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in selected])),
                "95_interval_across_item_splits": curriculum.interval([row[metric] for row in selected]),
            }
            for metric in metrics
        }
    return {"history_items_per_domain": by_k, "runs": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history-counts", type=int, nargs="+", default=[1, 2, 4, 8, 12])
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()

    correctness, domains, _ = sphere.load_correctness(args.data_dir)
    results = [run(correctness, domains, k, seed) for k in args.history_counts for seed in range(args.seeds)]
    summary = summarize(results, args.history_counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
