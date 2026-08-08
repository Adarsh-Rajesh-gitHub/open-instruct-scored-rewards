"""Test whether a mastery graph selects a real student's weakest next domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from projects.student_state_tutor import sphere_mastery_prediction as sphere
from scipy.stats import rankdata


def domain_state_matrices(
    correctness: np.ndarray, domains: np.ndarray, history_items: np.ndarray, target_items: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n_students = correctness.shape[0]
    n_domains = len(sphere.INSTRUMENTS)
    history_state = np.empty((n_students, n_domains), dtype=np.float64)
    heldout_state = np.empty_like(history_state)
    for domain in range(n_domains):
        history = history_items[domains[history_items] == domain]
        target = target_items[domains[target_items] == domain]
        history_state[:, domain] = sphere.posterior_mean(correctness[:, history], axis=1)
        heldout_state[:, domain] = sphere.posterior_mean(correctness[:, target], axis=1)
    return history_state, heldout_state


def row_pick(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return values[np.arange(len(values)), indices]


def rank_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_ranks = np.apply_along_axis(rankdata, 1, left)
    right_ranks = np.apply_along_axis(rankdata, 1, right)
    left_centered = left_ranks - left_ranks.mean(axis=1, keepdims=True)
    right_centered = right_ranks - right_ranks.mean(axis=1, keepdims=True)
    numerator = (left_centered * right_centered).sum(axis=1)
    denominator = np.sqrt((left_centered**2).sum(axis=1) * (right_centered**2).sum(axis=1))
    return numerator / np.maximum(denominator, 1e-12)


def run_seed(correctness: np.ndarray, domains: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    history_items, target_items = sphere.split_items(domains, rng)
    history_state, heldout_state = domain_state_matrices(correctness, domains, history_items, target_items)
    predicted_weak = np.argmin(history_state, axis=1)
    predicted_strong = np.argmax(history_state, axis=1)
    oracle_weak = np.argmin(heldout_state, axis=1)
    random_domain = rng.integers(0, len(sphere.INSTRUMENTS), len(correctness))
    shuffled_state = history_state[:, rng.permutation(len(sphere.INSTRUMENTS))]
    shuffled_weak = np.argmin(shuffled_state, axis=1)

    selected = row_pick(heldout_state, predicted_weak)
    random_selected = row_pick(heldout_state, random_domain)
    shuffled_selected = row_pick(heldout_state, shuffled_weak)
    strongest_selected = row_pick(heldout_state, predicted_strong)
    oracle_selected = row_pick(heldout_state, oracle_weak)
    correlations = rank_correlation(history_state, heldout_state)
    shuffled_correlations = rank_correlation(shuffled_state, heldout_state)

    return {
        "seed": seed,
        "selected_weak_mastery": float(selected.mean()),
        "random_domain_mastery": float(random_selected.mean()),
        "shuffled_state_mastery": float(shuffled_selected.mean()),
        "selected_strong_mastery": float(strongest_selected.mean()),
        "oracle_weak_mastery": float(oracle_selected.mean()),
        "random_minus_selected": float((random_selected - selected).mean()),
        "shuffled_minus_selected": float((shuffled_selected - selected).mean()),
        "strong_minus_weak": float((strongest_selected - selected).mean()),
        "oracle_regret": float((selected - oracle_selected).mean()),
        "weak_domain_top1": float(np.mean(predicted_weak == oracle_weak)),
        "rank_correlation": float(correlations.mean()),
        "shuffled_rank_correlation": float(shuffled_correlations.mean()),
    }


def interval(values: list[float]) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def summarize(results: list[dict], students: int, items: int) -> dict:
    metrics = [key for key in results[0] if key != "seed"]
    return {
        "students": students,
        "items": items,
        "domains": list(sphere.INSTRUMENTS),
        "seeds": len(results),
        "metrics": {
            metric: {
                "mean": float(np.mean([row[metric] for row in results])),
                "95_interval_across_item_splits": interval([row[metric] for row in results]),
            }
            for metric in metrics
        },
        "seed_results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()

    correctness, domains, item_names = sphere.load_correctness(args.data_dir)
    results = [run_seed(correctness, domains, seed) for seed in range(args.seeds)]
    summary = summarize(results, students=len(correctness), items=len(item_names))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
