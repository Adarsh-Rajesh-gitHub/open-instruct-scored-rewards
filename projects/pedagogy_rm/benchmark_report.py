"""Compare the benchmark runs, and say whether the differences are real.

    python projects/pedagogy_rm/benchmark_report.py --dir data/benchmarks

WHY A PAIRED TEST RATHER THAN TWO ACCURACIES. Every arm answers the same 400 items, so the
interesting quantity is which items changed hands, not the gap between two percentages. Treating
the arms as independent samples throws that pairing away and roughly doubles the interval: on 400
items at ~40% accuracy an unpaired comparison cannot resolve anything smaller than about 7 points,
which is wide enough to hide any plausible amount of forgetting. McNemar's test looks only at the
items where the two models disagree, and there are usually few enough of those to be decisive.

The null result this is built to support is the one that needs the most care, because "we saw no
change" and "we could not have seen a change" print identically. So the interval is reported
alongside every difference: what matters for the claim is not that p > 0.05 but that the interval
excludes a drop large enough to care about.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os


def mcnemar(a: list[int], b: list[int]) -> tuple[int, int, float]:
    """Exact two-sided binomial test on the items the two models disagree about."""
    gained = sum(1 for x, y in zip(a, b, strict=True) if y > x)
    lost = sum(1 for x, y in zip(a, b, strict=True) if y < x)
    n = gained + lost
    if n == 0:
        return 0, 0, 1.0
    # Exact rather than chi-square: the discordant counts here are often under 25, where the
    # normal approximation is not trustworthy.
    smaller = min(gained, lost)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / 2**n
    return gained, lost, min(1.0, 2 * tail)


def interval(a: list[int], b: list[int]) -> tuple[float, float]:
    """95% interval on the paired difference in accuracy, from the per-item differences."""
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    half = 1.96 * math.sqrt(var / n)
    return mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="data/benchmarks")
    parser.add_argument("--base", default="base")
    args = parser.parse_args()

    runs = {}
    for path in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        with open(path) as handle:
            blob = json.load(handle)
        runs[blob["tag"]] = blob["results"]
    if args.base not in runs:
        raise SystemExit(f"no {args.base}.json in {args.dir}; found {sorted(runs)}")

    tasks = list(runs[args.base])
    others = [t for t in runs if t != args.base]

    print(f"{'task':<16}{'base':>8}" + "".join(f"{t:>10}" for t in others))
    print("-" * (24 + 10 * len(others)))
    for task in tasks:
        row = f"{task:<16}{runs[args.base][task]['accuracy']:>7.1%} "
        for tag in others:
            row += f"{runs[tag][task]['accuracy']:>9.1%} " if task in runs[tag] else f"{'-':>10}"
        print(row)

    print("\npaired against base (negative means the trained model is worse)")
    for tag in others:
        print(f"\n  {tag}")
        for task in tasks:
            if task not in runs[tag]:
                continue
            a, b = runs[args.base][task].get("per_item"), runs[tag][task].get("per_item")
            if not a or not b or len(a) != len(b):
                print(f"    {task:<14} no per-item record; cannot pair")
                continue
            gained, lost, p = mcnemar(a, b)
            lo, hi = interval(a, b)
            delta = sum(b) / len(b) - sum(a) / len(a)
            verdict = "changed" if p < 0.05 else "no detectable change"
            print(f"    {task:<14} {delta:+.1%}  [{lo:+.1%}, {hi:+.1%}]  "
                  f"won {gained} / lost {lost}  p={p:.3f}  {verdict}")

    # The headline is the aggregate: three tasks each underpowered on their own are jointly
    # informative, and forgetting from a narrow objective would not politely confine itself to one.
    print("\npooled over all tasks")
    for tag in others:
        pa, pb = [], []
        for task in tasks:
            a, b = runs[args.base][task].get("per_item"), runs[tag].get(task, {}).get("per_item")
            if a and b and len(a) == len(b):
                pa += a
                pb += b
        if not pa:
            continue
        gained, lost, p = mcnemar(pa, pb)
        lo, hi = interval(pa, pb)
        delta = sum(pb) / len(pb) - sum(pa) / len(pa)
        print(f"  {tag:<8} n={len(pa)}  {delta:+.1%}  [{lo:+.1%}, {hi:+.1%}]  "
              f"won {gained} / lost {lost}  p={p:.3f}")


if __name__ == "__main__":
    main()
