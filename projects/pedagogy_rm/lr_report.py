"""Compare short runs at different learning rates on reward gained per unit of policy drift.

    python projects/pedagogy_rm/lr_report.py --logs 'logs/pedagogy_lr_*.out' \
        --baseline logs/pedagogy_grpo_d_19723914.out

WHY THIS RATIO AND NOT REWARD. A larger learning rate always produces more reward in a fixed
number of steps, so ranking runs by reward ranks them by learning rate and tells you nothing.
Arm B is the worked example: at 3.4x arm A's effective step size it drifted 3.3x as far in KL,
claimed +0.49 against arm A's +0.36, and was paid +0.37 by human raters against arm A's +0.35.
It spent 3.3 units of drift to buy 0.13 units of claimed gain and none of real gain.

Reward per unit KL is the thing with an interior maximum. Too small a step wastes wall clock;
too large distorts the policy faster than it improves it, and the surplus shows up as the gap
between what the head claims and what a person pays. So the run to pick is the one that buys
the most reward per unit of drift, not the most reward.

WHAT ELSE IS PRINTED, because the ratio alone can be gamed by a run that barely moves.
`sequence_lengths` and `dim_length` are shown so a run that got its reward by satisfying the
length term rather than the pedagogical dimensions is visible as such - and so is the collapse
case, where length falls off a cliff and the other dimensions follow.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import statistics


def series(path: str, key: str) -> list[float]:
    pattern = re.compile(re.escape(key) + r": ([-0-9.e+]+)")
    out = []
    with open(path, errors="ignore") as handle:
        for line in handle:
            out += [float(m) for m in pattern.findall(line)]
    return out


def head_tail(values: list[float], n: int = 5) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return statistics.fmean(values[:n]), statistics.fmean(values[-n:])


def label_of(path: str) -> str:
    with open(path, errors="ignore") as handle:
        for line in handle:
            if "LR=" in line:
                return "lr " + line.split("LR=")[1].split()[0]
            if line.startswith("mode="):
                break
    return os.path.basename(path).replace(".out", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", default="logs/pedagogy_lr_*.out")
    parser.add_argument("--baseline", default="", help="a run at the reference learning rate")
    parser.add_argument("--steps", type=int, default=35, help="compare over this many steps")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.logs))
    if args.baseline and os.path.exists(args.baseline):
        paths.append(args.baseline)
    if not paths:
        raise SystemExit(f"nothing matched {args.logs}")

    rows = []
    for path in paths:
        # `scores` is printed inside a box that wraps, so its value often lands on the next
        # line and a line-wise regex catches three of thirty-eight. `pedagogy/reward` is the
        # same quantity emitted as a plain key-value pair.
        reward = series(path, "pedagogy/reward")[: args.steps]
        kl = series(path, "kl1_avg")[: args.steps]
        length = series(path, "sequence_lengths")[: args.steps]
        band = series(path, "dim_length")[: args.steps]
        if len(reward) < 10 or len(kl) < 10:
            print(f"  {label_of(path):<12} only {len(reward)} steps logged - skipped")
            continue
        r0, r1 = head_tail(reward)
        _, k1 = head_tail(kl)
        l0, l1 = head_tail(length)
        _, b1 = head_tail(band)
        rows.append(
            {
                "label": label_of(path),
                "steps": len(reward),
                "gain": r1 - r0,
                "kl": k1,
                "per_kl": (r1 - r0) / k1 if k1 > 1e-6 else float("inf"),
                "len_from": l0,
                "len_to": l1,
                "band": b1,
            }
        )

    rows.sort(key=lambda r: -r["per_kl"])
    print(f"\n{len(rows)} runs, first {args.steps} steps\n")
    print(f"  {'run':<12} {'reward gain':>12} {'KL':>7} {'gain/KL':>9} {'tokens':>14} {'in band':>8}")
    print("  " + "-" * 70)
    for r in rows:
        print(
            f"  {r['label']:<12} {r['gain']:>+12.3f} {r['kl']:>7.3f} {r['per_kl']:>9.2f} "
            f"{r['len_from']:>6.0f}->{r['len_to']:<7.0f} {r['band']:>8.2f}"
        )

    if rows:
        best = rows[0]
        print(f"\n  most efficient: {best['label']} at {best['per_kl']:.2f} reward per unit KL")
        worst = rows[-1]
        if worst["per_kl"] > 0 and best["per_kl"] / worst["per_kl"] < 1.3:
            print("  BUT the spread is under 1.3x, so these are effectively tied and the")
            print("  smallest learning rate is the safer pick - it drifts least for the same buy.")
        collapsed = [r for r in rows if r["len_to"] < 15]
        if collapsed:
            print(f"  COLLAPSED (under 15 tokens): {', '.join(r['label'] for r in collapsed)}")


if __name__ == "__main__":
    main()
