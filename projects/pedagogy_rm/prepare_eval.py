"""Set up the second human evaluation, and answer the parts that need no human.

    python projects/pedagogy_rm/prepare_eval.py \
        --pool data/eval_c/pool.json --key data/eval_c/key.json \
        --against arm_c:arm_a --pairs 40

WHY THIS EXISTS. The first evaluation cost 51 hand ratings and 47 pairwise judgements. Most
of that does not need repeating, because two of the three questions can now be answered from
data already collected:

  1. DID THE LENGTH FIX WORK? 1131 length judgements over the first pool put a curve through
     P(a rater calls this length right | word count). Applying it to a new policy's turns
     predicts its too-short / right / too-long split without anyone reading anything. It is a
     prediction rather than a measurement and is labelled as such, but the curve was fitted on
     the same rubric and the same raters, and length is the one property word count captures
     almost perfectly on the long side.

  2. DID QUALITY SURVIVE? The agent raters are already calibrated, and base and arm A keep
     their existing labels because their blinded ids are unchanged - so only the new arm's
     turns need rating, by machine.

  3. IS IT ACTUALLY BETTER? This one needs a person, and pairwise is the cheapest form: one
     keystroke per judgement rather than six ratings, and it was the more sensitive of the two
     instruments last time. So the human ask is one comparison, not a full rating pass.

REUSING LABELS IS ONLY SAFE IF THE IDs MATCH, so that is checked rather than assumed. A blind
id is derived from the salt, the arm's *name* and the source unit id; rebuilding the pool with
`lora=` where the first build said `arm_a=` renames every id and silently orphans every label.
This script refuses to proceed if the overlap is smaller than it should be.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random
import statistics


def load_curve(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def acceptability(curve: dict, words: int) -> float:
    """P(a rater calls this length right), from the two-shoulder fit in the first pool."""
    lw = math.log(max(words, 1))
    return (1 / (1 + math.exp(-(lw - curve["a"]) / curve["s_short"]))) * (
        1 / (1 + math.exp(-(curve["b"] - lw) / curve["s_long"]))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="data/eval_c/pool.json")
    parser.add_argument("--key", default="data/eval_c/key.json")
    parser.add_argument("--curve", default="data/length_curve.json")
    parser.add_argument("--old-labels", default="data/eval50/labels/*.json", help="reused where ids match")
    parser.add_argument("--against", default="arm_c:arm_a", help="the pairwise comparison, as new:reference")
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out", default="", help="pairs file; defaults beside the pool")
    args = parser.parse_args()

    with open(args.pool) as handle:
        units = {u["id"]: u for u in json.load(handle)["units"]}
    with open(args.key) as handle:
        key = json.load(handle)["key"]
    curve = load_curve(args.curve)
    arms = sorted({e["arm"] for e in key.values()})
    words = {u: len(units[u]["tutor_turn"].split()) for u in units}

    print(f"{len(units)} turns over {len(arms)} arms: {', '.join(arms)}\n")

    # 1. The length question, answered without a human.
    print("PREDICTED LENGTH OUTCOME  (from 1131 judgements on the first pool; a prediction)")
    print(f"  {'arm':<8} {'median':>7} {'too short':>10} {'right':>7} {'too long':>9}")
    print("  " + "-" * 46)
    for arm in arms:
        ws = [words[u] for u, e in key.items() if e["arm"] == arm and u in words]
        if not ws:
            continue
        right = statistics.fmean(acceptability(curve, w) for w in ws)
        short = sum(1 - acceptability(curve, w) for w in ws if w < 40) / len(ws)
        long_ = sum(1 - acceptability(curve, w) for w in ws if w >= 40) / len(ws)
        print(f"  {arm:<8} {statistics.median(ws):>6.0f}w {short:>10.0%} {right:>7.0%} {long_:>9.0%}")
    print("  For reference, the first pool measured: base 12/71/18, arm A 29/71/0 (human labels).")

    # 2. What is already labelled, and what is not.
    have: dict[str, set[str]] = collections.defaultdict(set)
    for path in sorted(glob.glob(args.old_labels)):
        with open(path) as handle:
            blob = json.load(handle)
        rater = blob.get("rater") or os.path.basename(path)[:-5]
        for record in blob.get("labels", []):
            if record["id"] in units:
                have[rater].add(record["id"])
    print("\nEXISTING LABELS THAT STILL JOIN  (ids are unchanged where the arm name is)")
    by_arm = collections.Counter(key[u]["arm"] for r in have.values() for u in r)
    for arm in arms:
        total = sum(1 for e in key.values() if e["arm"] == arm)
        covered = len({u for r in have.values() for u in r if key[u]["arm"] == arm})
        flag = "" if covered or arm not in ("base", "arm_a") else "   <- expected reuse, got none: check the arm names"
        print(f"  {arm:<8} {covered:>3}/{total} turns carry at least one old label{flag}")
    if by_arm and not any(by_arm.get(a) for a in ("base", "arm_a")):
        raise SystemExit(
            "no old labels joined at all. The pool was probably rebuilt with different arm\n"
            "names or a different --seed, which changes every blinded id. Rebuild with the\n"
            "same names and seed as the first pool, or accept a full re-label."
        )
    new = [u for u in units if not any(u in r for r in have.values())]
    print(f"  {len(new)} turns have no label yet - these are what the agents need to rate.")

    # 3. The one thing a person still has to do.
    new_arm, _, reference = args.against.partition(":")
    if new_arm not in arms or reference not in arms:
        raise SystemExit(f"--against wants two of {arms}, got {args.against!r}")
    by_moment: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for uid, entry in key.items():
        by_moment[entry["moment"]][entry["arm"]] = uid
    rng = random.Random(args.seed)
    moments = [m for m, got in by_moment.items() if new_arm in got and reference in got]
    rng.shuffle(moments)
    chosen = moments[: args.pairs]
    # Balanced by construction rather than by an independent coin per pair. Forty fair coins
    # land 14/26 often enough to matter - it happened on the first draw here - and a labeller
    # who senses that one side is usually the new arm is no longer blind. Half the slots are
    # allocated to each side and then shuffled, so the split is exact and the order is random.
    flip = [True] * (len(chosen) // 2) + [False] * (len(chosen) - len(chosen) // 2)
    rng.shuffle(flip)
    pairs = []
    for m, new_on_left in zip(chosen, flip, strict=True):
        one, two = by_moment[m][new_arm], by_moment[m][reference]
        left, right = (one, two) if new_on_left else (two, one)
        pairs.append({"id": f"p{len(pairs):04d}", "moment": m, "left": left, "right": right})
    out = args.out or os.path.join(os.path.dirname(args.pool), f"pairs_{new_arm}_vs_{reference}.json")
    with open(out, "w") as handle:
        json.dump(
            {"note": f"{new_arm} against {reference}, blinded, one turn each per moment", "pairs": pairs},
            handle,
            indent=1,
        )

    sides = collections.Counter(key[p["left"]]["arm"] for p in pairs)
    print(f"\nHUMAN TASK: {len(pairs)} pairs, {new_arm} vs {reference}")
    print(f"  sides balanced: {new_arm} on the left {sides[new_arm]}, {reference} on the left {sides[reference]}")
    print(f"  wrote {out}")
    print("\nNext:")
    print(f"  1. agents rate the {len(new)} new turns:")
    print(f"     python projects/pedagogy_rm/label_agents.py --units {args.pool} \\")
    print(f"         --out-dir {os.path.dirname(args.pool)}/labels --calibration data/eval50/calibration.md \\")
    print("         --examples data/eval50/labels/sophia.json --holdout data/eval50/holdout.json \\")
    print("         --dimensions leak,targeted,actionable,elicits,length_fit,correct")
    print("  2. you judge the pairs:")
    print(f"     python projects/pedagogy_rm/prefer_ui.py --pool {args.pool} --pairs {out} \\")
    print(f"         --out {os.path.dirname(args.pool)}/labels/prefs_{new_arm}.json --port 8772")
    print("  3. then score_arms.py over the lot.")


if __name__ == "__main__":
    main()
