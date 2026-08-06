"""Build a small calibration pilot for the V2 rubric, with worked reference solutions.

    python projects/pedagogy_rm/build_pilot.py --n 20 --out data/pilot

WHY A PILOT BEFORE A ROUND. The largest effect in the tutor-rating literature is not which
constructs you choose, it is how the raters are prepared: the same construct names score
kappa 0.13-0.30 from cold crowdworkers and 0.65-0.71 from a handful of annotators given
anchors, worked examples and a calibration round first. Twenty turns rated against untested
anchors tells you whether the anchors work. Two hundred tells you nothing extra, and if the
anchors are wrong it is two hundred wasted.

WHAT THE PILOT IS FOR, concretely - three questions, none of them about model quality:
  - Do the anchors partition? Every level of every dimension should get used by somebody.
  - Are the marginals usable? A dimension that is 85% one value cannot show agreement, and
    kappa will understate it. Better to find that now than to discard a good dimension later.
  - Does anything get flagged? A dimension flagged often is one to rewrite.

WORKED SOLUTIONS, WHICH ARE THE POINT OF THIS SCRIPT. V2's `correct` says the rater checks
against the reference rather than re-deriving it, because that is the difference between the
published kappa of 0.25 and 0.75 on essentially the same question. The pool carries the right
ANSWER but not the steps, so the steps are generated here, once per question, and shown in
the interface. Same for the candidate `step_size`, whose whole definition is "roughly one
line of the reference".

STRATIFIED, NOT RANDOM. Twenty at random from this pool would be mostly mid-length turns from
whichever arm dominates. The point is to exercise the anchors, so the sample deliberately
spans the arms, the length range, and what the student had just demonstrated.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import random
import re

STUCK = re.compile(
    r"\b(i don'?t know|not sure|no idea|i'?m stuck|confused|help|can you (explain|show)|lost)\b", re.I
)
REASON = re.compile(r"\b(because|since|so |therefore|which means|if |then )", re.I)

PROMPT = """Write the worked solution to this question, for a rater who has to check whether a
tutor's statements are correct. Number the steps. Be complete but terse - each step one line.
Do not add commentary, encouragement, or teaching. End with the final answer on its own line.

QUESTION: {question}
{choices}CORRECT ANSWER: {gold}"""


def student_state(text: str) -> int:
    """The same three-way code the rubric asks the rater for, used here only to stratify."""
    t = (text or "").strip()
    if not t or STUCK.search(t):
        return 1
    if not (re.search(r"\d", t) or len(t.split()) >= 12):
        return 1
    return 3 if REASON.search(t) else 2


def pick(units: dict, key: dict, n: int, seed: int) -> list[str]:
    """A spread over arm x length x what the student showed, rather than a random draw."""
    rng = random.Random(seed)
    buckets: dict[tuple, list[str]] = collections.defaultdict(list)
    for uid, u in units.items():
        arm = (key.get(uid) or {}).get("arm", "?")
        words = len(u["tutor_turn"].split())
        size = 0 if words < 20 else 1 if words < 45 else 2
        buckets[(arm, size, student_state(u.get("student_before", "")))].append(uid)
    for v in buckets.values():
        rng.shuffle(v)
    chosen: list[str] = []
    order = sorted(buckets, key=lambda k: -len(buckets[k]))
    while len(chosen) < n and any(buckets[k] for k in order):
        for k in order:
            if buckets[k] and len(chosen) < n:
                chosen.append(buckets[k].pop())
    rng.shuffle(chosen)
    return chosen


async def solutions(units: list[dict], model: str) -> dict[str, str]:
    from projects.pedagogy_rm import gateway  # noqa: PLC0415

    client = gateway.make_client()
    seen: dict[str, dict] = {}
    for u in units:
        seen.setdefault(u["question"].strip(), u)

    async def one(q: str, u: dict) -> tuple[str, str]:
        choices = u.get("choices") or []
        rendered = "OPTIONS: " + "; ".join(map(str, choices)) + "\n" if choices else ""
        reply = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT.format(
                question=q, choices=rendered, gold=u.get("gold", "?"))}],
            temperature=0.0,
        )
        return q, (reply.choices[0].message.content or "").strip()

    done = await asyncio.gather(*(one(q, u) for q, u in seen.items()))
    return dict(done)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="data/eval_c/pool.json")
    parser.add_argument("--key", default="data/eval_c/key.json")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="openai-group/gpt-5.6-terra")
    parser.add_argument("--out", default="data/pilot")
    args = parser.parse_args()

    with open(args.pool) as h:
        units = {u["id"]: u for u in json.load(h)["units"]}
    with open(args.key) as h:
        key = json.load(h)["key"]

    chosen = pick(units, key, args.n, args.seed)
    picked = [units[u] for u in chosen]
    print(f"{len(picked)} turns from {len(units)}, spread over arm x length x student state:")
    spread = collections.Counter(
        ((key.get(u["id"]) or {}).get("arm", "?"), student_state(u.get("student_before", "")))
        for u in picked
    )
    for (arm, st), c in sorted(spread.items()):
        print(f"  {arm:<7} student showed {['nothing','an attempt','reasoning'][st-1]:<11} {c}")

    print(f"\nwriting worked solutions for {len({u['question'].strip() for u in picked})} questions...")
    worked = asyncio.run(solutions(picked, args.model))
    out = [dict(u, reference=worked.get(u["question"].strip(), "")) for u in picked]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "pool.json"), "w") as h:
        json.dump({"note": "V2 calibration pilot; `reference` is a generated worked solution",
                   "units": out}, h, indent=1)
    missing = sum(1 for u in out if not u["reference"])
    print(f"wrote {args.out}/pool.json  ({missing} turns without a reference solution)")
    print("\nLabel it with:")
    print(f"  python projects/pedagogy_rm/label_ui.py --units {args.out}/pool.json \\")
    print(f"      --out {args.out}/sophia.json --port 8771 \\")
    print("      --dimensions student_state,guidance,locates,hands_over,verdict,leak,correct")


if __name__ == "__main__":
    main()
