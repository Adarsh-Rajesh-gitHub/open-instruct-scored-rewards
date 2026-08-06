"""Run each rubric dimension as its own agent pass, then resolve disagreements against the human.

    # one pass per dimension, six models each
    python projects/pedagogy_rm/consensus.py rate \
        --units data/pilot/pool.json --out-dir data/pilot/labels \
        --examples data/pilot/sophia.json --holdout data/pilot/holdout.json

    # apply the resolution rule and write the final labels
    python projects/pedagogy_rm/consensus.py resolve \
        --units data/pilot/pool.json --labels 'data/pilot/labels/agent_*.json' \
        --human data/pilot/sophia.json --out data/pilot/resolved.json

ONE DIMENSION PER CALL, WHICH IS THE POINT OF THIS FILE. label_agents.py asks for all six at
once and takes a single JSON object back. That is cheaper, and it is also six judgements sharing
one context: the model has to hold six anchor sets in mind, its answer on `leak` sits in the
context when it answers `correct`, and a strong opinion on one dimension colours the rest. Asking
once per dimension costs six times the calls and removes that coupling entirely. Since the
dimensions are meant to be independent - and ours were not, `actionable` and `elicits` correlated
at 0.95 - anything that stops the rater coupling them is worth paying for.

Each pass also carries only its own dimension's anchors and only few-shot examples for that
dimension, so the prompt is shorter and more specific than the combined one.

THE RESOLUTION RULE, as specified. In order:

  1. If all raters agree on a value, take it. Unanimity across six models from different
     families is the one case where overruling the human is defensible - it is much more likely
     that a single rating slipped than that six independent models coincided on the same error.
  2. Otherwise, if at least one rater agrees with the human, keep the human. One corroborating
     rater is enough to establish that the human's reading is available to a competent reader,
     which is all that is being asked.
  3. Otherwise take the majority of the raters, since the human's value has no support at all.

Rule 3 is the one to watch. It fires exactly when the human sees something no model sees, and
that is both what a wrong label looks like and what an insight looks like. Every case is
recorded in `overruled` so they can be read rather than assumed.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import glob
import json
import os
import statistics

from projects.pedagogy_rm.rubric import BY_KEY

DEFAULT_RATERS = {
    "gpt": "openai-group/gpt-5.6-terra",
    "claude": "anthropic-group/claude-sonnet-4.5",
    "gemini": "google-group/gemini-3-pro",
    "grok": "xai-group/grok-4.1",
    "deepseek": "deepseek-group/deepseek-v3.2",
    "qwen": "qwen-group/qwen3-235b",
}

SYSTEM = """You are rating ONE property of ONE tutor turn. Reply with a single integer and
nothing else - no words, no punctuation, no explanation.

{key}: {question}

{anchors}

You are reproducing one particular person's judgements, not forming your own. Where the worked
examples below disagree with your instinct, the examples win."""

TASK = """QUESTION: {question}
{reference}STUDENT SAID: {student}
TUTOR TURN: {tutor}"""


def prompt_for(unit: dict, key: str, shots: list[tuple[dict, int]]) -> list[dict]:
    dim = BY_KEY[key]
    anchors = "\n".join(f"{s} = {t}" for s, t in sorted(dim.anchors.items()))
    messages = [{"role": "system", "content": SYSTEM.format(
        key=key, question=dim.question, anchors=anchors)}]
    for shot, score in shots:
        messages.append({"role": "user", "content": render(shot)})
        messages.append({"role": "assistant", "content": str(score)})
    messages.append({"role": "user", "content": render(unit)})
    return messages


def render(unit: dict) -> str:
    flat = lambda s: " ".join((s or "").split())  # noqa: E731
    ref = f"WORKED SOLUTION: {flat(unit['reference'])}\n" if unit.get("reference") else ""
    return TASK.format(question=flat(unit["question"]), reference=ref,
                       student=flat(unit.get("student_before")), tutor=flat(unit["tutor_turn"]))


def parse(text: str, key: str) -> int | None:
    dim = BY_KEY[key]
    for token in (text or "").split():
        digits = "".join(c for c in token if c.isdigit())
        if digits and dim.lo <= int(digits) <= dim.hi:
            return int(digits)
    return None


async def rate(args) -> None:
    from projects.pedagogy_rm import gateway  # noqa: PLC0415

    with open(args.units) as h:
        units = [u for u in json.load(h)["units"]]
    by_id = {u["id"]: u for u in units}
    keys = [k for k in args.dimensions.split(",") if k]

    held: set[str] = set()
    if args.holdout and os.path.exists(args.holdout):
        with open(args.holdout) as h:
            held = set(json.load(h)["ids"])
    human: dict[str, dict] = {}
    if args.examples:
        with open(args.examples) as h:
            human = {r["id"]: r for r in json.load(h)["labels"]}

    raters = DEFAULT_RATERS if not args.raters else {k: DEFAULT_RATERS[k] for k in args.raters.split(",")}
    client = gateway.make_client()
    sem = asyncio.Semaphore(args.concurrency)
    os.makedirs(args.out_dir, exist_ok=True)

    async def one(model: str, unit: dict, key: str, shots) -> tuple[str, str, int | None]:
        async with sem:
            for _ in range(args.retries + 1):
                try:
                    reply = await client.chat.completions.create(
                        model=model, messages=prompt_for(unit, key, shots), temperature=0.0
                    )
                    got = parse(reply.choices[0].message.content or "", key)
                    if got is not None:
                        return unit["id"], key, got
                except Exception:  # noqa: BLE001 - a failed call is a missing rating, not a crash
                    await asyncio.sleep(1.0)
            return unit["id"], key, None

    print(f"{len(units)} turns x {len(keys)} dimensions x {len(raters)} raters = "
          f"{len(units) * len(keys) * len(raters)} single-question calls")
    for name, model in raters.items():
        out: dict[str, dict] = collections.defaultdict(dict)
        jobs = []
        for key in keys:
            # Few-shot for THIS dimension only, drawn from outside the holdout.
            shots = [(by_id[i], r[key]) for i, r in human.items()
                     if i not in held and i in by_id and isinstance(r.get(key), int)][: args.max_shots]
            jobs += [one(model, u, key, shots) for u in units]
        done = await asyncio.gather(*jobs)
        missing = 0
        for uid, key, value in done:
            if value is None:
                missing += 1
            else:
                out[uid][key] = value
        path = os.path.join(args.out_dir, f"agent_{name}.json")
        with open(path, "w") as h:
            json.dump({"schema": "pedagogy-rm/labels-v1", "rater": name, "model": model,
                       "per_dimension": True,
                       "shots": {k: sorted({i for i in human if i not in held}) for k in keys},
                       "labels": [{"id": u, **v} for u, v in out.items()]}, h, indent=1)
        print(f"  {name:<9} {len(out):>4} turns, {missing} calls gave no usable answer -> {path}")


def resolve(args) -> None:
    with open(args.units) as h:
        units = {u["id"]: u for u in json.load(h)["units"]}
    with open(args.human) as h:
        human = {r["id"]: r for r in json.load(h)["labels"]}
    raters: dict[str, dict[str, dict]] = {}
    for path in sorted(glob.glob(args.labels)):
        with open(path) as h:
            blob = json.load(h)
        raters[blob.get("rater") or path] = {r["id"]: r for r in blob.get("labels", [])}
    keys = [k for k in args.dimensions.split(",") if k]

    final, audit = [], []
    tally = collections.Counter()
    for uid in units:
        row = {"id": uid}
        for key in keys:
            votes = [r[uid][key] for r in raters.values()
                     if isinstance(r.get(uid, {}).get(key), int)]
            mine = human.get(uid, {}).get(key)
            if not votes:
                if isinstance(mine, int):
                    row[key] = mine
                    tally["human only"] += 1
                continue
            if len(set(votes)) == 1 and (mine is None or votes[0] != mine):
                row[key] = votes[0]
                tally["unanimous raters override" if mine is not None else "raters only"] += 1
                if mine is not None:
                    audit.append({"id": uid, "dimension": key, "human": mine,
                                  "raters": votes, "took": votes[0], "why": "unanimous"})
            elif isinstance(mine, int) and mine in votes:
                row[key] = mine
                tally["human, corroborated"] += 1
            elif isinstance(mine, int):
                top = collections.Counter(votes).most_common(1)[0][0]
                row[key] = top
                tally["human overruled by majority"] += 1
                audit.append({"id": uid, "dimension": key, "human": mine,
                              "raters": votes, "took": top, "why": "no rater agreed"})
            else:
                row[key] = round(statistics.fmean(votes))
                tally["raters only"] += 1
        final.append(row)

    with open(args.out, "w") as h:
        json.dump({"schema": "pedagogy-rm/labels-v1", "rater": "resolved",
                   "rule": "unanimous raters win; else human if any rater agrees; else majority",
                   "overruled": audit, "labels": final}, h, indent=1)

    print(f"{len(final)} turns resolved over {len(keys)} dimensions, {len(raters)} raters\n")
    for reason, n in tally.most_common():
        print(f"  {n:>4}  {reason}")
    print(f"\nwrote {args.out}")
    if audit:
        print(f"\n{len(audit)} judgements where the human did not prevail. Read these - they are")
        print("either mislabels or the most interesting turns in the set:")
        for a in audit[:6]:
            print(f"  {a['dimension']:<12} you {a['human']}, raters {a['raters']} -> {a['took']}  ({a['why']})")
        if len(audit) > 6:
            print(f"  ... and {len(audit) - 6} more in {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rate", help="one agent pass per dimension")
    r.add_argument("--units", required=True)
    r.add_argument("--out-dir", default="data/pilot/labels")
    r.add_argument("--examples", default="", help="human labels, used as per-dimension few-shot")
    r.add_argument("--holdout", default="", help="ids withheld from every few-shot prompt")
    r.add_argument("--dimensions", default="guidance,locates,hands_over,verdict,leak,correct")
    r.add_argument("--raters", default="")
    r.add_argument("--max-shots", type=int, default=12)
    r.add_argument("--concurrency", type=int, default=24)
    r.add_argument("--retries", type=int, default=2)

    v = sub.add_parser("resolve", help="apply the resolution rule")
    v.add_argument("--units", required=True)
    v.add_argument("--labels", default="data/pilot/labels/agent_*.json")
    v.add_argument("--human", required=True)
    v.add_argument("--dimensions", default="guidance,locates,hands_over,verdict,leak,correct")
    v.add_argument("--out", default="data/pilot/resolved.json")

    args = parser.parse_args()
    if args.cmd == "rate":
        asyncio.run(rate(args))
    else:
        resolve(args)


if __name__ == "__main__":
    main()
