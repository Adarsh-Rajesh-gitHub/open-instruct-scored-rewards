"""Manufacture the bad turns the pool does not contain, by minimally corrupting good ones.

    python projects/pedagogy_rm/build_negatives.py \
        --pool data/pilot/pool.json --labels data/pilot/sophia.json --out data/negatives

WHY THIS IS NECESSARY. Every turn in the sampled pools came from OLMo-2-7B, trained or not, and
it is a competent enough tutor that the bottom of most scales is empty: in the twenty-turn
pilot, `locates` never received a 1 at all and `guidance` received one 5% of the time. A head
fitted on that cannot learn the bottom anchor, and a rater cannot calibrate against a level
they have never seen. This is the same hole `leak` had, and adversarial augmentation is what
closed it - attacks fooled the original leak head 93% of the time and 3% after.

MINIMAL EDITS, NOT GENERATED-FROM-SCRATCH BADNESS, and this is the whole design. A turn written
to be bad is bad in every way at once - wrong register, wrong length, obviously synthetic - and
a head fitted on those learns to detect synthetic text rather than the dimension. Corrupting one
real turn along one dimension holds everything else fixed, which is what makes the label mean
what it says. It is also how the leak augmentation worked: the gold answer written into an
otherwise answer-withholding turn, changing nothing else.

FEW-SHOT FROM THE HUMAN'S OWN LABELS, so the corruption targets what she scores low rather than
what the model imagines is bad. Each target dimension is prompted with her real examples at the
top and bottom of that scale.

THESE STILL NEED CHECKING BY HAND. A corruption that failed to corrupt is a mislabelled
example, which is worse than a missing one, so the output goes to the labelling interface with
its intended score recorded but not trusted.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import random

from projects.pedagogy_rm.rubric import BY_KEY

#: dimension -> (target score, what to do to the turn). The instruction is deliberately narrow:
#: one property changes, everything else is preserved.
CORRUPTIONS = {
    # These two were rewritten after the first batch was rated. Both originally landed only 50%
    # of the time - the model kept enough specificity or content to satisfy the middle anchor,
    # because "no longer points at anything specific" and "strip out content" both leave room to
    # do a partial job. The replacements are stated as hard prohibitions on identifiable material
    # rather than as reductions, which is a test the generator can check its own output against.
    "locates": (
        1,
        "Rewrite it so that NOT ONE number, quantity, step, operation, or content word from the "
        "student's message or the question appears anywhere in it. The result must be something "
        "you could paste unchanged after a wrong answer to a completely different problem in a "
        "different subject, and it must still read as a plausible tutor turn. Keep the same "
        "length. Do not add praise.",
    ),
    "guidance": (
        1,
        "Replace it with a turn from which the student can take away NOTHING about this problem: "
        "no rule, no quantity, no step, no direction, no named concept, not even a hint of which "
        "part to look at. Permitted material is encouragement, a restatement of the question in "
        "other words, a request to try again, or a request to explain their thinking. Keep the "
        "same length - pad with process talk rather than shortening it.",
    ),
    "verdict": (
        1,
        "The student's last message is wrong or partly wrong. Rewrite the tutor turn so it "
        "affirms or praises what the student said as though it were correct, then continues from "
        "it. Keep the same length. The affirmation should sound natural, not cartoonish.",
    ),
    "leak": (
        3,
        "Rewrite it so it states the correct answer outright, or rules out every option but the "
        "correct one. Keep the same length, tone and structure - it should still read like a "
        "tutor turn, not like an answer key.",
    ),
    "correct": (
        1,
        "Change exactly one factual thing so the turn now contradicts the worked solution: a "
        "number, a relationship, an operation, or a claim. Everything else stays identical. The "
        "error should be the kind a confident tutor makes, not an obvious typo.",
    ),
    "hands_over": (
        1,
        "Rewrite it so the tutor does the next step itself instead of handing it to the student. "
        "Keep the same length and content - the reasoning that was being asked for is now simply "
        "performed by the tutor.",
    ),
}

PROMPT = """You are building deliberately flawed tutoring turns, to teach a rating model what a
low score looks like. This is for evaluation data, so the flaw must be specific and everything
else must stay as it was.

THE DIMENSION BEING CORRUPTED: {key} — {question}
{anchors}

HOW THIS RATER SCORES IT, from their own labels:
{shots}

YOUR EDIT: {instruction}

Return ONLY the rewritten tutor turn. No preamble, no explanation, no quotation marks.

=== THE CASE ===
QUESTION: {q}
{reference}STUDENT SAID: {student}
TUTOR TURN TO CORRUPT: {tutor}"""


def render_shots(labels: list[dict], units: dict, key: str, low: int, limit: int = 3) -> str:
    """Her own turns at the bottom and top of this scale, so the target is hers not ours."""
    hi = BY_KEY[key].hi if low == BY_KEY[key].lo else BY_KEY[key].lo
    out = []
    for want, tag in ((low, f"SCORED {low} (what we are aiming for)"), (hi, f"SCORED {hi}")):
        got = [r for r in labels if r.get(key) == want and r["id"] in units][:limit]
        for r in got:
            out.append(f"--- {tag} ---\n{' '.join(units[r['id']]['tutor_turn'].split())}")
    return "\n".join(out) or "(no examples at either end; rely on the anchors)"


async def main_async(args) -> None:
    from projects.pedagogy_rm import gateway  # noqa: PLC0415

    with open(args.pool) as h:
        blob = json.load(h)
    units = {u["id"]: u for u in blob["units"]}
    with open(args.labels) as h:
        labels = json.load(h)["labels"]
    by_id = {r["id"]: r for r in labels}

    wanted = [k for k in args.dimensions.split(",") if k]
    rng = random.Random(args.seed)
    client = gateway.make_client()

    async def one(key: str, unit: dict) -> dict | None:
        target, instruction = CORRUPTIONS[key]
        dim = BY_KEY[key]
        anchors = "\n".join(f"  {s} = {t}" for s, t in sorted(dim.anchors.items()))
        ref = f"WORKED SOLUTION:\n{unit['reference']}\n" if unit.get("reference") else ""
        prompt = PROMPT.format(
            key=key, question=dim.question, anchors=anchors,
            shots=render_shots(labels, units, key, target),
            instruction=instruction, q=" ".join(unit["question"].split()),
            reference=ref, student=" ".join((unit.get("student_before") or "").split()),
            tutor=" ".join(unit["tutor_turn"].split()),
        )
        reply = await client.chat.completions.create(
            model=args.model, messages=[{"role": "user", "content": prompt}], temperature=0.7
        )
        text = (reply.choices[0].message.content or "").strip().strip('"')
        if not text or text == unit["tutor_turn"].strip():
            return None
        return dict(unit, id=f"n{key[:4]}{unit['id'][1:9]}", tutor_turn=text,
                    corrupted=key, intended=target, source_id=unit["id"])

    jobs = []
    for key in wanted:
        target, _ = CORRUPTIONS[key]
        # Corrupt turns that currently sit AWAY from the target, so the edit has somewhere to go.
        pool = [u for uid, u in units.items()
                if by_id.get(uid, {}).get(key) is not None and by_id[uid][key] != target]
        rng.shuffle(pool)
        jobs += [(key, u) for u in pool[: args.per_dimension]]

    print(f"corrupting {len(jobs)} turns across {len(wanted)} dimensions...")
    made = [r for r in await asyncio.gather(*(one(k, u) for k, u in jobs)) if r]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "pool.json"), "w") as h:
        json.dump({"note": "minimally corrupted turns; `intended` is a hypothesis, not a label",
                   "units": made}, h, indent=1)
    counts = collections.Counter(r["corrupted"] for r in made)
    print(f"\nwrote {args.out}/pool.json — {len(made)} turns")
    for k in wanted:
        target, _ = CORRUPTIONS[k]
        print(f"  {k:<12} {counts.get(k,0):>2} turns, intended {k}={target}")
    print("\nCheck them by hand before trusting `intended` - a corruption that did not corrupt")
    print("is a mislabelled example, which is worse than a missing one.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="data/pilot/pool.json")
    parser.add_argument("--labels", default="data/pilot/sophia.json")
    parser.add_argument("--dimensions", default="locates,guidance,verdict,leak,correct,hands_over")
    parser.add_argument("--per-dimension", type=int, default=4)
    parser.add_argument("--model", default="openai-group/gpt-5.6-terra")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default="data/negatives")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
