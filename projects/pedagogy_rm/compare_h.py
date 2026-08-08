"""Arm H against arm E at matched steps: what happens to length with no length term.

    python projects/pedagogy_rm/compare_h.py          # on a machine that can reach wandb

Arm H is arm E with the length term removed and nothing else changed. The reward that remains
correlates -0.60 with log word count, so the prediction is that length falls and the five probe
dimensions rise past where arm E left them. This prints both arms at the same steps so the
divergence is visible while the run is still going.
"""

from __future__ import annotations

import argparse

RUNS = {"H": "npcdu27n", "E": "118zudzw"}
DIMS = ("leak", "targeted", "actionable", "elicits", "correct")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity-project", default="eduLLM/pedagogy-rm")
    args = parser.parse_args()

    import wandb  # noqa: PLC0415

    api = wandb.Api()
    keys = ["_step", "scores", "scored/pedagogy/words", "objective/kl1_avg"] + [
        f"scored/pedagogy/dim_{d}" for d in DIMS
    ]

    data, states = {}, {}
    for tag, rid in RUNS.items():
        run = api.run(f"{args.entity_project}/{rid}")
        states[tag] = run.state
        have = set(run.summary.keys())
        ask = [k for k in keys if k == "_step" or k in have]
        data[tag] = {r["_step"]: r for r in run.scan_history(keys=ask)}

    hi = max(data["H"]) if data["H"] else 0
    print(f"arm H: {states['H']}, {hi} steps    arm E: {states['E']}, {max(data['E'])} steps")

    def mean5(row):
        # leak arrives already negated, the other four positive: this is what GRPO optimises.
        vals = [row.get(f"scored/pedagogy/dim_{d}") for d in DIMS]
        return sum(vals) / 5 if all(isinstance(v, (int, float)) for v in vals) else None

    print()
    # KL is here because it decides whether the comparison is readable at all. Dropping the length
    # term removes within-group reward variance, which shrinks the effective step size, so if H has
    # drifted much less than E then it was simply trained more gently and the rest means little.
    print(f"{'step':>5} | {'H words':>8} {'H 5-dim':>8} {'H KL':>7} | {'E words':>8} {'E 5-dim':>8} {'E KL':>7}")
    print("-" * 74)
    for s in [1, *range(10, hi + 1, 10)]:
        h = data["H"].get(s)
        if not h:
            continue
        e = data["E"].get(s)
        hm, em = mean5(h), mean5(e) if e else None
        ew = f"{e['scored/pedagogy/words']:8.1f}" if e else f"{'-':>8}"
        es = f"{em:8.3f}" if em is not None else f"{'-':>8}"
        ek = f"{e.get('objective/kl1_avg', 0):7.3f}" if e else f"{'-':>7}"
        hs = f"{hm:8.3f}" if hm is not None else f"{'-':>8}"
        print(f"{s:>5} | {h['scored/pedagogy/words']:8.1f} {hs} {h.get('objective/kl1_avg', 0):7.3f} | {ew} {es} {ek}")

    # The comparison the run exists to make: arm E finished with a 5-dim mean of 1.852 while its
    # length band held words at 28. If arm H passes 1.852 with far shorter turns, the band was what
    # kept the policy off the degenerate path rather than the probe being out of range.
    last = data["H"][hi]
    hm = mean5(last)
    if hm is not None:
        print()
        print(f"arm H at step {hi}: {last['scored/pedagogy/words']:.1f} words, 5-dim mean {hm:.3f}")
        print("arm E final:        27.9 words, 5-dim mean 1.852")
        print(f"  -> 5-dim gap vs arm E's final: {hm - 1.852:+.3f}")


if __name__ == "__main__":
    main()
