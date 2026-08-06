"""Draw the length-reward design: what people accept, and where each candidate settles.

    python projects/pedagogy_rm/make_length_figure.py

The left panel is measurement: 1131 length judgements over 180 turns, binned, with the fitted
two-shoulder curve through them. The right panel is the design question, which is not "what do
people accept" but "where does the policy end up", because the four quality dimensions already
pay -0.33 per log-word and any length term has to be read against that pull.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="data/eval50/pool.json")
    parser.add_argument("--curve", default="data/length_curve.json")
    parser.add_argument("--out", default="projects/pedagogy_rm/figures/length_reward.png")
    parser.add_argument("--slope", type=float, default=-0.33, help="quality reward per log word")
    parser.add_argument("--lam", type=float, default=2.0)
    args = parser.parse_args()

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": 0.25})

    with open(args.pool) as handle:
        units = {u["id"]: u for u in json.load(handle)["units"]}
    with open(args.curve) as handle:
        c = json.load(handle)

    def accept(w: float) -> float:
        lw = math.log(max(w, 1))
        return (1 / (1 + math.exp(-(lw - c["a"]) / c["s_short"]))) * (
            1 / (1 + math.exp(-(c["b"] - lw) / c["s_long"]))
        )

    def binary(w: float) -> float:
        return 1.0 if 30 <= w <= 58 else 0.0

    judgements = []
    for path in sorted(glob.glob("data/eval50/labels/*.json")):
        with open(path) as handle:
            blob = json.load(handle)
        for record in blob.get("labels", []):
            unit = units.get(record["id"])
            if unit and isinstance(record.get("length_fit"), int):
                judgements.append((len(unit["tutor_turn"].split()), record["length_fit"] == 2))

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.1))

    # LEFT: the measurement.
    bins = [(0, 10), (10, 13), (13, 16), (16, 20), (20, 26), (26, 35), (35, 50), (50, 70), (70, 120), (120, 400)]
    xs, ys, ns = [], [], []
    for lo, hi in bins:
        sel = [ok for w, ok in judgements if lo <= w < hi]
        if len(sel) >= 8:
            xs.append(math.sqrt(lo * max(hi, lo + 1)))
            ys.append(statistics.fmean(sel))
            ns.append(len(sel))
    left.scatter(xs, ys, s=[min(n, 220) for n in ns], color="#2f6db5", alpha=0.55,
                 edgecolor="#1a3f6b", zorder=3, label="measured (dot size = judgements)")
    grid = [1.05**i for i in range(0, 130) if 1.05**i <= 300]
    left.plot(grid, [accept(w) for w in grid], color="#c0392b", lw=2.2, zorder=4, label="fitted curve")
    left.axvspan(21, 58, color="#4c9f70", alpha=0.13, zorder=0)
    left.text(35, 0.06, "90%+ acceptable\n21–58 words", ha="center", fontsize=8.5, color="#2c6b4a")
    left.set_xscale("log")
    left.set_xticks([5, 10, 20, 40, 80, 160, 300])
    left.set_xticklabels(["5", "10", "20", "40", "80", "160", "300"])
    left.set_xlabel("words in the tutor turn")
    left.set_ylabel("fraction of raters calling the length right")
    left.set_title("What people accept  (1131 judgements)", fontsize=10.5)
    left.set_ylim(0, 1.05)
    left.legend(loc="lower center", fontsize=8, framealpha=0.95)

    # RIGHT: where the policy settles, which is the actual design question.
    combined = lambda fn, w: args.slope * math.log(w) + args.lam * fn(w)  # noqa: E731
    # The two resting points are close together, so the labels are pushed to opposite sides
    # rather than both sitting above the marker, where they overlapped.
    for name, fn, colour, nudge in (
        ("smooth", accept, "#c0392b", (-48, 24)),
        ("binary 30–58", binary, "#2f6db5", (46, 8)),
    ):
        vals = [combined(fn, w) for w in grid]
        peak = max(vals)
        right.plot(grid, [v - peak for v in vals], color=colour, lw=2.2, label=name)
        best = max(grid, key=lambda w: combined(fn, w))  # noqa: B023
        right.plot([best], [0], "o", color=colour, ms=8, zorder=5)
        right.annotate(f"settles at {best:.0f}w", (best, 0), textcoords="offset points",
                       xytext=nudge, ha="center", fontsize=8.5, color=colour, fontweight="bold")
    quality_only = [args.slope * math.log(w) for w in grid]
    peak = max(quality_only)
    right.plot(grid, [v - peak for v in quality_only], color="#888888", lw=1.8, ls="--",
               label="no length term (what we ran)")
    for w, label, colour in ((17, "arm A sits here", "#888888"), (34, "base", "#888888")):
        right.axvline(w, color=colour, lw=1, ls=":", alpha=0.7)
        right.text(w, -2.55, label, rotation=90, fontsize=7.5, color="#555555", va="bottom", ha="right")
    right.set_xscale("log")
    right.set_xticks([5, 10, 20, 40, 80, 160, 300])
    right.set_xticklabels(["5", "10", "20", "40", "80", "160", "300"])
    right.set_ylim(-2.6, 0.62)
    right.set_xlabel("words in the tutor turn")
    right.set_ylabel("total reward, relative to its own best")
    right.set_title(f"Where the policy settles  (quality pays {args.slope} per log-word)", fontsize=10.5)
    right.legend(loc="lower left", fontsize=8, framealpha=0.95)

    fig.suptitle("Designing the length reward", fontsize=12.5, fontweight="bold", y=1.0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
