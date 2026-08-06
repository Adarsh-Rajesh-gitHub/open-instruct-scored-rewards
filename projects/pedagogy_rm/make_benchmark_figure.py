"""Draw the academic-ability comparison.

    python projects/pedagogy_rm/make_benchmark_figure.py

TWO PANELS, BECAUSE THEY ANSWER DIFFERENT QUESTIONS AND NEITHER SUBSTITUTES FOR THE OTHER.

The lower panel is the result: the paired difference from base with its interval. That is the only
place the null can be read as *bounded* rather than merely unrefuted, and it is why differences are
plotted rather than raw accuracies - at 42% and 84% the raw bars are visually identical whatever the
difference between them is.

The upper panel is the context, and this figure was wrong without it. A difference of -0.3 points
means something quite different at 42% than at 84%, and a reader looking only at differences cannot
tell whether the benchmark was near chance, near saturation, or in between. MMLU stem at 42.5% is
close enough to the 25% floor that little of the scale is in play; GSM8K at 83.8% has little headroom
left. Both facts change how much a small null is worth, and neither is visible in a difference plot.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

TITLE = {"arc_challenge": "ARC-Challenge\n(science)", "mmlu_stem": "MMLU stem\n(school maths/science)",
         "gsm8k": "GSM8K\n(grade-school maths)", "pooled": "all three\npooled"}
SHORT = {"arc_challenge": "ARC-Challenge", "mmlu_stem": "MMLU stem", "gsm8k": "GSM8K",
         "pooled": "all three pooled"}
COLOUR = {"arm_e": "#4a7ebb", "arm_g": "#c0504d"}
LABEL = {"arm_e": "arm E (lr 2e-5)", "arm_g": "arm G (lr 8e-5)"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="data/benchmarks")
    parser.add_argument("--figures", default="projects/pedagogy_rm/figures")
    args = parser.parse_args()

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})

    runs = {}
    for path in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        with open(path) as handle:
            blob = json.load(handle)
        runs[blob["tag"]] = blob["results"]

    tasks = [t for t in ("arc_challenge", "mmlu_stem", "gsm8k") if t in runs["base"]]
    arms = [a for a in ("arm_e", "arm_g") if a in runs]

    def paired(a: list[int], b: list[int]) -> tuple[float, float]:
        diffs = [y - x for x, y in zip(a, b, strict=True)]
        n = len(diffs)
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
        return mean, 1.96 * math.sqrt(var / n)

    # Stacked rather than side by side so the figure is taller than the text block is wide; laid out
    # in a row it prints at 0.7 scale and the tick labels arrive at 6pt.
    fig, (top, ax) = plt.subplots(2, 1, figsize=(7.0, 5.9),
                                  gridspec_kw={"height_ratios": [1.0, 1.15]})

    # ---- upper panel: where the model actually sits ----
    models = ["base", *arms]
    shade = {"base": "#9a9a9a", **COLOUR}
    name = {"base": "base (untrained)", **LABEL}
    bar = 0.26
    for j, tag in enumerate(models):
        xs = [i + (j - 1) * bar for i in range(len(tasks))]
        ys = [runs[tag][task]["accuracy"] * 100 for task in tasks]
        top.bar(xs, ys, bar, label=name[tag], color=shade[tag])
        for x, y in zip(xs, ys, strict=True):
            top.text(x, y + 1.4, f"{y:.1f}", ha="center", fontsize=6.8, color="#333")
    # The floor matters for reading the panel: 25% is what guessing scores on the two
    # multiple-choice benchmarks, so MMLU stem's 42.5% uses much less of the scale than it appears to.
    top.axhline(25, color="#b0453f", lw=0.9, ls="--")
    top.set_xticks(range(len(tasks)))
    top.set_xticklabels([TITLE[t] for t in tasks], fontsize=8)
    # Headroom to 118 with ticks stopping at 100, so the legend sits above the tallest bar instead
    # of on top of it - GSM8K reaches 83.8 and a full-width three-column legend covered its label.
    top.set_ylim(0, 118)
    top.set_yticks([0, 20, 40, 60, 80, 100])
    top.set_ylabel("accuracy (%)")
    # The chance level is explained here rather than beside the line it marks: every gap between bar
    # groups is narrower than the label, so an in-panel annotation ended up printed over the ARC bar.
    top.set_title("Where the models sit: absolute accuracy, 400 items per benchmark\n"
                  "dashed line is 25%, what guessing scores on the two multiple-choice sets",
                  fontsize=8.8)
    top.legend(loc="upper left", fontsize=7.6, framealpha=0.9, ncol=3)

    # ---- lower panel: the result ----
    columns = [*tasks, "pooled"]
    width = 0.36
    for j, arm in enumerate(arms):
        centres, means, errs = [], [], []
        pooled_a: list[int] = []
        pooled_b: list[int] = []
        for i, task in enumerate(tasks):
            a = runs["base"][task]["per_item"]
            b = runs[arm][task]["per_item"]
            pooled_a += a
            pooled_b += b
            mean, err = paired(a, b)
            centres.append(i + (j - 0.5) * width)
            means.append(mean * 100)
            errs.append(err * 100)
        mean, err = paired(pooled_a, pooled_b)
        centres.append(len(tasks) + (j - 0.5) * width)
        means.append(mean * 100)
        errs.append(err * 100)
        ax.bar(centres, means, width, yerr=errs, capsize=3, label=LABEL[arm],
               color=COLOUR[arm], error_kw={"lw": 1, "ecolor": "#333"})

    ax.axhline(0, color="#333", lw=1)
    # No "a drop this big would matter" band here, though one was tried. Shading ±2 points covered
    # most of the panel and so read as decoration, and worse, it implied every interval excluded a
    # two-point drop when the per-task ones do not - GSM8K alone reaches -3.6. Only the pooled
    # column is tight enough to support that claim, so the figure shows the intervals and lets
    # them speak instead of drawing a threshold they do not all clear.
    #
    # Arm E on ARC is annotated because a zero-height bar with no interval looks like missing data
    # rather than like the result it is: not one item of 400 changed hands.
    for j, arm in enumerate(arms):
        a, b = runs["base"]["arc_challenge"]["per_item"], runs[arm]["arc_challenge"]["per_item"]
        if sum(1 for x, y in zip(a, b, strict=True) if x != y) == 0:
            ax.annotate("identical\non all 400", xy=((j - 0.5) * width, 0), xytext=(0, 22),
                        textcoords="offset points", ha="center", fontsize=7.5, color="#333",
                        arrowprops={"arrowstyle": "-", "lw": 0.8, "color": "#666"})
    ax.set_xticks(range(len(columns)))
    # Single-line labels here: four two-line labels collide at this width, and the upper panel has
    # already said what each benchmark is, so repeating the descriptions costs space for nothing.
    ax.set_xticklabels([SHORT[c] for c in columns], fontsize=8.4)
    ax.set_ylabel("change in accuracy vs base (points)")
    ax.set_title("The result: paired change against base\n"
                 "(same items for every arm; bars are 95% intervals)", fontsize=9.2)
    ax.legend(loc="lower left", fontsize=7.6, framealpha=0.9)
    fig.suptitle("Academic ability after 200 steps of tutoring-shaped RL", fontsize=10.5,
                 fontweight="bold", y=0.995)
    fig.tight_layout()
    os.makedirs(args.figures, exist_ok=True)
    out = os.path.join(args.figures, "academic_ability.png")
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
