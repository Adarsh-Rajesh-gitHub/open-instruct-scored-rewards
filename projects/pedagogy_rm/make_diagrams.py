"""Draw the two explanatory diagrams: how the reward model works, and what each arm tested.

    python projects/pedagogy_rm/make_diagrams.py

These are hand-laid rather than generated from the code, which is a deliberate trade: a diagram
derived from the call graph would show every module and hide the one idea. What a reader needs from
the first is that the encoder is frozen and the same weights appear twice - once to fit the heads,
once to score rollouts - and from the second, which single thing changed between consecutive arms.
"""

from __future__ import annotations

import argparse
import os

INK = "#222222"
MUTED = "#6b6b6b"


def box(ax, x, y, w, h, text, face="#ffffff", edge=INK, size=8.0, weight="normal", lw=1.1, radius=0.02):
    from matplotlib.patches import FancyBboxPatch  # noqa: PLC0415

    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.004,rounding_size={radius}",
            linewidth=lw,
            edgecolor=edge,
            facecolor=face,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=INK,
        weight=weight,
        zorder=3,
        linespacing=1.35,
    )


def arrow(ax, start, end, text="", style="-|>", colour=INK, size=7.2, lw=1.1, rad=0.0, off=(0, 0), dashed=False):
    from matplotlib.patches import FancyArrowPatch  # noqa: PLC0415

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=11,
            linewidth=lw,
            color=colour,
            zorder=4,
            linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if text:
        mx, my = (start[0] + end[0]) / 2 + off[0], (start[1] + end[1]) / 2 + off[1]
        ax.text(
            mx,
            my,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color=colour,
            zorder=5,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none"},
        )


def system(path: str) -> None:
    """The reward model, and the loop it sits inside.

    Laid out in two rows because the frozen encoder does two different jobs and conflating them is
    the thing readers get wrong: on top it is a measuring instrument fitted once against human
    ratings, underneath it is a scorer called on every rollout. Same weights, never updated, which
    is what makes the reward stationary while the policy moves.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # 7.4in rather than 9.6in wide so that fitting it to a 6.5in text block barely shrinks it: at
    # 9.6in the labels arrive on the page at 5.4pt, which is below what prints legibly.
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")

    # ---- Row 1: fitting the reward model, done once ----
    # LABELS ARE SHORT AND FONTS SMALL BECAUSE THE FIGURE IS NARROW. At 9.6in wide the longer
    # wording fitted, but the figure then printed at 0.68 scale and arrived at 5.4pt. Narrowing it
    # to 7.4in fixed the print size and broke the labels instead - "OLMo-2-7B-Instruct" in bold at
    # 8pt ran straight through its box edge. Both constraints are real, so the text gives way.
    ax.text(1, 55.5, "ONCE  ·  fit the reward model against human ratings", fontsize=9, weight="bold", color=INK)
    box(ax, 1, 40, 16, 11.5, "600 tutor turns\n300 questions\n4 prompt styles", face="#f2f2f2", size=7.2)
    box(ax, 20.5, 40, 15, 11.5, "human + model\nratings on 5\nqualities, 1–3", face="#fdf0d5", size=7.2)
    box(ax, 39, 40, 17, 11.5, "OLMo-2-7B\nInstruct, FROZEN\none forward pass", face="#dbe7f3", weight="bold", size=7.2)
    box(ax, 59.5, 45.6, 17, 5.9, "hidden state,\nlayer 16, mean", face="#ffffff", size=7.0)
    box(ax, 59.5, 39.4, 17, 5.9, "ridge: one head\nper quality", face="#e8f0e4", size=7.0)
    box(ax, 80, 40, 18, 11.5, "head5.npz\n5 linear heads\n+ length band", face="#ffffff", weight="bold", size=7.2)

    arrow(ax, (17.2, 45.7), (20.3, 45.7))
    arrow(ax, (35.7, 45.7), (38.8, 45.7))
    arrow(ax, (56.2, 45.7), (59.3, 48.5))
    arrow(ax, (68, 45.5), (68, 45.4))
    arrow(ax, (76.7, 42.3), (79.8, 45))

    # ---- Row 2: the training loop, every step ----
    ax.text(1, 26.5, "EVERY STEP  ·  train the tutor with GRPO", fontsize=9, weight="bold", color=INK)
    box(ax, 1, 12, 16, 10.5, "a question and\nthe dialogue\nso far", face="#f2f2f2", size=7.2)
    box(ax, 20.5, 12, 17, 10.5, "policy\nOLMo-2-7B\n+ LoRA, r=32", face="#f6e2e2", weight="bold", size=7.2)
    box(ax, 41, 12, 15, 10.5, "16 candidate\ntutor turns\n(one group)", face="#ffffff", size=7.2)
    box(ax, 59.5, 12, 17, 10.5, "frozen encoder\n+ 5 heads\n→ 5 ratings", face="#dbe7f3", size=7.2)
    box(ax, 80, 12, 18, 10.5, "reward =\nmean of 5\n+ 2.0 × length", face="#e8f0e4", weight="bold", size=7.2)

    arrow(ax, (17.2, 17.2), (20.3, 17.2))
    arrow(ax, (37.7, 17.2), (40.8, 17.2), "sample", off=(0, 2.0), size=6.6)
    arrow(ax, (56.2, 17.2), (59.3, 17.2))
    arrow(ax, (76.7, 17.2), (79.8, 17.2))

    # The reuse arrow lands on the row-2 encoder rather than stopping in the gap, because the
    # single claim this diagram exists to make is that these two boxes are the same weights.
    arrow(
        ax,
        (47.5, 39.6),
        (68, 22.8),
        "the same frozen weights,\nnever updated",
        colour="#b0453f",
        lw=1.6,
        rad=-0.16,
        off=(-9.5, 2.4),
        size=7.6,
    )

    # the update path, drawn back along the bottom
    arrow(ax, (89, 11.8), (29, 6.2), "", rad=-0.10, colour="#b0453f", lw=1.5)
    ax.text(
        58,
        4.2,
        "advantage = reward − group mean   →   update LoRA only   (KL penalty β = 0.02)",
        ha="center",
        fontsize=8.2,
        color="#b0453f",
        weight="bold",
    )
    arrow(ax, (29, 6.2), (28.9, 11.8), colour="#b0453f", lw=1.5)

    ax.text(
        99,
        0.6,
        "nothing outside the LoRA adapter is ever updated",
        ha="right",
        fontsize=7.4,
        color=MUTED,
        style="italic",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path}")


# Every string is wrapped to fit the column it is drawn in. An earlier version wrote these as one
# or two long lines and the text ran straight out through the box edges and over its neighbours,
# which is worse than a diagram with less in it.
ARMS = [
    (
        "A",
        "first run\nraw mean of\n4 qualities",
        "does the reward\nmove at all?",
        "reward 0.83 → 1.62,\nbut turns collapsed\nto 12 words",
        "#c9d9ea",
    ),
    (
        "B",
        "z-score each\nquality",
        "do unequal spreads\ndistort it?",
        "identical to A —\nscalarisation was\nnot the problem",
        "#c9d9ea",
    ),
    (
        "C",
        "hard length\nband, 30–58\nwords",
        "can length be\nfixed by a band?",
        "median 46 words,\nbut 29% now too\nshort: no floor",
        "#f5dfc0",
    ),
    (
        "D",
        "5th quality\n(correct)\n+ linear ramps",
        "does a ramp beat\na hard edge?",
        "crashed at step 46\n(preempted before\nfirst checkpoint)",
        "#e8e8e8",
    ),
    (
        "E",
        "lr 2e-5,\n200 steps\nband 18–40",
        "the measured-best\nstep size",
        "reward 3.84, KL 0.25,\n100% right length,\nhuman prefers 29–11",
        "#cfe3cd",
    ),
    (
        "F",
        "lr 4e-5",
        "does 2× lr change\nthe ceiling?",
        "reward 3.88, KL 0.29 —\nsame place in\nhalf the steps",
        "#cfe3cd",
    ),
    (
        "G",
        "lr 8e-5",
        "does 4× lr change\nthe ceiling?",
        "reward 3.86, KL 0.29 —\nsame place, and no\nfurther drift than F",
        "#cfe3cd",
    ),
]


def ablations(path: str) -> None:
    """What each arm changed, what it was asking, and what came back.

    ONE ROW PER ARM, WHICH IS A LEGIBILITY CONSTRAINT RATHER THAN A PREFERENCE. Laid out as seven
    columns the figure is 2.7 times wider than tall, so fitting it to a portrait text width scales
    it to 0.47 and turns 8pt labels into 3.8pt. Rows make the aspect ratio taller than the text
    block instead, so it prints at roughly full size. Reading order stays top-to-bottom, which also
    matches the fact that the arms were sequential - each was designed after reading the previous
    one - where a tree would imply they were planned together.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    n = len(ARMS)
    fig, ax = plt.subplots(figsize=(7.4, 8.3))
    ax.set_xlim(0, 100)
    top = n * 13 + 10
    # Negative lower bound reserves room for the footer; at 0 it was drawn on top of arm G's row.
    ax.set_ylim(-11, top)
    ax.axis("off")

    ax.text(11, top - 4.4, "what changed", fontsize=8.4, weight="bold", color=MUTED)
    ax.text(40, top - 4.4, "what it asked", fontsize=8.4, weight="bold", color=MUTED)
    ax.text(69, top - 4.4, "what came back", fontsize=8.4, weight="bold", color=MUTED)

    for i, (name, changed, asked, got, shade) in enumerate(ARMS):
        y = top - 7.5 - (i + 1) * 13 + 2.2
        ax.text(4.5, y + 5.4, name, ha="center", va="center", fontsize=13, weight="bold", color=INK)
        box(ax, 10, y, 27, 10.8, changed, face=shade, size=8.0)
        box(ax, 39, y, 27, 10.8, asked, face="#ffffff", size=7.8, edge=MUTED, lw=0.9)
        box(ax, 68, y, 31, 10.8, got, face="#fafafa", size=7.8, edge=MUTED, lw=0.9)
        if i < n - 1:
            arrow(ax, (4.5, y - 0.4), (4.5, y - 2.0), lw=1.2, colour=MUTED)

    # Called out because it is the reason the arms stop here: once the ceiling is known to belong to
    # the reward model, further optimiser tuning cannot move it.
    ax.text(
        50,
        -6.5,
        "E, F and G land in the same place at 2×, 4× and 8× the step size:\n"
        "the optimiser sets how fast, the reward model sets how far.",
        ha="center",
        fontsize=8.8,
        color="#b0453f",
        weight="bold",
        linespacing=1.4,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figures", default="projects/pedagogy_rm/figures")
    args = parser.parse_args()
    os.makedirs(args.figures, exist_ok=True)
    system(os.path.join(args.figures, "system.png"))
    ablations(os.path.join(args.figures, "ablations.png"))


if __name__ == "__main__":
    main()
