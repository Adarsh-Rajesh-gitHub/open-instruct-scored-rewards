"""Draw the three-part overview figure: what was tried, what worked, what is left.

    python projects/pedagogy_rm/make_overview.py

A SCHEMATIC RATHER THAN A PLOT, which is why it is not in make_eval_figures.py. Nothing here
is computed from data; the numbers are quoted results and the shapes are an argument about how
the parts relate. Keeping it separate means the data figures can be regenerated without anyone
wondering whether this one is stale for the same reason.

WHY THE ABANDONED PART IS DRAWN AT ALL. The first row is a dead end, and a reader who does not
see it will keep asking the obvious question - why not just reward the student for solving the
problem? - through the whole of the second and third rows. The negative result is load-bearing:
it is the reason the reward reads the turn instead of the outcome, and it cost four training
runs to establish. Drawing it greyed out says both things at once, that it was tried and that
it is not part of the current system.
"""

from __future__ import annotations

import argparse
import os

# One row per part. Each is (title, status, boxes, note), where status picks the palette.
PARTS = [
    (
        "1.  Reward the student's answer",
        "abandoned",
        ["tutor turn", "simulated\nstudent", "did the student\nsolve it?", "reward"],
        "Four runs. Leaking fell every time, teaching never moved.\n"
        "Quality vs solving  r = -0.012      Leaking vs solving  r = +0.291\n"
        "Outcome carries one signal, and the only lever on it is giving the answer away.",
    ),
    (
        "2.  Reward the turn itself",
        "done",
        ["tutor turn", "frozen\nOLMo-2-7B", "activations,\nlayer 16", "linear head", "6 scores"],
        "Fitted on 600 human-rated turns. One dot product per turn at training time.\n"
        "targeted:  0.85 from activations against 0.36 from crude text statistics\n"
        "leak hardened: answer-injection attacks fool it 3% of the time, down from 93%.",
    ),
    (
        "3.  Train against it, and check with people",
        "done",
        ["GRPO", "arms A-C\n120 steps", "arms E-G\n200 steps", "blind eval\n240 turns",
         "benchmarks\nbase vs trained"],
        "Arm A claimed +0.36, two independent panels paid +0.34 and +0.35.\n"
        "Length is fixed: arm E has 100% of turns rated right-length, against base's 71%.\n"
        "2x, 4x and 8x the step size all reach the same reward, so the ceiling is the reward model.\n"
        "Academic ability is unchanged: -0.3 points over ARC, MMLU stem and GSM8K pooled.",
    ),
]

PALETTE = {
    # fill, edge, text - abandoned is grey so it reads as history rather than as a component
    "abandoned": ("#ededed", "#9a9a9a", "#5f5f5f"),
    "done": ("#dcefe2", "#3f8f5f", "#1d4d31"),
    "partly done": ("#dde7f5", "#2f6db5", "#173a63"),
}
BADGE = {"abandoned": "#8a8a8a", "done": "#3f8f5f", "partly done": "#2f6db5"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="projects/pedagogy_rm/figures/overview.png")
    args = parser.parse_args()

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(11.5, 7.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Explicit offsets from each row's top, so the three blocks cannot drift into each other
    # when a note gains a line. An earlier version placed the badge at an estimated text width
    # and the separator at a guessed offset; both collided with their neighbours.
    row_top, row_height = 96.0, 31.5
    box_top, box_height, note_top = 6.5, 8.0, 17.0

    for index, (title, status, boxes, note) in enumerate(PARTS):
        fill, edge, ink = PALETTE[status]
        top = row_top - index * row_height

        ax.text(2, top, title, fontsize=13, fontweight="bold", va="top", color="#1a1a1a")
        # Right-aligned, because anything positioned after the title needs the rendered width
        # of the title, which is not knowable before the draw.
        ax.text(
            98, top - 0.3, status.upper(), fontsize=8.5, fontweight="bold",
            va="top", ha="right", color="white",
            bbox={"boxstyle": "round,pad=0.34", "facecolor": BADGE[status], "edgecolor": "none"},
        )

        # Boxes are laid out to fill the width whatever their number, so a row with five
        # stages and a row with four still line up at the margins.
        y = top - box_top
        gap, margin = 3.2, 2.0
        width = (100 - 2 * margin - gap * (len(boxes) - 1)) / len(boxes)
        for position, label in enumerate(boxes):
            x = margin + position * (width + gap)
            # The last box of the unfinished row is dashed: it does not exist yet.
            pending = status == "partly done" and position == len(boxes) - 1
            ax.add_patch(
                FancyBboxPatch(
                    (x, y - box_height), width, box_height,
                    boxstyle="round,pad=0.28,rounding_size=0.9",
                    facecolor="white" if pending else fill,
                    edgecolor=edge, linewidth=1.6,
                    linestyle="--" if pending else "-",
                )
            )
            ax.text(
                x + width / 2, y - box_height / 2, label, fontsize=10.5, ha="center", va="center",
                color=ink, fontweight="bold" if pending else "normal", linespacing=1.35,
            )
            if position < len(boxes) - 1:
                ax.add_patch(
                    FancyArrowPatch(
                        (x + width + 0.35, y - box_height / 2), (x + width + gap - 0.35, y - box_height / 2),
                        arrowstyle="-|>", mutation_scale=13, linewidth=1.5, color=edge,
                    )
                )

        ax.text(2, top - note_top, note, fontsize=9.2, va="top", color="#333333", linespacing=1.7)

        if index < len(PARTS) - 1:
            ax.plot([2, 98], [top - row_height + 4.0] * 2, color="#d8d8d8", linewidth=1)

    fig.suptitle(
        "Reading a teaching reward out of a frozen model — three parts",
        fontsize=14.5, fontweight="bold", y=0.965,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
