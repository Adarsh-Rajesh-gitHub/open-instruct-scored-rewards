"""Draw the training-curve figures for the paper from the W&B dump.

    python projects/pedagogy_rm/fetch_history.py --out data/allhist.json   # on the cluster
    python projects/pedagogy_rm/make_training_figures.py --history data/allhist.json

FOUR FIGURES, SPLIT BY WHAT A READER WOULD ASK RATHER THAN BY WHERE THE NUMBERS LIVE: whether the
optimisation was healthy (losses, gradient norm), where the reward went, whether the group signal
survived, and which qualities actually moved.

ROUNDS ARE NOT PLOTTED ON SHARED AXES, which is the one thing to get right here. Arms A-C score a
raw mean of four qualities and top out near 3; arms E-G score five qualities plus a length term
weighted 2.0 and top out near 5. Drawing them on one axis would show a step change between rounds
that is a change of units and nothing else.
"""

from __future__ import annotations

import argparse
import json
import os

ROUND1 = ("A", "B", "C")
LADDER = ("E", "F", "G")
COLOUR = {
    "A": "#1f77b4",
    "B": "#d62728",
    "C": "#e08214",
    "D": "#999999",
    "E": "#2c7fb8",
    "F": "#41ab5d",
    "G": "#c0504d",
}
LABEL = {
    "A": "A · raw mean, lr 1e-5",
    "B": "B · z-scored, lr 1e-5",
    "C": "C · hard band, lr 1e-5",
    "D": "D · crashed at 46",
    "E": "E · lr 2e-5",
    "F": "F · lr 4e-5",
    "G": "G · lr 8e-5",
}
DIMS = ("leak", "targeted", "actionable", "elicits", "correct", "length")
NICE = {
    "leak": "leak  ·  1 = gives nothing away\n(LOWER is better)",
    "targeted": "targeted",
    "actionable": "actionable",
    "elicits": "elicits",
    "correct": "correct",
    "length": "length fit, 0–1",
}
# `leak` is the one quality the reward subtracts, and open-instruct logs each dimension with its
# reward sign already applied - so it arrives as -1.45 rather than 1.45. Plotting it as logged put
# it outside a 1-3 axis and drew an empty panel. Negating it back shows the rating a rater would
# have written, which is the number the axis is labelled for.
FLIP = {"leak": -1.0}


# Arm F ran as three W&B runs, because the partition preempted it twice and each requeue resumed
# from the last checkpoint under a new run id. The segments overlap where a run had progressed past
# its last checkpoint before dying (1-53, 51-76, 71-200), so later segments win on the overlap:
# those steps were computed twice and only the surviving attempt led anywhere.
SEGMENTS = {"F": ("F_try1", "F_try2", "F")}


def rows_for(history: dict, arm: str, where: str) -> list[dict]:
    parts = SEGMENTS.get(arm, (arm,))
    merged: dict[float, dict] = {}
    for part in parts:
        if part in history:
            for row in history[part][where]:
                if isinstance(row.get("_step"), (int, float)):
                    merged[row["_step"]] = row
    return [merged[k] for k in sorted(merged)]


def series(rows: list[dict], key: str) -> tuple[list, list]:
    xs, ys = [], []
    for row in rows:
        value, step = row.get(key), row.get("_step")
        if isinstance(value, (int, float)) and isinstance(step, (int, float)):
            xs.append(step)
            ys.append(value)
    return xs, ys


def smooth(ys: list[float], window: int = 9) -> list[float]:
    """A centred running mean, so the trend is visible under the step-to-step noise.

    The raw series is always drawn too, faintly, rather than replaced: a smoothed curve alone hides
    how noisy these runs are, and the noise is part of what a reader should see when judging whether
    two arms differ.
    """
    if len(ys) < window:
        return ys
    out = []
    half = window // 2
    for i in range(len(ys)):
        lo, hi = max(0, i - half), min(len(ys), i + half + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


def draw_pair(ax, rows, key, arm, raw_alpha=0.16, scale=1.0):
    xs, ys = series(rows, key)
    if not xs:
        return False
    if scale != 1.0:
        ys = [y * scale for y in ys]
    ax.plot(xs, ys, color=COLOUR[arm], alpha=raw_alpha, lw=0.8, zorder=1)
    ax.plot(xs, smooth(ys), color=COLOUR[arm], lw=1.6, label=LABEL[arm], zorder=2)
    return True


def losses(history: dict, path: str) -> None:
    """Is the optimisation healthy, and does the step size show up where it should?"""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # LOG SCALE ON THE TWO PANELS THAT HAVE A TRANSIENT, and linear on the two that do not. Arm G
    # spikes to a gradient norm of 12.8 and a KL term of 0.106 in its first ten steps against
    # medians of 0.033 and 0.0034. On a linear axis that one excursion flattens all three arms into
    # the x-axis for the remaining 190 steps, so the panel shows only the spike. On a log axis both
    # are legible at once, and the spike is worth seeing rather than clipping away: it is the only
    # sign that 8e-5 starts unstably before recovering.
    panels = [
        ("loss/policy_avg", "policy loss", "step", False),
        ("loss/kl_avg", "KL term of the loss (log)", "step", True),
        ("loss/total_avg", "total loss", "step", False),
        ("optim/grad_norm", "gradient norm (log)", "optim", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.4))
    for ax, (key, title, where, logy) in zip(axes.ravel(), panels, strict=True):
        for arm in LADDER:
            if arm in history:
                draw_pair(ax, rows_for(history, arm, where), key, arm)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        if logy:
            ax.set_yscale("log")
        else:
            ax.axhline(0, color="#888", lw=0.7, zorder=0)
    axes[0][1].annotate(
        "arm G's start is unstable\nbefore it settles",
        xy=(10, 0.09),
        xytext=(48, 0.045),
        fontsize=7,
        color="#8a3530",
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#8a3530"},
    )
    axes[0][0].legend(fontsize=7.4, loc="best", framealpha=0.9)
    fig.suptitle("The learning-rate ladder is healthy at every rate, and differs only in scale", fontsize=10, y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    print(f"wrote {path}")


def rewards(history: dict, path: str) -> None:
    """Where the reward went — one panel per reward definition, never two on one axis.

    A PANEL IS A REWARD FUNCTION, and that is the whole layout decision. Three things changed
    between arms that all show up as "reward": arm A averages four qualities and tops out at 3; arm
    C averages five and adds a length term, topping out at 5; arms E-G use that same form with a
    different length band. Putting any two of them on a shared axis makes a change of units look
    like a change of quality, and the tempting misreading - that E's 3.83 beats C's 3.33 - is not
    supported, because they are scored against different length bands.

    ARM B IS ANNOTATED RATHER THAN PLOTTED. Its reward is z-scored per dimension, so the group mean
    is zero at every step by construction and the curve is a flat line at zero. That is not a
    failed run, it is what the arm was, and a flat line in a legend invites the opposite reading.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    panels = [
        (("A",), "Arm A · mean of 4 qualities\n(max 3.0)"),
        (("C",), "Arm C · 5 qualities + 2.0 × length,\nhard band 30–58 (max 5.0)"),
        (LADDER, "Arms E–G · same form, band 18–40\nwith ramps (max 5.0)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4))
    for ax, (arms, title) in zip(axes, panels, strict=True):
        labelled = False
        for arm in arms:
            if arm not in history or not draw_pair(ax, rows_for(history, arm, "step"), "scores", arm):
                continue
            # Held-out questions on the same axis: the gap between the two is the only in-run
            # evidence about whether the reward is being earned or exploited.
            prefix = history[arm]["prefix"]
            ex, ey = series(rows_for(history, arm, "eval"), f"eval/scored/{prefix}/reward")
            if not ex:
                ex, ey = series(rows_for(history, arm, "eval"), "eval/scores")
            if ex:
                # One legend entry for the marker style, not one per arm: three arms sharing a
                # panel would otherwise list "held out" three times.
                ax.plot(
                    ex,
                    ey,
                    "o",
                    ms=3.4,
                    color=COLOUR[arm],
                    mfc="white",
                    mew=1.2,
                    zorder=3,
                    label=None if labelled else "held out",
                )
                labelled = True
        ax.set_title(title, fontsize=8.6)
        ax.set_xlabel("step", fontsize=8.5)
        ax.legend(fontsize=7.2, loc="lower right", framealpha=0.9)
    axes[0].set_ylabel("reward", fontsize=8.5)
    axes[0].text(
        0.04,
        0.94,
        "arm B is the same run with each\nquality z-scored, so its reward is\n0.000 at every step by construction",
        transform=axes[0].transAxes,
        fontsize=6.8,
        va="top",
        color="#8a3530",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fdf2f2", "edgecolor": "#d8b0b0"},
    )
    fig.suptitle(
        "Values are comparable within a panel and not across them: the reward changed between arms",
        fontsize=9,
        y=1.005,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


def health(history: dict, path: str) -> None:
    """How far the policy travelled, what it did to length, and the schedule that drove it.

    THREE PANELS, NOT SIX, because the other three candidates are constant. Clipped-token fraction
    is exactly 0, importance-ratio variance is 1e-15, and the mean advantage is 1e-17, all for the
    same reason: with one mini-batch and one epoch the update is strictly on-policy, so the
    importance ratio is 1 by construction and there is nothing to clip. The fraction of groups with
    zero advantage - all 16 samples scoring alike, which would mean no gradient - is also 0
    throughout. Those are four pieces of good news and they belong in a sentence; drawn as curves
    they are four flat lines that crowd out the panels carrying signal.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    panels = [
        ("objective/kl1_avg", "KL from the starting policy (nats)", "step"),
        ("scored/pedagogy/words", "tutor turn length (words)", "step"),
        ("lr", "learning rate (3% warmup, then constant)", "step"),
    ]
    for ax, (key, title, where) in zip(axes, panels, strict=True):
        for arm in LADDER:
            if arm in history:
                draw_pair(ax, rows_for(history, arm, where), key, arm)
        ax.set_title(title, fontsize=8.8)
        ax.set_xlabel("step", fontsize=8.5)
    # The band is the target the length term rewards, so showing where it sits turns the middle
    # panel from "length fell" into "length fell into the band and stopped".
    axes[1].axhspan(18, 40, color="#4a7ebb", alpha=0.12, zorder=0)
    axes[1].text(196, 41.5, "rewarded band, 18–40 words", ha="right", fontsize=7, color="#33608f")
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=7.4, loc="lower right", framealpha=0.9)
    fig.suptitle(
        "Higher rates travel further from the starting policy for the same reward — the cost of the ladder",
        fontsize=9.4,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


def dimensions(history: dict, path: str) -> None:
    """Which qualities moved, and which were already saturated."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.0))
    for ax, dim in zip(axes.ravel(), DIMS, strict=True):
        for arm in LADDER:
            if arm not in history:
                continue
            prefix = history[arm]["prefix"]
            draw_pair(ax, rows_for(history, arm, "step"), f"scored/{prefix}/dim_{dim}", arm, scale=FLIP.get(dim, 1.0))
        ax.set_title(NICE[dim], fontsize=8.6)
        ax.set_xlabel("step", fontsize=8)
        # The rating scale is 1-3 for the qualities and 0-1 for the length term, so a shared y
        # limit would flatten one of them into a line.
        if dim != "length":
            ax.set_ylim(0.9, 3.1)
            ax.axhline(1 if dim in FLIP else 3, color="#888", lw=0.7, ls=":", zorder=0)
    axes[0][0].legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.suptitle(
        "Length and leak are the only qualities with room to move; the other four start near the top",
        fontsize=10,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(path)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", default="data/allhist.json")
    parser.add_argument("--figures", default="projects/pedagogy_rm/figures")
    args = parser.parse_args()

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.alpha": 0.22, "figure.dpi": 150})

    with open(args.history) as handle:
        history = json.load(handle)
    os.makedirs(args.figures, exist_ok=True)
    losses(history, os.path.join(args.figures, "loss_curves.png"))
    rewards(history, os.path.join(args.figures, "reward_all_arms.png"))
    health(history, os.path.join(args.figures, "training_health.png"))
    dimensions(history, os.path.join(args.figures, "dimension_curves.png"))


if __name__ == "__main__":
    main()
